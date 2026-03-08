import json
import logging
import uuid
from pathlib import Path

import anthropic

from .protocols import ClusterMapping, GraphCreatorFn, Highlights

logger = logging.getLogger(__name__)

_LOCAL_LLM_SYSTEM_PROMPT = """\
You are a knowledge graph architect. You will receive clusters of highlights from a book or article.
Produce a JSON Canvas knowledge graph that compresses and connects the information.

Rules:
- Each cluster becomes one group node with a short label (2-4 words).
- Inside each group create 1-4 text nodes. Each text node is a compressed concept.
- Add edges between text nodes when a meaningful relationship exists.
- Use edge labels of 1-3 words (e.g. "enables", "contrasts with").
- Node colors: "6" for core concepts, "2" for examples, "4" for caveats, "5" for actionable tips.

Output ONLY a JSON object. No markdown fences. No commentary. Example structure:
{"nodes":[{"id":"1","x":0,"y":0,"width":400,"height":120,"type":"group","label":"Topic"},{"id":"2","x":20,"y":40,"width":360,"height":80,"type":"text","text":"Key idea","color":"6"}],"edges":[{"id":"3","fromNode":"2","fromSide":"bottom","toNode":"4","toSide":"top","label":"enables"}]}

Layout: place groups in a row, 500px apart horizontally. Each group is 400px wide.
Text nodes are 360x80px, stacked vertically inside the group with 20px padding.
"""

_SYSTEM_PROMPT = """\
You are a knowledge graph architect. You will receive clusters of highlights from a book or article.
Your job is to produce a JSON Canvas knowledge graph that compresses and connects the information.

Rules:
- Each cluster becomes one group node with a short, descriptive label (2-4 words).
- Inside each group create 1-4 text nodes. Each text node is a compressed concept from that cluster.
  Compress multiple related highlights into a single concise bullet-point summary where possible.
- Add edges between text nodes across (and within) groups when a meaningful conceptual relationship exists.
  Use an edge label of 1-3 words that names the relationship (e.g. "enables", "explains", "contrasts with").
- Node colors (optional): use "6" (purple) for core/foundational concepts, "2" (green) for examples,
  "4" (red) for caveats, "5" (pink) for actionable advice. Omit color for neutral nodes.
- Keep the graph informative and non-redundant. Prefer depth over breadth.

Output ONLY a JSON object with two keys: "nodes" and "edges".
Do not output any markdown fences or commentary. Make sure the output is valid JSON. This means using double quotes (not smart quotes) for all strings, including keys and values.
Don't make the json pretty-printed (i.e. no newlines or indentation), but it must be parseable by a standard JSON parser.

Node schema (group):
  {"id": "<uuid>", "x": <int>, "y": <int>, "width": <int>, "height": <int>,
   "type": "group", "label": "<cluster label>"}

Node schema (text):
  {"id": "<uuid>", "x": <int>, "y": <int>, "width": <int>, "height": <int>,
   "type": "text", "text": "<content>", "color": "<1-6>"}

Edge schema:
  {"id": "<uuid>", "fromNode": "<node_id>", "fromSide": "bottom",
   "toNode": "<node_id>", "toSide": "top", "label": "<relationship>"}

Layout: place groups in a horizontal row. Each group is 400px wide with 100px gaps between groups.
Each text node inside a group is 360px wide and 80px tall, padded 20px from the group left edge,
stacked vertically with 20px padding from the top and 10px between nodes.
Groups are tall enough to fit their text nodes with 20px padding at the bottom.
"""

_EXAMPLES_DIR = Path(__file__).parents[2] / "examples"

with open(_EXAMPLES_DIR / "How We Learn - Benedict Carey.canvas", "r") as f:
    _EXAMPLE_CANVAS = json.load(f)

with open(_EXAMPLES_DIR / "How We Learn - Benedict Carey.md", "r") as f:
    _EXAMPLE_HIGHLIGHTS = f.read()

_SYSTEM_PROMPT += f"\n\nHere is an example of good output. The input highlights are delimited by triple backticks, and the corresponding JSON Canvas graph is shown after that. For the output JSON, note how the nodes and edges belong in their own list:\n\n```\n{_EXAMPLE_HIGHLIGHTS}\n```\n\nExample output:\n\n{json.dumps(_EXAMPLE_CANVAS, indent=2)}\n"


def _build_user_message(cluster_mapping: ClusterMapping, highlights: Highlights) -> str:
    lines = []
    for cluster_key, indices in cluster_mapping.items():
        lines.append(f"=== Cluster {cluster_key} ===")
        for idx in sorted(indices):
            lines.append(f"  - {highlights[idx].text}")
    return "\n".join(lines)


def null_graph_creator() -> None:
    """Sentinel factory. Returns None to signal the pipeline to skip graph creation."""
    return None


def passthrough_graph_creator() -> GraphCreatorFn:
    """Return a graph creator that lays out raw highlights as canvas nodes without any LLM call."""
    GROUP_WIDTH = 400
    GROUP_PADDING = 20
    TEXT_HEIGHT = 80
    TEXT_WIDTH = 360
    TEXT_X_OFFSET = 20
    GROUP_SPACING_X = 500

    def create_graph(cluster_mapping: ClusterMapping, highlights: Highlights) -> dict:
        nodes = []
        edges: list = []

        for group_idx, (cluster_key, indices) in enumerate(cluster_mapping.items()):
            sorted_indices = sorted(indices)
            group_x = group_idx * GROUP_SPACING_X
            group_y = 0
            inner_height = len(sorted_indices) * (TEXT_HEIGHT + GROUP_PADDING)
            group_height = inner_height + GROUP_PADDING * 2

            group_id = str(uuid.uuid4())
            nodes.append({
                "id": group_id,
                "x": group_x,
                "y": group_y,
                "width": GROUP_WIDTH,
                "height": group_height,
                "type": "group",
                "label": f"Cluster {cluster_key}",
            })

            for text_idx, highlight_idx in enumerate(sorted_indices):
                text_y = group_y + GROUP_PADDING + text_idx * (TEXT_HEIGHT + GROUP_PADDING)
                nodes.append({
                    "id": str(uuid.uuid4()),
                    "x": group_x + TEXT_X_OFFSET,
                    "y": text_y,
                    "width": TEXT_WIDTH,
                    "height": TEXT_HEIGHT,
                    "type": "text",
                    "text": highlights[highlight_idx].text,
                })

        return {"nodes": nodes, "edges": edges}

    return create_graph


def claude_graph_creator(model: str = "claude-haiku-4-5-20251001", api_key: str | None = None) -> GraphCreatorFn:
    """Return a graph creator that uses Claude to produce a JSON Canvas knowledge graph.

    Args:
        model: Claude model ID to use.
        api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
    """
    client = anthropic.Anthropic(api_key=api_key)

    def create_graph(cluster_mapping: ClusterMapping, highlights: Highlights) -> dict:
        user_content = _build_user_message(cluster_mapping, highlights)

        message = client.messages.create(
            model=model,
            max_tokens=4 * 4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()

        logger.debug("Raw graph JSON from Claude:\n%s", raw)
        return json.loads(raw)

    return create_graph


def local_llm_graph_creator(
    model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device: str = "auto",
) -> GraphCreatorFn:
    """Return a graph creator that uses a local HuggingFace model to produce a JSON Canvas graph.

    Args:
        model: HuggingFace model ID to use.
        device: Device to run the model on. "auto" uses GPU if available,
                otherwise falls back to CPU. Explicit values: "cpu", "cuda", "mps", etc.
    """
    from huggingface_hub import login
    from transformers import pipeline

    login()

    kwargs = {"device_map": "auto"} if device == "auto" else {"device": device}
    pipe = pipeline("text-generation", model=model, **kwargs)

    def create_graph(cluster_mapping: ClusterMapping, highlights: Highlights) -> dict:
        user_content = _build_user_message(cluster_mapping, highlights)

        messages = [
            {"role": "system", "content": _LOCAL_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        result = pipe(messages, max_new_tokens=2048, max_length=None)
        raw = result[0]["generated_text"][-1]["content"].strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()

        logger.debug("Raw graph JSON from local LLM:\n%s", raw)
        return json.loads(raw)

    return create_graph
