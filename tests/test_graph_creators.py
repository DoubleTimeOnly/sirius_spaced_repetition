"""Tests for graph_creator factory functions."""

import json
from unittest.mock import MagicMock, patch

import pytest

from sirius.graph_creators import (
    claude_graph_creator,
    local_llm_graph_creator,
    null_graph_creator,
    passthrough_graph_creator,
)
from sirius.protocols import ClusterMapping, Highlight, Highlights

CLUSTER_MAPPING: ClusterMapping = {0: {0, 1}, 1: {2, 3}}
HIGHLIGHTS: Highlights = [
    Highlight(text="Storage strength is how well something is learned."),
    Highlight(text="Retrieval strength is how easily info comes to mind."),
    Highlight(text="Spacing improves long-term retention."),
    Highlight(text="Testing beats re-reading on delayed tests."),
]


# ---------------------------------------------------------------------------
# null_graph_creator
# ---------------------------------------------------------------------------


def test_null_graph_creator_returns_none():
    assert null_graph_creator() is None


# ---------------------------------------------------------------------------
# passthrough_graph_creator
# ---------------------------------------------------------------------------


def test_passthrough_returns_callable():
    assert callable(passthrough_graph_creator())


def test_passthrough_output_structure():
    create_graph = passthrough_graph_creator()
    canvas = create_graph(CLUSTER_MAPPING, HIGHLIGHTS)
    assert "nodes" in canvas
    assert "edges" in canvas
    assert isinstance(canvas["nodes"], list)
    assert isinstance(canvas["edges"], list)


def test_passthrough_creates_group_per_cluster():
    create_graph = passthrough_graph_creator()
    canvas = create_graph(CLUSTER_MAPPING, HIGHLIGHTS)
    groups = [n for n in canvas["nodes"] if n["type"] == "group"]
    assert len(groups) == len(CLUSTER_MAPPING)


def test_passthrough_creates_text_node_per_highlight():
    create_graph = passthrough_graph_creator()
    canvas = create_graph(CLUSTER_MAPPING, HIGHLIGHTS)
    text_nodes = [n for n in canvas["nodes"] if n["type"] == "text"]
    total_indices = sum(len(v) for v in CLUSTER_MAPPING.values())
    assert len(text_nodes) == total_indices


def test_passthrough_text_nodes_contain_highlight_text():
    create_graph = passthrough_graph_creator()
    canvas = create_graph(CLUSTER_MAPPING, HIGHLIGHTS)
    text_nodes = [n for n in canvas["nodes"] if n["type"] == "text"]
    texts = {n["text"] for n in text_nodes}
    for idx in [0, 1, 2, 3]:
        assert HIGHLIGHTS[idx].text in texts


def test_passthrough_no_edges():
    create_graph = passthrough_graph_creator()
    canvas = create_graph(CLUSTER_MAPPING, HIGHLIGHTS)
    assert canvas["edges"] == []


def test_passthrough_empty_mapping():
    create_graph = passthrough_graph_creator()
    canvas = create_graph({}, HIGHLIGHTS)
    assert canvas == {"nodes": [], "edges": []}


def test_passthrough_nodes_have_required_fields():
    create_graph = passthrough_graph_creator()
    canvas = create_graph(CLUSTER_MAPPING, HIGHLIGHTS)
    for node in canvas["nodes"]:
        assert "id" in node
        assert "x" in node
        assert "y" in node
        assert "width" in node
        assert "height" in node
        assert "type" in node


# ---------------------------------------------------------------------------
# claude_graph_creator
# ---------------------------------------------------------------------------


def test_claude_graph_creator_returns_callable():
    with patch("sirius.graph_creators.anthropic.Anthropic"):
        create_graph = claude_graph_creator()
    assert callable(create_graph)


def test_claude_graph_creator_does_not_call_api_at_factory_time():
    mock_client = MagicMock()
    with patch("sirius.graph_creators.anthropic.Anthropic", return_value=mock_client):
        claude_graph_creator()
    mock_client.messages.create.assert_not_called()


def test_claude_graph_creator_calls_api_with_model():
    valid_canvas = json.dumps({"nodes": [], "edges": []})
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=valid_canvas)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    with patch("sirius.graph_creators.anthropic.Anthropic", return_value=mock_client):
        create_graph = claude_graph_creator(model="claude-test-model")

    create_graph(CLUSTER_MAPPING, HIGHLIGHTS)

    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-test-model"


def test_claude_graph_creator_sends_cluster_content():
    valid_canvas = json.dumps({"nodes": [], "edges": []})
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=valid_canvas)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    with patch("sirius.graph_creators.anthropic.Anthropic", return_value=mock_client):
        create_graph = claude_graph_creator()

    create_graph(CLUSTER_MAPPING, HIGHLIGHTS)

    call_kwargs = mock_client.messages.create.call_args[1]
    user_content = call_kwargs["messages"][0]["content"]
    assert "Cluster 0" in user_content
    assert "Cluster 1" in user_content


def test_claude_graph_creator_returns_parsed_dict():
    canvas_dict = {"nodes": [{"id": "abc", "type": "text"}], "edges": []}
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=json.dumps(canvas_dict))]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    with patch("sirius.graph_creators.anthropic.Anthropic", return_value=mock_client):
        create_graph = claude_graph_creator()

    result = create_graph(CLUSTER_MAPPING, HIGHLIGHTS)
    assert result == canvas_dict


def test_claude_graph_creator_strips_markdown_fences():
    inner = {"nodes": [], "edges": []}
    fenced = f"```json\n{json.dumps(inner)}\n```"
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=fenced)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    with patch("sirius.graph_creators.anthropic.Anthropic", return_value=mock_client):
        create_graph = claude_graph_creator()

    result = create_graph(CLUSTER_MAPPING, HIGHLIGHTS)
    assert result == inner


def test_claude_graph_creator_strips_plain_fences():
    inner = {"nodes": [], "edges": []}
    fenced = f"```\n{json.dumps(inner)}\n```"
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=fenced)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    with patch("sirius.graph_creators.anthropic.Anthropic", return_value=mock_client):
        create_graph = claude_graph_creator()

    result = create_graph(CLUSTER_MAPPING, HIGHLIGHTS)
    assert result == inner


# ---------------------------------------------------------------------------
# local_llm_graph_creator
# ---------------------------------------------------------------------------

def _make_mock_pipe(response_text: str):
    """Build a mock transformers pipeline instance with a canned response.

    transformers text-generation returns:
      [{"generated_text": [{"role": ..., "content": ...}, ...]}]
    """
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": response_text},
    ]}]
    return mock_pipe


def _local_llm_sys_modules(mock_pipe):
    """Return a patch.dict context that replaces transformers and huggingface_hub.

    patch("transformers.pipeline") cannot intercept a lazy `from transformers import
    pipeline` inside a factory function because transformers uses module-level __getattr__
    for lazy loading. Replacing the entry in sys.modules is the reliable alternative.
    """
    mock_transformers = MagicMock()
    mock_transformers.pipeline.return_value = mock_pipe
    mock_hf = MagicMock()
    return patch.dict("sys.modules", {
        "transformers": mock_transformers,
        "huggingface_hub": mock_hf,
    })


def test_local_llm_graph_creator_returns_callable():
    mock_pipe = MagicMock()
    with _local_llm_sys_modules(mock_pipe):
        create_graph = local_llm_graph_creator()
    assert callable(create_graph)


def test_local_llm_graph_creator_does_not_run_inference_at_factory_time():
    mock_pipe = MagicMock()
    with _local_llm_sys_modules(mock_pipe):
        local_llm_graph_creator()
    mock_pipe.assert_not_called()


def test_local_llm_graph_creator_returns_parsed_dict():
    canvas_dict = {"nodes": [{"id": "x", "type": "text"}], "edges": []}
    mock_pipe = _make_mock_pipe(json.dumps(canvas_dict))

    with _local_llm_sys_modules(mock_pipe):
        create_graph = local_llm_graph_creator()

    result = create_graph(CLUSTER_MAPPING, HIGHLIGHTS)
    assert result == canvas_dict


def test_local_llm_graph_creator_strips_markdown_fences():
    inner = {"nodes": [], "edges": []}
    mock_pipe = _make_mock_pipe(f"```json\n{json.dumps(inner)}\n```")

    with _local_llm_sys_modules(mock_pipe):
        create_graph = local_llm_graph_creator()

    result = create_graph(CLUSTER_MAPPING, HIGHLIGHTS)
    assert result == inner


def test_local_llm_graph_creator_sends_cluster_content():
    valid_canvas = json.dumps({"nodes": [], "edges": []})
    mock_pipe = _make_mock_pipe(valid_canvas)

    with _local_llm_sys_modules(mock_pipe):
        create_graph = local_llm_graph_creator()

    create_graph(CLUSTER_MAPPING, HIGHLIGHTS)

    call_args = mock_pipe.call_args[0][0]  # first positional arg (messages list)
    user_msg = next(m for m in call_args if m["role"] == "user")
    assert "Cluster 0" in user_msg["content"]
    assert "Cluster 1" in user_msg["content"]


def test_local_llm_graph_creator_uses_cpu_device():
    mock_pipe = MagicMock()
    mock_transformers = MagicMock()
    mock_transformers.pipeline.return_value = mock_pipe
    mock_hf = MagicMock()

    with patch.dict("sys.modules", {"transformers": mock_transformers, "huggingface_hub": mock_hf}):
        local_llm_graph_creator(device="cpu")

    _, call_kwargs = mock_transformers.pipeline.call_args
    assert call_kwargs.get("device") == "cpu"
    assert "device_map" not in call_kwargs
