# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync              # Install dependencies (CPU PyTorch)
uv sync --extra gpu  # Install with CUDA/GPU support

pytest               # Run tests
pytest tests/test_clustering.py::test_passthrough_pipeline_returns_dict  # Run single test

python experiments/1-baseline/process_highlights.py           # Hardcoded highlights smoke test
python experiments/2-full-highlights/process_highlights.py    # Full pipeline on a real .md file
python experiments/3-knowledge-graph-creation/process_highlights.py  # KG creation with Readwise API
python experiments/4-contextual-embeddings/process_highlights.py     # Contextual embeddings + KG
```

## Architecture

The pipeline processes book/article highlights into semantically clustered groups and optionally produces a knowledge graph:

```
HighlightParser → ExtractFn → EncodeFn → ClusterFn → (GraphCreatorFn) → ClusterMapping
```

All stages are **callable protocols** defined in [src/sirius/protocols.py](src/sirius/protocols.py) and instantiated at runtime via Hydra. This makes every stage independently swappable.

**`ClusterMapping = dict[Any, set[int]]`** maps cluster keys to sets of highlight indices. Highlights can belong to multiple clusters (many-to-many via cosine similarity threshold).

**`Highlight`** is a dataclass with `text: str` and `context: tuple[str, str] | None` (before/after sentences). `highlight.combine()` returns the text with surrounding context joined as a single string — this is what gets passed to `EncodeFn`.

**`EncodeFn = Callable[[list[str]], list[np.ndarray]]`** — batch interface. Takes all texts at once; returns one embedding per text. This enables contextual encoders that need cross-highlight attention.

**`GraphCreatorFn = Callable[[ClusterMapping, Highlights], dict]`** — optional final stage that produces a JSON Canvas file (Obsidian-compatible). `null_graph_creator()` returns `None` as a sentinel to skip this stage.

### Configuration (Hydra)

Entry point: `configs/default_process.yaml` → composes `configs/pipeline/default.yaml` → each component has its own config dir (`extractor/`, `encoder/`, `clusterer/`, `highlight_parser/`, `graph_creator/`).

`pipeline.device` in the root config propagates to sub-configs via OmegaConf interpolation (`${pipeline.device}`). To switch device, set `pipeline.device: cuda` in `default_process.yaml`.

Component configs use `_target_` pointing to factory functions in `src/sirius/` (e.g., `sirius.extractors.local_llm_extractor`). `hydra.utils.instantiate` calls these with the config fields as kwargs.

The `graph_creator` stage defaults to `none` (skipped). Set it to `claude_api`, `passthrough`, or `local_llm` in `default_process.yaml` to enable graph creation. Each run saves a resolved `config.yaml` and optionally a `.canvas` file to `outputs/<timestamp>_<stem>/`.

### Key source files

- [src/sirius/pipeline.py](src/sirius/pipeline.py) — factory (`create_pipeline_fn`) that instantiates all five components; saves resolved config and canvas to `outputs/`
- [src/sirius/clustering.py](src/sirius/clustering.py) — orchestrates extract → encode → cluster over a list of highlights
- [src/sirius/clusterers.py](src/sirius/clusterers.py) — HDBSCAN with soft/many-to-many membership via cosine similarity threshold
- [src/sirius/extractors.py](src/sirius/extractors.py) — passthrough, Claude API, Claude CLI subprocess, and local HuggingFace LLM implementations
- [src/sirius/encoders.py](src/sirius/encoders.py) — `sentence_transformer_encoder` and `contextual_encoder` (Perplexity late-chunking model)
- [src/sirius/highlight_parsers.py](src/sirius/highlight_parsers.py) — `readwise_markdown_parser` (lines starting with `"> "`) and `readwise_api_parser` (fetches highlights + surrounding context via Readwise v2/v3 APIs; supports articles and EPUBs)
- [src/sirius/graph_creators.py](src/sirius/graph_creators.py) — `null_graph_creator`, `passthrough_graph_creator`, `claude_graph_creator`, `local_llm_graph_creator`
- [src/sirius/utils/hydra_utils.py](src/sirius/utils/hydra_utils.py) — `load_config()` helper for loading the Hydra config outside of a `@hydra.main` context

### Adding a new component

1. Write a factory function in the relevant `src/sirius/*.py` file matching the protocol signature
2. Add a YAML config in the corresponding `configs/<type>/` directory with `_target_` pointing to the factory
3. Reference the new config in `configs/pipeline/default.yaml` defaults

## Known Issues

- **`device="auto"` crashes**: Always use `device="cpu"` (or an explicit device string) in tests — PyTorch will crash with `device="auto"`.

## Test Maintenance

- Update tests whenever a pipeline stage is added or removed (tests that exercise the full pipeline must match the current stage count/types)
- `tests/test_pipeline.py` mocks `sirius.pipeline.instantiate` — this is intentional to avoid the device parameter bug, not because instantiate is hard to use

## Logging

- Must be scoped to the `sirius` package only — don't configure the root logger. See `src/sirius/utils/logging.py`.
