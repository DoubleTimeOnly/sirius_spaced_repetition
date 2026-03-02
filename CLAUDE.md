# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync              # Install dependencies (CPU PyTorch)
uv sync --extra gpu  # Install with CUDA/GPU support

pytest               # Run tests
pytest tests/test_clustering.py::test_passthrough_pipeline_returns_dict  # Run single test

python experiments/1-baseline/process_highlights.py      # Hardcoded highlights smoke test
python experiments/2-full-highlights/process_highlights.py  # Full pipeline on a real .md file
```

## Architecture

The pipeline processes book/article highlights into semantically clustered groups (for eventual flashcard generation):

```
HighlightParser → ExtractFn → EncodeFn → ClusterFn → ClusterMapping
```

All four stages are **callable protocols** defined in [src/sirius/protocols.py](src/sirius/protocols.py) and instantiated at runtime via Hydra. This makes every stage independently swappable.

**`ClusterMapping = dict[Any, set[int]]`** maps cluster keys to sets of highlight indices. Highlights can belong to multiple clusters (many-to-many via cosine similarity threshold).

### Configuration (Hydra)

Entry point: `configs/default_process.yaml` → composes `configs/pipeline/default.yaml` → each component has its own config dir (`extractor/`, `encoder/`, `clusterer/`, `highlight_parser/`).

`pipeline.device` in the root config propagates to sub-configs via OmegaConf interpolation (`${pipeline.device}`). To switch device, set `pipeline.device: cuda` in `default_process.yaml`.

Component configs use `_target_` pointing to factory functions in `src/sirius/` (e.g., `sirius.extractors.local_llm_extractor`). `hydra.utils.instantiate` calls these with the config fields as kwargs.

### Key source files

- [src/sirius/pipeline.py](src/sirius/pipeline.py) — factory (`create_pipeline_fn`) that instantiates all four components from config
- [src/sirius/clustering.py](src/sirius/clustering.py) — orchestrates extract → encode → cluster over a list of highlights
- [src/sirius/clusterers.py](src/sirius/clusterers.py) — HDBSCAN with soft/many-to-many membership via cosine similarity threshold
- [src/sirius/extractors.py](src/sirius/extractors.py) — passthrough, Claude API, Claude CLI subprocess, and local HuggingFace LLM implementations
- [src/sirius/encoders.py](src/sirius/encoders.py) — SentenceTransformers encoder
- [src/sirius/highlight_parsers.py](src/sirius/highlight_parsers.py) — Readwise markdown parser (lines starting with `"> "`)
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
