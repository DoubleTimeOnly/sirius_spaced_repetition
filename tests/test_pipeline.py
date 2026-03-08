"""Tests for Hydra config loading and pipeline creation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from sirius.pipeline import _run_dir, _save_canvas, create_pipeline_fn
from sirius.protocols import Highlight, Highlights
from sirius.utils.hydra_utils import load_config


HIGHLIGHTS: Highlights = [
    Highlight(text="Storage strength is how well learned something is."),
    Highlight(text="Retrieval strength is how easily information comes to mind."),
    Highlight(text="Spacing improves long-term retention."),
    Highlight(text="Testing beats re-reading on delayed tests."),
    Highlight(text="The harder you work to recall, the more you learn."),
    Highlight(text="Cramming works short-term; spacing works long-term."),
]


def _make_mock_components():
    """Return (parse, extract, encode, cluster) mocks suitable for pipeline tests."""
    mock_parse = MagicMock(return_value=HIGHLIGHTS)
    mock_extract = MagicMock(side_effect=lambda h, ctx=None: h)
    # encode now takes a list of texts and returns a list of vectors
    mock_encode = MagicMock(side_effect=lambda texts: [np.ones(8) for _ in texts])
    mock_cluster = MagicMock(return_value={0: {0, 1}, 1: {2, 3}})
    return mock_parse, mock_extract, mock_encode, mock_cluster


def _minimal_pipeline_cfg() -> DictConfig:
    return OmegaConf.create({
        "device": "cpu",
        "highlight_parser": {"_target_": "sirius.highlight_parsers.readwise_markdown_parser"},
        "extractor": {"_target_": "sirius.extractors.passthrough_extractor"},
        "encoder": {
            "_target_": "sirius.encoders.sentence_transformer_encoder",
            "model": "all-MiniLM-L6-v2",
            "device": "cpu",
        },
        "clusterer": {
            "_target_": "sirius.clusterers.hdbscan_clusterer",
            "min_cluster_size": 2,
            "threshold": 0.5,
        },
    })


# ---------------------------------------------------------------------------
# Hydra config loading
# ---------------------------------------------------------------------------


def test_default_config_loads():
    cfg = load_config("default_process.yaml")
    assert isinstance(cfg, DictConfig)
    assert "pipeline" in cfg and "logging" in cfg
    p = cfg.pipeline
    assert "device" in p


def test_config_extractor_overrides():
    for name, target in [
        ("claude_api", "sirius.extractors.claude_extractor"),
        ("claude_code", "sirius.extractors.claude_code_extractor"),
        ("local_llm", "sirius.extractors.local_llm_extractor"),
    ]:
        cfg = load_config(
            "default_process.yaml",
            overrides=[f"extractor@pipeline.extractor={name}"],
        )
        assert cfg.pipeline.extractor._target_ == target


def test_config_device_override():
    cfg = load_config("default_process.yaml", overrides=["pipeline.device=cuda"])
    assert cfg.pipeline.device == "cuda"


# ---------------------------------------------------------------------------
# create_pipeline_fn
# ---------------------------------------------------------------------------


def test_create_pipeline_fn_wiring():
    """Returns a callable and instantiates components with the right device args."""
    parse, extract, encode, cluster = _make_mock_components()
    cfg = _minimal_pipeline_cfg()

    with patch("sirius.pipeline.instantiate", side_effect=[parse, extract, encode, cluster]) as mock_inst:
        pipeline = create_pipeline_fn(cfg)

    assert callable(pipeline)
    assert mock_inst.call_count == 4


def test_create_pipeline_fn_output(tmp_path, monkeypatch):
    """Pipeline returns a dict[Any, set] and drives each component correctly."""
    parse, extract, encode, cluster = _make_mock_components()
    cfg = _minimal_pipeline_cfg()

    monkeypatch.chdir(tmp_path)

    with patch("sirius.pipeline.instantiate", side_effect=[parse, extract, encode, cluster]):
        pipeline = create_pipeline_fn(cfg)

    result = pipeline("fake_path.md")
    assert isinstance(result, dict) and all(isinstance(v, set) for v in result.values())
    parse.assert_called_once_with("fake_path.md")
    assert extract.call_count == len(HIGHLIGHTS)
    # encode is now called once with all texts in a batch
    assert encode.call_count == 1
    cluster.assert_called_once()


def _minimal_pipeline_cfg_with_graph_creator() -> DictConfig:
    cfg = _minimal_pipeline_cfg()
    return OmegaConf.merge(cfg, OmegaConf.create({
        "graph_creator": {"_target_": "sirius.graph_creators.passthrough_graph_creator"}
    }))


# ---------------------------------------------------------------------------
# graph_creator integration
# ---------------------------------------------------------------------------


def test_create_pipeline_fn_with_graph_creator_wiring():
    parse, extract, encode, cluster = _make_mock_components()
    mock_create_graph = MagicMock(return_value={"nodes": [], "edges": []})
    cfg = _minimal_pipeline_cfg_with_graph_creator()

    with patch("sirius.pipeline.instantiate",
               side_effect=[parse, extract, encode, cluster, mock_create_graph]) as mock_inst:
        pipeline = create_pipeline_fn(cfg)

    assert callable(pipeline)
    assert mock_inst.call_count == 5


def test_pipeline_calls_graph_creator_and_saves_canvas(tmp_path, monkeypatch):
    parse, extract, encode, cluster = _make_mock_components()
    mock_create_graph = MagicMock(return_value={"nodes": [], "edges": []})
    cfg = _minimal_pipeline_cfg_with_graph_creator()

    monkeypatch.chdir(tmp_path)

    with patch("sirius.pipeline.instantiate",
               side_effect=[parse, extract, encode, cluster, mock_create_graph]):
        pipeline = create_pipeline_fn(cfg)

    pipeline("fake_highlights.md")

    mock_create_graph.assert_called_once()
    canvas_files = list(tmp_path.rglob("*.canvas"))
    assert len(canvas_files) == 1
    assert canvas_files[0].name == "knowledge-graph-fake_highlights.canvas"


def test_pipeline_canvas_contains_valid_json(tmp_path, monkeypatch):
    parse, extract, encode, cluster = _make_mock_components()
    canvas_data = {"nodes": [{"id": "a", "type": "text"}], "edges": []}
    mock_create_graph = MagicMock(return_value=canvas_data)
    cfg = _minimal_pipeline_cfg_with_graph_creator()

    monkeypatch.chdir(tmp_path)

    with patch("sirius.pipeline.instantiate",
               side_effect=[parse, extract, encode, cluster, mock_create_graph]):
        pipeline = create_pipeline_fn(cfg)

    pipeline("fake_highlights.md")

    canvas_file = list(tmp_path.rglob("*.canvas"))[0]
    import json
    saved = json.loads(canvas_file.read_text())
    assert saved == canvas_data


def test_pipeline_skips_graph_creator_when_null(tmp_path, monkeypatch):
    parse, extract, encode, cluster = _make_mock_components()
    cfg = _minimal_pipeline_cfg_with_graph_creator()

    monkeypatch.chdir(tmp_path)

    # null_graph_creator() returns None
    with patch("sirius.pipeline.instantiate",
               side_effect=[parse, extract, encode, cluster, None]):
        pipeline = create_pipeline_fn(cfg)

    result = pipeline("fake_highlights.md")

    assert isinstance(result, dict)
    assert list(tmp_path.rglob("*.canvas")) == []


def test_pipeline_without_graph_creator_key_still_works():
    """Configs without a graph_creator key skip instantiation gracefully."""
    parse, extract, encode, cluster = _make_mock_components()
    cfg = _minimal_pipeline_cfg()  # no graph_creator key

    with patch("sirius.pipeline.instantiate",
               side_effect=[parse, extract, encode, cluster]) as mock_inst:
        pipeline = create_pipeline_fn(cfg)

    assert callable(pipeline)
    assert mock_inst.call_count == 4


# ---------------------------------------------------------------------------
# _output_dir / _save_canvas helpers
# ---------------------------------------------------------------------------


def test_run_dir_format():
    import re
    path = _run_dir("examples/How We Learn - Benedict Carey.md", "outputs")
    assert path.name.endswith("_How We Learn - Benedict Carey")
    # timestamp prefix: YYYY-MM-DD_HH:MM
    assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}:\d{2}_", path.name)
    assert path.parent == Path("outputs")


def test_save_canvas_creates_file(tmp_path):
    canvas = {"nodes": [], "edges": []}
    out = _save_canvas(canvas, tmp_path, "my_highlights")
    assert out.exists()
    assert out.name == "knowledge-graph-my_highlights.canvas"
