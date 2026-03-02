"""Tests for Hydra config loading and pipeline creation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from sirius.pipeline import create_pipeline_fn
from sirius.utils.hydra_utils import load_config


HIGHLIGHTS = [
    "Storage strength is how well learned something is.",
    "Retrieval strength is how easily information comes to mind.",
    "Spacing improves long-term retention.",
    "Testing beats re-reading on delayed tests.",
    "The harder you work to recall, the more you learn.",
    "Cramming works short-term; spacing works long-term.",
]


def _make_mock_components():
    """Return (extract, encode, cluster) mocks suitable for pipeline tests."""
    mock_extract = MagicMock(side_effect=lambda h, ctx=None: h)
    mock_encode = MagicMock(side_effect=lambda t: np.ones(8))
    mock_cluster = MagicMock(return_value={0: {0, 1}, 1: {2, 3}})
    return mock_extract, mock_encode, mock_cluster


def _minimal_pipeline_cfg() -> DictConfig:
    return OmegaConf.create({
        "device": "cpu",
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
    assert p.extractor._target_ == "sirius.extractors.passthrough_extractor"
    assert p.encoder._target_ == "sirius.encoders.sentence_transformer_encoder"
    assert p.encoder.model == "all-MiniLM-L6-v2"
    assert p.clusterer._target_ == "sirius.clusterers.hdbscan_clusterer"
    assert p.clusterer.min_cluster_size == 2
    assert p.clusterer.threshold == pytest.approx(0.5)


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
    extract, encode, cluster = _make_mock_components()
    cfg = _minimal_pipeline_cfg()

    with patch("sirius.pipeline.instantiate", side_effect=[extract, encode, cluster]) as mock_inst:
        pipeline = create_pipeline_fn(cfg)

    assert callable(pipeline)
    assert mock_inst.call_count == 3
    calls = mock_inst.call_args_list
    assert calls[0].kwargs.get("device") == "cpu"  # extractor
    assert calls[1].kwargs.get("device") == "cpu"  # encoder
    assert "device" not in calls[2].kwargs          # clusterer


def test_create_pipeline_fn_output():
    """Pipeline returns a dict[Any, set] and drives each component correctly."""
    extract, encode, cluster = _make_mock_components()
    cfg = _minimal_pipeline_cfg()

    with patch("sirius.pipeline.instantiate", side_effect=[extract, encode, cluster]):
        pipeline = create_pipeline_fn(cfg)

    result = pipeline(HIGHLIGHTS)
    assert isinstance(result, dict) and all(isinstance(v, set) for v in result.values())
    assert extract.call_count == len(HIGHLIGHTS)
    assert encode.call_count == len(HIGHLIGHTS)
    cluster.assert_called_once()
