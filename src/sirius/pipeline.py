import json
import logging
from datetime import datetime
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

from sirius.clustering import cluster_highlights
from sirius.protocols import ClusterMapping, PipelineFn

logger = logging.getLogger(__name__)


def _output_dir(filepath: str) -> Path:
    stem = Path(filepath).stem
    timestamp = datetime.now().strftime("%Y-%m-%d:%H-%M")
    return Path(f"{timestamp}-{stem}")


def _save_canvas(canvas: dict, filepath: str) -> Path:
    out_dir = _output_dir(filepath)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filepath).stem
    out_path = out_dir / f"knowledge-graph-{stem}.canvas"
    out_path.write_text(json.dumps(canvas, indent="\t"))
    logger.info(f"Knowledge graph saved to {out_path}")
    return out_path


def create_pipeline_fn(pipeline_cfg: DictConfig) -> PipelineFn:
    parse = instantiate(pipeline_cfg.highlight_parser)
    extract = instantiate(pipeline_cfg.extractor)
    encode = instantiate(pipeline_cfg.encoder)
    cluster = instantiate(pipeline_cfg.clusterer)
    # null_graph_creator() returns None → skip graph creation
    create_graph = instantiate(pipeline_cfg.graph_creator) if hasattr(pipeline_cfg, "graph_creator") else None

    def pipeline(filepath: str) -> ClusterMapping:
        highlights = parse(filepath)
        cluster_mapping = cluster_highlights(highlights, extract, encode, cluster)
        if create_graph is not None:
            canvas = create_graph(cluster_mapping, highlights)
            _save_canvas(canvas, filepath)
        return cluster_mapping

    return pipeline
