import json
import logging
from datetime import datetime
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

from sirius.clustering import cluster_highlights
from sirius.protocols import ClusterMapping, PipelineFn
from sirius.utils.logging import add_file_handler

logger = logging.getLogger(__name__)


def _run_dir(filepath: str, base_dir: str) -> Path:
    stem = Path(filepath).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
    return Path(base_dir) / f"{timestamp}_{stem}"


def _save_canvas(canvas: dict, run_dir: Path, stem: str) -> Path:
    out_path = run_dir / f"knowledge-graph-{stem}.canvas"
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
    output_base_dir = getattr(pipeline_cfg, "output_base_dir", "outputs")

    def pipeline(filepath: str) -> ClusterMapping:
        stem = Path(filepath).stem
        run_dir = _run_dir(filepath, output_base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        add_file_handler(run_dir / "run.log")

        highlights = parse(filepath)
        cluster_mapping = cluster_highlights(highlights, extract, encode, cluster)
        if create_graph is not None:
            canvas = create_graph(cluster_mapping, highlights)
            _save_canvas(canvas, run_dir, stem)
        return cluster_mapping

    return pipeline
