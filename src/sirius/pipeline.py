from hydra.utils import instantiate
from omegaconf import DictConfig

from sirius.clustering import cluster_highlights
from sirius.protocols import ClusterMapping, PipelineFn


def create_pipeline_fn(pipeline_cfg: DictConfig) -> PipelineFn:
    parse = instantiate(pipeline_cfg.highlight_parser)
    extract = instantiate(pipeline_cfg.extractor)
    encode = instantiate(pipeline_cfg.encoder)
    cluster = instantiate(pipeline_cfg.clusterer)

    def pipeline(filepath: str) -> ClusterMapping:
        highlights = parse(filepath)
        return cluster_highlights(highlights, extract, encode, cluster)

    return pipeline
