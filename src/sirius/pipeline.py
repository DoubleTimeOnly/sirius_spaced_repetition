from hydra.utils import instantiate
from omegaconf import DictConfig

from sirius.clustering import cluster_highlights
from sirius.protocols import ClusterMapping, Highlights, PipelineFn


def create_pipeline_fn(pipeline_cfg: DictConfig) -> PipelineFn:
    # TODO: make the configs read the device from the top-level instead of passing it down to each component
    device = pipeline_cfg.device
    extract = instantiate(pipeline_cfg.extractor, device=device)
    encode = instantiate(pipeline_cfg.encoder, device=device)
    cluster = instantiate(pipeline_cfg.clusterer)

    def pipeline(highlights: Highlights) -> ClusterMapping:
        return cluster_highlights(highlights, extract, encode, cluster)

    return pipeline
