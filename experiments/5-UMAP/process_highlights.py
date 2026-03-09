from hydra.utils import instantiate

from sirius.pipeline import create_pipeline_fn
from sirius.utils.cluster_viz import pprint_clusters
from sirius.utils.hydra_utils import load_config
from sirius.utils.logging import set_logging_level


FILEPATH = "How to Know a Person"


if __name__ == "__main__":
    cfg = load_config(
        config_name="default_process.yaml",
        overrides=[
            "pipeline.enable_cluster_visualization=true",
            "clustering_preprocessing@pipeline.clustering_preprocessing=umap",
            "graph_creator@pipeline.graph_creator=none",
            # "graph_creator@pipeline.graph_creator=claude_api",
            # "pipeline.graph_creator.model=claude-sonnet-4-6",
        ],
    )
    set_logging_level(cfg.logging.level)

    pipeline_fn = create_pipeline_fn(cfg.pipeline, full_cfg=cfg)
    clusters = pipeline_fn(FILEPATH)

