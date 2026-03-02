from hydra.utils import instantiate

from sirius.pipeline import create_pipeline_fn
from sirius.utils.cluster_viz import pprint_clusters
from sirius.utils.hydra_utils import load_config
from sirius.utils.logging import set_logging_level


FILEPATH = "examples/How We Learn - Benedict Carey.md"


if __name__ == "__main__":
    cfg = load_config(
        config_name="default_process.yaml",
        overrides=[
            "extractor@pipeline.extractor=passthrough",
        ],
    )
    set_logging_level(cfg.logging.level)

    pipeline_fn = create_pipeline_fn(cfg.pipeline)
    clusters = pipeline_fn(FILEPATH)

    # Parse separately to get highlight text for visualization
    parse = instantiate(cfg.pipeline.highlight_parser)
    highlights = parse(FILEPATH)
    pprint_clusters(clusters, highlights)
