from hydra.utils import instantiate

from sirius.pipeline import create_pipeline_fn
from sirius.utils.cluster_viz import pprint_clusters
from sirius.utils.hydra_utils import load_config
from sirius.utils.logging import set_logging_level


FILEPATH = "How We Learn"


if __name__ == "__main__":
    cfg = load_config(
        config_name="default_process.yaml",
        overrides=[
            "highlight_parser@pipeline.highlight_parser=readwise_api",

            "extractor@pipeline.extractor=passthrough",
            # "extractor@pipeline.extractor=local_llm",
            # "pipeline.extractor.model=google/gemma-3-1b-it",

            "pipeline.encoder.model=google/embeddinggemma-300m",

            "graph_creator@pipeline.graph_creator=claude_api",
            # "graph_creator@pipeline.graph_creator=local_llm",
            # "pipeline.graph_creator.model=google/gemma-3-1b-it",
            # "pipeline.graph_creator.model=google/gemma-3n-E2B-it",
        ],
    )
    set_logging_level(cfg.logging.level)

    pipeline_fn = create_pipeline_fn(cfg.pipeline)
    clusters = pipeline_fn(FILEPATH)

