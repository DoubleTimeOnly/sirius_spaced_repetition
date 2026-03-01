import logging

from sirius.pipeline import create_pipeline_fn
from sirius.utils.cluster_viz import pprint_clusters
from sirius.utils.hydra_utils import load_config
from sirius.utils.logging import set_logging_level


HIGHLIGHTS = [
    # Memory / retrieval group
    "Any memory has two strengths, a storage strength and a retrieval strength.",
    "Storage strength is a measure of how well learned something is.",
    "Retrieval strength is a measure of how easily a nugget of information comes to mind.",
    "No memory is ever 'lost'; it is just not currently accessible — its retrieval strength is low.",
    # Spacing / practice group
    "People learn at least as much, and retain it much longer, when they space their study time.",
    "Cramming works fine in a pinch. It just doesn't last. Spacing does.",
    "The optimal interval between sessions scales with how far away the exam is.",
    # Desirable difficulty group
    "The harder your brain has to work to dig out a memory, the greater the increase in learning.",
    "Testing > studying, by a country mile, on delayed tests.",
    "Spend the first third of your time memorizing, and the remaining two thirds reciting from memory.",
]


if __name__ == "__main__":
    cfg = load_config(
        config_name="default_process.yaml",
        overrides=[
            "extractor@pipeline.extractor=local_llm",
            "pipeline.device=cuda",
        ],
    )
    set_logging_level(cfg.logging.level)

    pipeline_fn = create_pipeline_fn(cfg.pipeline)
    clusters = pipeline_fn(HIGHLIGHTS)

    pprint_clusters(clusters, HIGHLIGHTS)
