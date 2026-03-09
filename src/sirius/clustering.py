import logging
from typing import Any, Optional

from sirius.utils.cluster_viz import pprint_clusters

from .protocols import ClusterFn, ClusteringPreprocessingFn, EncodeFn, ExtractFn, Highlights

logger = logging.getLogger(__name__)


def cluster_highlights(
    highlights: Highlights,
    extract: ExtractFn,
    encode: EncodeFn,
    cluster: ClusterFn,
    preprocess: Optional[ClusteringPreprocessingFn] = None,
) -> dict[Any, set[int]]:
    """Cluster a list of highlights by semantic similarity.

    Args:
        highlights: List of Highlight objects.
        extract: Function to distil a highlight into its core information.
        encode: Function to encode a text string into an embedding vector.
        preprocess: Optional function to preprocess vectors (e.g., dimensionality reduction).
        cluster: Function to cluster a list of vectors into groups.

    Returns:
        Mapping of cluster key -> set of highlight indices. A highlight may
        appear in multiple clusters (many-to-many).
    """
    core_infos = [extract(h) for h in highlights]
    logger.debug(f"Extracted {len(core_infos)} core infos from highlights")
    for i, (h, info) in enumerate(zip(highlights, core_infos)):
        output_str = ""
        output_str += f"\n===== Highlight {i} =====\n"
        len_header = len(output_str)
        output_str += f"Original: {h.text}\n"
        output_str += "-" * len_header + "\n"
        output_str += f"Core info: {info}\n"
        logger.debug(output_str)

    # Combine core infos with context and batch encode all at once
    texts_to_encode = [h.combine() for h in highlights]
    vectors = encode(texts_to_encode)
    logger.debug(f"Encoded {len(vectors)} vectors")

    # Apply preprocessing if provided
    if preprocess is not None:
        vectors = preprocess(vectors)
        logger.debug(f"Preprocessed vectors")

    clusters = cluster(vectors)
    logger.debug(f"Clustered into {len(clusters)} clusters")
    logger.debug(f"Cluster output: {clusters}")

    if logger.level <= logging.DEBUG:
        pprint_clusters(clusters, highlights)

    return clusters
