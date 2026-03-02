import logging
from typing import Any

from .protocols import ClusterFn, EncodeFn, ExtractFn

logger = logging.getLogger(__name__)


def cluster_highlights(
    highlights: list[str],
    extract: ExtractFn,
    encode: EncodeFn,
    cluster: ClusterFn,
) -> dict[Any, set[int]]:
    """Cluster a list of highlights by semantic similarity.

    Args:
        highlights: Raw highlight strings.
        extract: Function to distil a highlight into its core information.
        encode: Function to encode a text string into an embedding vector.
        cluster: Function to cluster a list of vectors into groups.

    Returns:
        Mapping of cluster key -> set of highlight indices. A highlight may
        appear in multiple clusters (many-to-many).
    """
    core_infos = [extract(h, None) for h in highlights]
    logger.debug(f"Extracted {len(core_infos)} core infos from highlights")
    for i, (h, info) in enumerate(zip(highlights, core_infos)):
        output_str = ""
        output_str += f"\/terminaln===== Highlight {i} =====\n"
        len_header = len(output_str)
        output_str += f"Original: {h}\n"
        output_str += "-" * len_header + "\n"
        output_str += f"Core info: {info}\n"
        logger.debug(output_str)

    vectors = [encode(info) for info in core_infos]
    logger.debug(f"Encoded {len(vectors)} vectors")

    clusters = cluster(vectors)
    logger.debug(f"Clustered into {len(clusters)} clusters")
    logger.debug(f"Cluster output: {clusters}")

    return clusters
