from typing import Any

from .protocols import ClusterFn, EncodeFn, ExtractFn


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
    vectors = [encode(info) for info in core_infos]
    return cluster(vectors)
