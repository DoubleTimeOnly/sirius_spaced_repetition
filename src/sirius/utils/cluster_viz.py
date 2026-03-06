

import logging

from sirius.protocols import ClusterMapping, Highlight

logger = logging.getLogger("sirius.cluster_viz")


def pprint_clusters(clusters: ClusterMapping, highlights: list[Highlight]) -> None:
    """Pretty-print clusters of highlights."""
    for cluster_key, indices in clusters.items():
        logger.debug("Cluster %s:", cluster_key)
        for i in indices:
            logger.debug("  - %s", highlights[i].text)
        logger.debug("")