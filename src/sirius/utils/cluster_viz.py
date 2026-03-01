

from sirius.protocols import ClusterMapping


def pprint_clusters(clusters: ClusterMapping, highlights: list[str]) -> None:
    """Pretty-print clusters of highlights."""
    for cluster_key, indices in clusters.items():
        print(f"Cluster {cluster_key}:")
        for i in indices:
            print(f"  - {highlights[i]}")
        print()