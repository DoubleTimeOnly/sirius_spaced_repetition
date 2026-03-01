from typing import Any

import numpy as np

from .protocols import ClusterFn


def hdbscan_clusterer(min_cluster_size: int = 2, threshold: float = 0.7) -> ClusterFn:
    """Return a clusterer using HDBSCAN with cosine-similarity-based soft membership.

    After clustering, each highlight is also added to any cluster whose centroid
    has a cosine similarity >= `threshold` with that highlight's vector. This gives
    many-to-many membership: a highlight that spans multiple topics can appear in
    more than one cluster.

    Args:
        min_cluster_size: Minimum number of points to form a cluster.
        threshold: Cosine similarity threshold for cross-cluster membership
                   (0.0-1.0). Higher values are more restrictive.

    Returns:
        ClusterFn mapping cluster label (int) -> set of highlight indices.
    """
    from sklearn.cluster import HDBSCAN

    def cluster(vectors: list[np.ndarray]) -> dict[Any, set[int]]:
        matrix = np.stack(vectors)

        # Normalise to unit length so that dot product == cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized = matrix / norms

        clf = HDBSCAN(min_cluster_size=min_cluster_size, copy=True)
        clf.fit(normalized)

        unique_labels = sorted(set(clf.labels_) - {-1})
        if not unique_labels:
            return {}

        # Compute per-cluster centroids (already in unit-norm space; re-normalise)
        centroids: dict[int, np.ndarray] = {}
        for label in unique_labels:
            mask = clf.labels_ == label
            centroid = normalized[mask].mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            centroids[label] = centroid / centroid_norm if centroid_norm > 0 else centroid

        # Soft many-to-many assignment: include a point in every cluster whose
        # centroid has cosine similarity >= threshold with that point.
        result: dict[int, set[int]] = {}
        for point_idx in range(len(vectors)):
            for label, centroid in centroids.items():
                similarity = float(np.dot(normalized[point_idx], centroid))
                if similarity >= threshold:
                    result.setdefault(label, set()).add(point_idx)

        return result

    return cluster
