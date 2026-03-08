from typing import Any

import numpy as np

from .protocols import ClusterFn


def hdbscan_clusterer(
    hdbscan_kwargs: dict[str, Any],
    threshold: float = 0.5,
) -> ClusterFn:
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

        clf = HDBSCAN(**hdbscan_kwargs)
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

        updated_labels = clf.labels_.copy()
        result = {label: set(int(i) for i in np.where(clf.labels_ == label)[0]) for label in unique_labels}
        if threshold > 0:
            # Soft many-to-many assignment: include a point in every cluster whose
            # centroid has cosine similarity >= threshold with that point.
            for point_idx in range(len(vectors)):
                for label, centroid in centroids.items():
                    similarity = float(np.dot(normalized[point_idx], centroid))
                    if similarity >= threshold:
                        result.setdefault(label, set()).add(point_idx)
                        # We track if a point belongs to any cluster, even if it wasn't assigned to one by HDBSCAN
                        updated_labels[point_idx] = label 
        
        solo_clusters = np.where(updated_labels == -1)
        for cluster_idx, highlight_idx in enumerate(solo_clusters[0], start=len(result)):
            result[cluster_idx] = {int(highlight_idx)}

        return result

    return cluster
