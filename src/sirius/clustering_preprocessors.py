import numpy as np

from sirius.protocols import ClusteringPreprocessingFn


def passthrough_preprocessing() -> ClusteringPreprocessingFn:
    """Returns vectors unchanged (no preprocessing)."""

    def _passthrough(vectors: list[np.ndarray]) -> list[np.ndarray]:
        return vectors

    return _passthrough


def umap_preprocessing(
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> ClusteringPreprocessingFn:
    """Apply UMAP dimensionality reduction to vectors.

    Args:
        n_components: Target dimensionality for the reduced space.
        n_neighbors: Number of neighbors for local structure approximation.
        min_dist: Minimum distance between points in low-dimensional space.
        metric: Distance metric to use.
        random_state: Random seed for reproducibility.

    Returns:
        A function that applies UMAP transformation to a list of vectors.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for UMAP preprocessing. "
            "Install it with: uv sync --extra umap"
        )

    def _umap(vectors: list[np.ndarray]) -> list[np.ndarray]:
        # Stack vectors into matrix for UMAP
        X = np.array(vectors)

        # Fit and transform
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        X_reduced = reducer.fit_transform(X)

        # Convert back to list of vectors
        return [X_reduced[i] for i in range(len(X_reduced))]

    return _umap
