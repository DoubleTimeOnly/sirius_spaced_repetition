"""Tests for clusterer factory functions."""

import numpy as np

from sirius.clusterers import hdbscan_clusterer


def _make_cluster_vectors(center: np.ndarray, n: int = 5, noise: float = 0.05, seed: int = 0) -> list[np.ndarray]:
    """Return *n* vectors near *center* with small Gaussian noise."""
    rng = np.random.RandomState(seed)
    return list(center + rng.randn(n, len(center)) * noise)


def test_hdbscan_clusterer_returns_callable():
    assert callable(hdbscan_clusterer())
    assert callable(hdbscan_clusterer(min_cluster_size=3, threshold=0.8))


def test_hdbscan_clusterer_output_structure():
    cluster = hdbscan_clusterer(min_cluster_size=2, threshold=0.5)
    dim = 16
    vectors = (
        _make_cluster_vectors(np.array([1.0] + [0.0] * (dim - 1)), n=5, seed=0)
        + _make_cluster_vectors(np.array([0.0, 1.0] + [0.0] * (dim - 2)), n=5, seed=1)
    )
    result = cluster(vectors)
    assert isinstance(result, dict)
    for indices in result.values():
        assert isinstance(indices, set)
        assert all(isinstance(i, int) and 0 <= i < len(vectors) for i in indices)


def test_hdbscan_clusterer_separates_two_groups():
    cluster = hdbscan_clusterer(min_cluster_size=2, threshold=0.9)
    dim = 32
    group_a, group_b = set(range(5)), set(range(5, 10))
    vectors = (
        _make_cluster_vectors(np.array([1.0] + [0.0] * (dim - 1)), n=5, seed=0)
        + _make_cluster_vectors(np.array([0.0, 1.0] + [0.0] * (dim - 2)), n=5, seed=1)
    )
    result = cluster(vectors)
    assert len(result) >= 2
    for indices in result.values():
        assert not (indices & group_a and indices & group_b), (
            f"Cluster mixes group A and group B: {indices}"
        )


def test_hdbscan_clusterer_respects_min_cluster_size():
    """min_cluster_size larger than each group but ≤ total produces no clusters."""
    cluster = hdbscan_clusterer(min_cluster_size=4, threshold=0.5)
    dim = 8
    vectors = (
        _make_cluster_vectors(np.array([1.0] + [0.0] * (dim - 1)), n=3, seed=0)
        + _make_cluster_vectors(np.array([0.0, 1.0] + [0.0] * (dim - 2)), n=3, seed=1)
    )
    assert cluster(vectors) == {}
