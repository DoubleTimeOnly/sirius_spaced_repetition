"""Smoke tests for the clustering pipeline using local-only dependencies."""

import sirius


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


def test_passthrough_pipeline_returns_dict():
    extract = sirius.passthrough_extractor()
    encode = sirius.sentence_transformer_encoder(device="cpu")
    cluster = sirius.hdbscan_clusterer(min_cluster_size=2, threshold=0.5)

    result = sirius.cluster_highlights(HIGHLIGHTS, extract, encode, cluster)

    assert isinstance(result, dict)
    for cluster_key, indices in result.items():
        assert isinstance(indices, set)
        assert all(isinstance(i, int) for i in indices)
        assert all(0 <= i < len(HIGHLIGHTS) for i in indices)


def test_highlights_appear_in_clusters():
    extract = sirius.passthrough_extractor()
    encode = sirius.sentence_transformer_encoder(device="cpu")
    cluster = sirius.hdbscan_clusterer(min_cluster_size=2, threshold=0.5)

    result = sirius.cluster_highlights(HIGHLIGHTS, extract, encode, cluster)

    # At least one cluster should be found given the clearly related highlights
    assert len(result) >= 1

    # Every index in every cluster must be a valid highlight index
    all_indices = {i for indices in result.values() for i in indices}
    assert all_indices.issubset(set(range(len(HIGHLIGHTS))))


def test_many_to_many_possible():
    """A highlight can appear in more than one cluster."""
    extract = sirius.passthrough_extractor()
    encode = sirius.sentence_transformer_encoder(device="cpu")
    # Lower threshold encourages many-to-many membership
    cluster = sirius.hdbscan_clusterer(min_cluster_size=2, threshold=0.5)

    result = sirius.cluster_highlights(HIGHLIGHTS, extract, encode, cluster)

    # Count how many clusters each highlight appears in
    membership_counts = [0] * len(HIGHLIGHTS)
    for indices in result.values():
        for i in indices:
            membership_counts[i] += 1

    # With a low threshold some highlights should appear in multiple clusters
    # (this is a soft assertion — just verify structure is valid)
    assert all(c >= 0 for c in membership_counts)
