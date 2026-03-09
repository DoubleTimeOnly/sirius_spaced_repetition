"""Tests for encoder factory functions."""

import numpy as np
import pytest

from sirius.encoders import sentence_transformer_encoder, contextual_encoder


@pytest.fixture(scope="module")
def sentence_transformer_encoder_instance():
    """Shared SentenceTransformer encoder to avoid loading the model multiple times per module."""
    return sentence_transformer_encoder(model="all-MiniLM-L6-v2", device="cpu")


def test_sentence_transformer_encoder_returns_callable():
    assert callable(sentence_transformer_encoder(model="all-MiniLM-L6-v2", device="cpu"))


def test_sentence_transformer_encoder_output_format(sentence_transformer_encoder_instance):
    texts = ["Short.", "A much longer sentence with many more words in it."]
    vectors = sentence_transformer_encoder_instance(texts)
    assert isinstance(vectors, list) and len(vectors) == 2
    v1, v2 = vectors
    assert isinstance(v1, np.ndarray) and v1.ndim == 1 and np.issubdtype(v1.dtype, np.floating)
    assert v1.shape == v2.shape


def test_sentence_transformer_encoder_encoding_properties(sentence_transformer_encoder_instance):
    text = "Memory has storage strength and retrieval strength."
    vectors = sentence_transformer_encoder_instance([text])
    vectors_again = sentence_transformer_encoder_instance([text])
    np.testing.assert_array_equal(vectors[0], vectors_again[0])  # deterministic

    v1_batch = sentence_transformer_encoder_instance(["Spaced repetition improves long-term retention."])
    v2_batch = sentence_transformer_encoder_instance(["The mitochondria is the powerhouse of the cell."])
    assert not np.allclose(v1_batch[0], v2_batch[0])  # discriminative


def test_contextual_encoder_returns_callable():
    assert callable(contextual_encoder(model="perplexity-ai/pplx-embed-context-v1-0.6B", device="cpu"))


@pytest.fixture(scope="module")
def contextual_encoder_instance():
    """Shared contextual encoder to avoid loading the model multiple times per module."""
    return contextual_encoder(model="perplexity-ai/pplx-embed-context-v1-0.6B", device="cpu")


def test_contextual_encoder_output_format(contextual_encoder_instance):
    texts = ["Short.", "A much longer sentence with many more words in it."]
    vectors = contextual_encoder_instance(texts)
    assert isinstance(vectors, list) and len(vectors) == 2
    v1, v2 = vectors
    assert isinstance(v1, np.ndarray) and v1.ndim == 1 and np.issubdtype(v1.dtype, np.floating)
    # Contextual model uses 1024-dim embeddings
    assert v1.shape == (1024,) and v2.shape == (1024,)


def test_contextual_encoder_encoding_properties(contextual_encoder_instance):
    text = "Memory has storage strength and retrieval strength."
    vectors = contextual_encoder_instance([text])
    vectors_again = contextual_encoder_instance([text])
    np.testing.assert_array_equal(vectors[0], vectors_again[0])  # deterministic

    v1_batch = contextual_encoder_instance(["Spaced repetition improves long-term retention."])
    v2_batch = contextual_encoder_instance(["The mitochondria is the powerhouse of the cell."])
    assert not np.allclose(v1_batch[0], v2_batch[0])  # discriminative


def test_contextual_encoder_batching(contextual_encoder_instance):
    """Test that batching works correctly with large numbers of texts."""
    # Create more than 100 texts to trigger batching (batch_size=100)
    texts = [f"This is highlight number {i}. It contains some text." for i in range(150)]

    # Encode all texts at once
    vectors = contextual_encoder_instance(texts)

    # Verify output format
    assert isinstance(vectors, list) and len(vectors) == 150
    for v in vectors:
        assert isinstance(v, np.ndarray) and v.ndim == 1 and v.shape == (1024,)
        assert np.issubdtype(v.dtype, np.floating)

    # Verify determinism - same input should produce same output
    vectors_again = contextual_encoder_instance(texts)
    assert len(vectors_again) == len(vectors)
    for v1, v2 in zip(vectors, vectors_again):
        np.testing.assert_array_equal(v1, v2)

    # Verify that different texts produce different embeddings
    assert not np.allclose(vectors[0], vectors[1])
