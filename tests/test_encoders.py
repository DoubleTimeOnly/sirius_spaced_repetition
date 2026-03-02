"""Tests for encoder factory functions."""

import numpy as np
import pytest

from sirius.encoders import sentence_transformer_encoder


@pytest.fixture(scope="module")
def encoder():
    """Shared encoder to avoid loading the model multiple times per module."""
    return sentence_transformer_encoder(model="all-MiniLM-L6-v2", device="cpu")


def test_sentence_transformer_encoder_returns_callable():
    assert callable(sentence_transformer_encoder(model="all-MiniLM-L6-v2", device="cpu"))


def test_sentence_transformer_encoder_output_format(encoder):
    v1 = encoder("Short.")
    v2 = encoder("A much longer sentence with many more words in it.")
    assert isinstance(v1, np.ndarray) and v1.ndim == 1 and np.issubdtype(v1.dtype, np.floating)
    assert v1.shape == v2.shape


def test_sentence_transformer_encoder_encoding_properties(encoder):
    text = "Memory has storage strength and retrieval strength."
    np.testing.assert_array_equal(encoder(text), encoder(text))  # deterministic
    assert not np.allclose(
        encoder("Spaced repetition improves long-term retention."),
        encoder("The mitochondria is the powerhouse of the cell."),
    )  # discriminative
