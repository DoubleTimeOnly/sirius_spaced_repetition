import numpy as np

from .protocols import EncodeFn


def sentence_transformer_encoder(model: str = "all-MiniLM-L6-v2") -> EncodeFn:
    """Return an encoder backed by a local SentenceTransformer model.

    Args:
        model: HuggingFace model name. Defaults to all-MiniLM-L6-v2 (~80 MB,
               fast, strong semantic similarity).
    """
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(model)

    def encode(text: str) -> np.ndarray:
        return st_model.encode(text)

    return encode
