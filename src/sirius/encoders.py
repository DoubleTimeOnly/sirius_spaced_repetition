import numpy as np

from .protocols import EncodeFn


def sentence_transformer_encoder(
    model: str = "all-MiniLM-L6-v2",
    device: str = "auto",
) -> EncodeFn:
    """Return an encoder backed by a local SentenceTransformer model.

    Args:
        model: HuggingFace model name. Defaults to all-MiniLM-L6-v2 (~80 MB,
               fast, strong semantic similarity).
        device: Device to run the model on. "auto" uses GPU if available,
                otherwise falls back to CPU. Explicit values: "cpu", "cuda",
                "cuda:0", "mps", etc.
    """
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(model, device=device)

    def encode(text: str) -> np.ndarray:
        return st_model.encode(text)

    return encode
