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

    st_model = SentenceTransformer(model, device=device, trust_remote_code=True)

    def encode(texts: list[str]) -> list[np.ndarray]:
        """Encode a batch of texts to embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            List of 1D numpy arrays, one per text.
        """
        embeddings = st_model.encode(texts)
        # st_model.encode returns 2D array (n_texts, embedding_dim)
        # Convert to list of 1D arrays
        return list(embeddings)

    return encode


def contextual_encoder(
    model: str = "perplexity-ai/pplx-embed-context-v1-0.6B",
    device: str = "auto",
) -> EncodeFn:
    """Return an encoder backed by Perplexity's contextual embedding model.

    Uses late chunking: all highlights are passed as a single "document" to enable
    cross-highlight attention. The model internally joins chunks with SEP tokens,
    performs a single forward pass, and mean-pools embeddings per chunk.

    Args:
        model: HuggingFace model name. Defaults to perplexity-ai/pplx-embed-context-v1-0.6B.
        device: Device to run the model on. "auto" uses GPU if available,
                otherwise falls back to CPU. Explicit values: "cpu", "cuda",
                "cuda:0", "mps", etc.
    """
    from transformers import AutoModel

    # Load model with trust_remote_code=True to support custom tokenization
    llm_model = AutoModel.from_pretrained(model, trust_remote_code=True)
    llm_model = llm_model.to(device)
    llm_model.eval()

    def encode(texts: list[str]) -> list[np.ndarray]:
        """Encode a batch of texts to embeddings with cross-highlight context.

        Args:
            texts: List of text strings (highlights) to encode. All are treated
                   as a single "document" for late chunking.

        Returns:
            List of 1D numpy arrays, one per text. Each array has shape (1024,).
        """
        # Perplexity model expects: [[chunk1, chunk2, ..., chunkN]]
        # (list of documents, each document is a list of chunks)
        # Process in batches to avoid OOM
        batch_size = 100
        batch_embeddings_list = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = llm_model.encode([batch], quantization="binary")[0]
            # batch_embeddings shape: (len(batch), 1024)
            batch_embeddings_list.append(batch_embeddings)

        # Concatenate all batch embeddings, then convert to list of 1D arrays
        all_embeddings = np.concatenate(batch_embeddings_list, axis=0)
        return list(all_embeddings)

    return encode
