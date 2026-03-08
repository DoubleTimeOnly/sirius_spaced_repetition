# Context

Improve highlight embedding for clustering using Perplexity's contextual embedding model [pplx-embed-context-v1-0.6B](https://huggingface.co/perplexity-ai/pplx-embed-context-v1-0.6B), which produces embeddings aware of cross-highlight relationships (unlike the current independent embeddings from SentenceTransformers).

## High-Level Approach

1. **Change EncodeFn protocol to batch**: `Callable[[list[str]], list[np.ndarray]]` instead of single-string encoding. This allows the contextual model to process all highlights together.

2. **Load model locally via HuggingFace**: Use `AutoModel.from_pretrained("perplexity-ai/pplx-embed-context-v1-0.6B", trust_remote_code=True)`. The model's custom `.encode()` method handles "late chunking" internally — it joins all chunks with SEP tokens, runs a single forward pass (so chunks influence each other via attention), then mean-pools per-chunk spans.

3. **All highlights = one "document"**: Pass all N highlight texts as a single document `[[chunk1, chunk2, ...N]]` to the model. It returns a numpy array of shape `(N, 1024)` with cross-chunk aware embeddings.

4. **Use context-enriched text**: In `clustering.py`, pass `[h.combine() for h in highlights]` to the encoder (which includes surrounding before/after sentences), not just the extracted core info.

## Files to Modify

- `src/sirius/protocols.py` — change `EncodeFn` type alias to batch
- `src/sirius/encoders.py` — update `sentence_transformer_encoder` + add `contextual_encoder` factory
- `src/sirius/clustering.py` — batch encode call + use `h.combine()` for context
- `tests/test_encoders.py` — update for batch signature
- `tests/test_clustering.py` — update for batch signature

## New Files

- `configs/encoder/contextual.yaml` — Hydra config for the contextual encoder

## Success Criteria

1. Batch encoding works: both `sentence_transformer_encoder` and `contextual_encoder` accept `list[str]` and return `list[np.ndarray]`
2. Tests pass: `pytest tests/test_encoders.py tests/test_clustering.py`
3. Smoke test: `python experiments/2-full-highlights/process_highlights.py` with contextual encoder in config produces valid clusters

## Task Breakdown

### Dependency Tree

```
Task 1: Update EncodeFn protocol signature
    ├── Task 2: Update sentence_transformer_encoder to batch
    │   ├── Task 5: Update clustering.py for batch encoding
    │   │   └── Task 7: Update test_clustering.py
    │   │       └── Task 8: Run smoke test
    │   └── Task 6: Update test_encoders.py
    │       └── Task 8: Run smoke test
    └── Task 3: Implement contextual_encoder factory
        ├── Task 4: Create contextual.yaml config
        │   └── Task 8: Run smoke test
        └── Task 6: Update test_encoders.py
            └── Task 8: Run smoke test
```

**Critical path**: 1 → 2 → 5 → 7 → 8

### Individual Tasks

| # | Task | Summary |
|---|------|---------|
| 1 | Update EncodeFn protocol | Change `EncodeFn` type alias in `src/sirius/protocols.py` from `Callable[[str], np.ndarray]` to `Callable[[list[str]], list[np.ndarray]]`. |
| 2 | Update sentence_transformer_encoder | Modify the existing encoder in `src/sirius/encoders.py` to accept a list of strings and return a list of embeddings instead of processing one string at a time. |
| 3 | Implement contextual_encoder | Add a new factory in `src/sirius/encoders.py` that loads the Perplexity contextual model and implements batch encoding with internal late chunking. |
| 4 | Create contextual.yaml config | Create `configs/encoder/contextual.yaml` with Hydra configuration pointing to the `contextual_encoder` factory. |
| 5 | Update clustering.py | Modify `clustering.py` to call encoders with batched highlights using `h.combine()` for context-enriched text instead of individual calls. |
| 6 | Update test_encoders.py | Update test cases in `tests/test_encoders.py` to validate both encoder factories work with the new batch signature. |
| 7 | Update test_clustering.py | Update test cases in `tests/test_clustering.py` to work with the new batch encoding in the clustering pipeline. |
| 8 | Run smoke test | Execute `python experiments/2-full-highlights/process_highlights.py` with contextual encoder config to validate the full pipeline works end-to-end. |
