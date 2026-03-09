# Background

I'd like to have a pre-processing step before clustering to perform dimensionality reduction using [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html).
The goal is to take the input embeddings which are high-dimensional (e.g. d = 1024) and reduce them down to a dimension of something like 10 or 50.
The hope is that it improves clustering performance.

# Success Criteria

The pipeline should:
- Run **both with and without UMAP** (optional stage)
- Have sensible parameter defaults
- Allow adjustment of parameters via Hydra config
- Especially allow specifying the reduced dimension
- Pass all tests without errors

# Implementation Details

## Architecture

UMAP will be added as a new `ClusteringPreprocessingFn` stage in the pipeline:

```
HighlightParser → ExtractFn → EncodeFn → ClusteringPreprocessingFn → ClusterFn → ClusterMapping
```

## Components

### Protocol (in `src/sirius/protocols.py`)
```python
ClusteringPreprocessingFn = Callable[[list[np.ndarray]], list[np.ndarray]]
```

### Implementations (in `src/sirius/clustering_preprocessors.py`)

1. **Passthrough** - returns vectors unchanged
   ```python
   def passthrough_preprocessing() -> ClusteringPreprocessingFn:
       def _passthrough(vectors: list[np.ndarray]) -> list[np.ndarray]:
           return vectors
       return _passthrough
   ```

2. **UMAP** - dimensionality reduction
   ```python
   def umap_preprocessing(
       n_components: int = 10,
       n_neighbors: int = 15,
       min_dist: float = 0.1,
       metric: str = "euclidean",
       random_state: int = 42,
   ) -> ClusteringPreprocessingFn:
       # Fits UMAP on the vectors and applies transform
   ```

### Config Files

- `configs/clustering_preprocessing/none.yaml` - passthrough (default)
- `configs/clustering_preprocessing/umap.yaml` - UMAP with defaults

### Pipeline Integration

- Update `src/sirius/pipeline.py:create_pipeline_fn()` to instantiate `pipeline_cfg.clustering_preprocessing`
- Update `src/sirius/clustering.py:cluster_highlights()` to call the preprocessing function after encoding, before clustering
- Update `configs/pipeline/default.yaml` to include the new stage with default set to passthrough

