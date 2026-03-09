# Background
I want to visualize my clusters in 3D space with an interactive plot. I want to be able to see each embedding as a point in space and its color is its cluster label.
The intent is to see what clusters vaguely look like, if clusters are close to each other, and understand the structure of the embedding space.

This should be an optional debug output within the pipeline. After the cluster mapping is made, the visualization process occurs and an interactive HTML file is saved to disk.

## Design

### Where visualization happens
Inside `cluster_highlights()` in `src/sirius/clustering.py`, after `cluster(vectors)` returns the `ClusterMapping`. Embeddings are available here as `vectors: list[np.ndarray]`.

Gated by a new `enable_cluster_visualization: bool` field on the pipeline config, defaulting to `false`.

### Data pipeline for visualization
1. Take `vectors` (original embeddings, already computed, passed to `cluster()`)
2. Fit UMAP with `n_components=3` to get 3D coordinates — always done fresh regardless of whether `umap_preprocessing` is active (the preprocessing UMAP may use different params/dims)
3. Invert `ClusterMapping` to get a per-point label: `point_label[i] = cluster_key` (first assignment wins for many-to-many points)

### Point labels
Every highlight is guaranteed to be in at least one cluster in `ClusterMapping` (solo clusters are created for HDBSCAN noise points). All points are colored by their cluster label via a categorical colormap.

### Visualization output
- Library: **Plotly** (`plotly.graph_objects.Scatter3d`) — saves as self-contained `.html`
- Interactive 3D scatter with hover tooltip showing truncated highlight text and cluster label
- Output file: `{run_dir}/cluster_visualization.html`

### Implementation approach

#### Config changes
- Add `enable_cluster_visualization: false` to `configs/pipeline/default.yaml`
- `cluster_highlights()` signature gets `enable_visualization: bool = False` and `run_dir: str | None = None`
- `pipeline.py` passes both from config + run_dir when calling `cluster_highlights()`

#### New module
`src/sirius/visualization.py` — single public function:
```python
def save_cluster_visualization(
    vectors: list[np.ndarray],
    cluster_mapping: ClusterMapping,
    highlights: Highlights,
    output_path: str,
) -> None
```
Keeps visualization logic isolated from clustering logic.

#### Critical files to modify
- `src/sirius/clustering.py` — add visualization call after `cluster(vectors)`
- `src/sirius/pipeline.py` — pass `enable_visualization` + `run_dir` to `cluster_highlights()`
- `src/sirius/visualization.py` — new file with `save_cluster_visualization()`
- `configs/pipeline/default.yaml` — add `enable_cluster_visualization: false`
- `pyproject.toml` — add `plotly` to dependencies (umap-learn already present as optional)

#### Dependencies
- `plotly` — add to main dependencies (lightweight, no GPU)
- `umap-learn` — already an optional dependency; must be installed for visualization to work

## Task Breakdown

**Phase 1: Setup (can run in parallel)**
1. **Add plotly dependency** — Update `pyproject.toml` to include plotly in main dependencies
2. **Create visualization module** — Implement `src/sirius/visualization.py` with `save_cluster_visualization()` function (takes vectors, cluster mapping, highlights, output path; produces 3D Plotly scatter HTML with hover tooltips)

**Phase 2: Config Changes (depends on Phase 1)**
3. **Update cluster_highlights signature** — Add `enable_visualization: bool = False` and `run_dir: str | None = None` parameters to `cluster_highlights()` in `src/sirius/clustering.py`
4. **Add config field** — Add `enable_cluster_visualization: false` to `configs/pipeline/default.yaml`

**Phase 3: Pipeline Integration (depends on Phase 2)**
5. **Integrate visualization in clustering** — After `cluster(vectors)` call in `cluster_highlights()`, conditionally call `save_cluster_visualization()` if `enable_visualization=True`
6. **Pass config through pipeline** — Update `pipeline.py` to extract `enable_cluster_visualization` and `run_dir` from config and pass them to `cluster_highlights()`

**Phase 4: Verification (depends on Phase 3)**
7. **Test with visualization enabled** — Run pipeline with `enable_cluster_visualization: true` and verify `cluster_visualization.html` appears with correct 3D points and hover tooltips
8. **Test with visualization disabled** — Run pipeline with `enable_cluster_visualization: false` (default) and verify no HTML file is created and no performance impact
9. **Run test suite** — Verify existing `pytest` tests pass without modification

## Dependency Tree

```
Phase 1 (parallel):
├── Task 1: Add plotly dependency
└── Task 2: Create visualization module
    ↓
Phase 2:
├── Task 3: Update cluster_highlights signature
└── Task 4: Add config field
    ↓
Phase 3:
├── Task 5: Integrate visualization in clustering
└── Task 6: Pass config through pipeline
    ↓
Phase 4:
├── Task 7: Test with visualization enabled
├── Task 8: Test with visualization disabled
└── Task 9: Run test suite
```

## Definition of Success

- A `cluster_visualization.html` file is written to the run output directory alongside `run.log` and `config.yaml`
- Each highlight appears as a point in 3D space; color encodes cluster membership
- Hover tooltips show truncated highlight text and cluster label
- The feature is off by default (`enable_cluster_visualization: false`)
- No performance regression when the feature is disabled
- Existing `pytest` suite passes without modification

## Verification
1. Run the pipeline with `enable_cluster_visualization: true` on a real highlights file
2. Confirm `cluster_visualization.html` appears in the run output directory
3. Open the HTML in a browser — points should be colored by cluster, interactive rotation/zoom works, hover shows highlight text
4. Run with `enable_cluster_visualization: false` (default) — no HTML file, no performance impact
5. Run `pytest` — existing tests pass without modification