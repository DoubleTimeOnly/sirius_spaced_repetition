import logging
from typing import Any

import numpy as np
import plotly.graph_objects as go

from sirius.protocols import ClusterMapping, Highlights

logger = logging.getLogger(__name__)


def save_cluster_visualization(
    vectors: list[np.ndarray],
    cluster_mapping: ClusterMapping,
    highlights: Highlights,
    output_path: str,
) -> None:
    """Create and save an interactive 3D cluster visualization.

    Args:
        vectors: List of embedding vectors (original, pre-preprocessing).
        cluster_mapping: Mapping of cluster key -> set of highlight indices.
        highlights: List of Highlight objects.
        output_path: Path to save the HTML visualization file.
    """
    try:
        import umap
    except ImportError:
        logger.warning(
            "umap-learn not installed. Install with: uv sync --extra umap"
        )
        return

    # Fit UMAP with 3D reduction
    vectors_array = np.array(vectors)
    reducer = umap.UMAP(n_components=3, random_state=42)
    coords_3d = reducer.fit_transform(vectors_array)
    logger.debug(f"UMAP fit complete; 3D coordinates shape: {coords_3d.shape}")

    # Invert cluster mapping: point_index -> cluster_label
    # First assignment wins for many-to-many points
    point_labels: dict[int, Any] = {}
    for cluster_key, indices in cluster_mapping.items():
        for idx in indices:
            if idx not in point_labels:
                point_labels[idx] = cluster_key

    # Build point labels for all indices (unseen points default to "Unclustered")
    labels = [point_labels.get(i, "Unclustered") for i in range(len(vectors))]

    # Prepare hover text: truncated highlight text + cluster label
    hover_texts = []
    for i, highlight in enumerate(highlights):
        text_preview = highlight.text[:100].replace("<br>", " ")
        cluster_label = labels[i]
        hover_text = f"{text_preview}...<br>Cluster: {cluster_label}"
        hover_texts.append(hover_text)

    # Create 3D scatter plot
    fig = go.Figure()

    # Get unique labels for color mapping
    unique_labels = list(set(labels))
    label_to_color = {label: i for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]

    fig.add_trace(
        go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=colors,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Cluster",
                    tickvals=list(range(len(unique_labels))),
                    ticktext=[str(label) for label in unique_labels],
                ),
            ),
            text=hover_texts,
            hoverinfo="text",
            name="Highlights",
        )
    )

    fig.update_layout(
        title="3D Cluster Visualization",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
        hovermode="closest",
        width=1200,
        height=800,
    )

    fig.write_html(output_path)
    logger.info(f"Cluster visualization saved to {output_path}")
