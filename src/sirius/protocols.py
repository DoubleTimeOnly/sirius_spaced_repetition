from typing import Any, Callable

import numpy as np

Highlights = list[str]
ClusterMapping = dict[Any, set[int]]  # cluster_key -> set of highlight indices

# highlight, optional_context -> core_info_string
ExtractFn = Callable[[str, str | None], str]

# text -> embedding vector
EncodeFn = Callable[[str], np.ndarray]

# vectors -> {cluster_key: {highlight_indices}}
ClusterFn = Callable[[list[np.ndarray]], ClusterMapping]

# filepath -> highlights
HighlightParserFn = Callable[[str], Highlights]

# filepath -> clusters
PipelineFn = Callable[[str], ClusterMapping]

# (cluster_mapping, raw highlights) -> JSON Canvas dict
GraphCreatorFn = Callable[[ClusterMapping, Highlights], dict]
