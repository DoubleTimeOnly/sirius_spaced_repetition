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

PipelineFn = Callable[[Highlights], ClusterMapping]
