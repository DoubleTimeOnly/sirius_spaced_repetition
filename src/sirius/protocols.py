from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class Highlight:
    text: str
    '''Optional context around highlight in (before, after) format.'''
    context: tuple[str, str] | None = None

    def combine(self, bold_highlight: bool = False) -> str:
        if self.context:
            before, after = self.context
            text = self.text if not bold_highlight else f"**{self.text}**"
            return f"{before} {text} {after}"
        return self.text


Highlights = list[Highlight]
ClusterMapping = dict[Any, set[int]]  # cluster_key -> set of highlight indices

# Highlight -> core_info_string
ExtractFn = Callable[[Highlight], str]

# list of texts -> list of embedding vectors
EncodeFn = Callable[[list[str]], list[np.ndarray]]

# vectors -> {cluster_key: {highlight_indices}}
ClusterFn = Callable[[list[np.ndarray]], ClusterMapping]

# filepath -> highlights
HighlightParserFn = Callable[[str], Highlights]

# filepath -> clusters
PipelineFn = Callable[[str], ClusterMapping]

# (cluster_mapping, raw highlights) -> JSON Canvas dict
GraphCreatorFn = Callable[[ClusterMapping, Highlights], dict]
