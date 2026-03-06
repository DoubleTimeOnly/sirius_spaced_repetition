"""Integration tests for Claude-based extractor and graph creator.

These tests call the real Anthropic API and are skipped by default.
Run with: pytest tests/test_integration.py --run-integration
Requires: ANTHROPIC_API_KEY environment variable
"""

import pytest

from sirius.extractors import claude_extractor
from sirius.graph_creators import claude_graph_creator
from sirius.protocols import ClusterMapping, Highlight, Highlights

_MODEL = "claude-haiku-4-5-20251001"

_HIGHLIGHTS: Highlights = [
    Highlight("Spaced repetition improves long-term retention."),
    Highlight("Testing yourself beats re-reading the same material."),
]
_CLUSTER_MAPPING: ClusterMapping = {0: {0, 1}}


@pytest.mark.integration
def test_claude_extractor_api():
    extract = claude_extractor(model=_MODEL)
    result = extract(_HIGHLIGHTS[0].text)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
def test_claude_graph_creator_api():
    create_graph = claude_graph_creator(model=_MODEL)
    result = create_graph(_CLUSTER_MAPPING, _HIGHLIGHTS)
    assert isinstance(result, dict)
    assert "nodes" in result
    assert "edges" in result
    assert isinstance(result["nodes"], list)
    assert isinstance(result["edges"], list)
    assert len(result["nodes"]) > 0
