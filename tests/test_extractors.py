"""Tests for extractor factory functions."""

from unittest.mock import MagicMock

import pytest

from sirius.extractors import (
    claude_code_extractor,
    claude_extractor,
    passthrough_extractor,
)
from sirius.protocols import Highlight


def test_passthrough_extractor_passes_through_input():
    extract = passthrough_extractor()
    assert callable(extract)
    highlight = Highlight(text="Memory has storage strength and retrieval strength.")
    result = extract(highlight)
    assert result == "Memory has storage strength and retrieval strength."

    highlight_with_context = Highlight(
        text="core concept",
        context=("Before text", "After text")
    )
    result_with_context = extract(highlight_with_context)
    assert result_with_context == "Before text **core concept** After text"

    empty_highlight = Highlight(text="")
    assert extract(empty_highlight) == ""


def test_claude_code_extractor_returns_callable():
    assert callable(claude_code_extractor())
    assert callable(claude_code_extractor(model="sonnet"))


def test_claude_code_extractor_subprocess_call(monkeypatch):
    mock_result = MagicMock()
    mock_result.stdout = "\n  core info  \n"
    mock_run = MagicMock(return_value=mock_result)
    monkeypatch.setattr("sirius.extractors.subprocess.run", mock_run)

    extract = claude_code_extractor(model="sonnet")
    highlight = Highlight(text="Testing beats studying.")
    result = extract(highlight)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "claude" in cmd
    assert "sonnet" in cmd
    assert result == "core info"
