"""Tests for extractor factory functions."""

from unittest.mock import MagicMock

import pytest

from sirius.extractors import (
    claude_code_extractor,
    claude_extractor,
    passthrough_extractor,
)


def test_passthrough_extractor_passes_through_input():
    extract = passthrough_extractor()
    assert callable(extract)
    highlight = "Memory has storage strength and retrieval strength."
    assert extract(highlight) == highlight
    assert extract(highlight, None) == highlight
    assert extract(highlight, context="Learning science") == highlight
    assert extract("") == ""


def test_claude_code_extractor_returns_callable():
    assert callable(claude_code_extractor())
    assert callable(claude_code_extractor(model="sonnet"))


def test_claude_code_extractor_subprocess_call(monkeypatch):
    mock_result = MagicMock()
    mock_result.stdout = "\n  core info  \n"
    mock_run = MagicMock(return_value=mock_result)
    monkeypatch.setattr("sirius.extractors.subprocess.run", mock_run)

    extract = claude_code_extractor(model="sonnet")
    result = extract("Testing beats studying.")

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "claude" in cmd
    assert "sonnet" in cmd
    assert result == "core info"
