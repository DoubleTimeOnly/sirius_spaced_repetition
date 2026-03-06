import os
from pathlib import Path

import pytest


_SECRETS_PATH = Path(__file__).parent.parent / "configs" / "secrets.yaml"


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that call external APIs",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests that call external APIs (skipped by default)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip = pytest.mark.skip(reason="Pass --run-integration to run")
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip)
    else:
        _load_secrets_into_env()


def _load_secrets_into_env() -> None:
    """Load API keys from configs/secrets.yaml into environment variables if not already set."""
    if not _SECRETS_PATH.exists():
        return
    from sirius.utils.hydra_utils import load_config
    cfg = load_config("secrets")
    secrets = cfg.get("secrets", {})
    if secrets.get("claude_api") and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = secrets["claude_api"]
