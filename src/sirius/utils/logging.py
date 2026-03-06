import logging
from pathlib import Path


def set_logging_level(level: str):
    # Configure logging level from config (only for this repository)
    logging.basicConfig(level=logging.WARNING)
    logging_level = getattr(logging, level.upper(), logging.DEBUG)
    logging.getLogger("sirius").setLevel(logging_level)


def add_file_handler(log_file: Path):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger("sirius").addHandler(handler)
