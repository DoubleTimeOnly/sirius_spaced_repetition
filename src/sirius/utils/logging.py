import logging


def set_logging_level(level: str):
    # Configure logging level from config (only for this repository)
    logging.basicConfig(level=logging.WARNING)
    logging_level = getattr(logging, level.upper(), logging.DEBUG)
    logging.getLogger("sirius").setLevel(logging_level)
