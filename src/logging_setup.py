import logging
import sys

_CONFIG = {"level": logging.INFO, "fmt": "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"}


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("nove")
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_CONFIG["fmt"]))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def set_verbose(level: int = logging.DEBUG) -> None:
    setup_logging()
    logger = logging.getLogger("nove")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
