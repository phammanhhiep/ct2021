from sys import stdout
import logging


def create_root_logger(level=logging.INFO):
    FORMAT = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    logger = logging.getLogger()
    handler = logging.StreamHandler(stdout)
    handler.setFormatter(FORMAT)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
