import sys

from loguru import logger


MAIN_FORMAT = (
    "<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> | {extra[component]} | "
    "<lvl>{level}</lvl> | <c>{file}</c>:<c>{function}</c>:<c>{line}</c> - "
    "<level>{message}</level>"
)

def set_default_logger():
    logger.remove()
    # logger.add(sys.stderr, format=f"{MAIN_FORMAT}", level="ERROR", diagnose=True)
    logger.add(sys.stdout, format=f"{MAIN_FORMAT}")
    return logger


def get_logger(name: str):
    logger = set_default_logger()
    logger = logger.bind(component=name)
    return logger
