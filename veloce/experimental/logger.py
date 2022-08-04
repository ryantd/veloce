import sys

from loguru import logger


MAIN_FORMAT = (
    "{time:HH:mm:ss} {extra[component]}.{extra[index]} | "
    "<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> <lvl>{level}</lvl> [<c>{file}</c>:<c>{line}</c>] "
    "<level>{message}</level>"
)

def set_default_logger():
    logger.remove()
    logger.add(sys.stdout, format=f"{MAIN_FORMAT}")
    return logger


def get_logger(name: str, index: int = 0):
    logger = set_default_logger()
    logger = logger.bind(component=name, index=index)
    return logger
