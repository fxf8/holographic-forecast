import os
import sys
import logging


def get_logger(
    name: str,
    log_path: str,
):
    os.makedirs("logs", exist_ok=True)
    open("logs/collection-test.log", "w").close()

    logger = logging.getLogger(name)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )

    return logger
