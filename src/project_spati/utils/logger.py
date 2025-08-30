import logging
import sys
from typing import Optional

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get configured logger instance

    Args:
        name (str, optional): Logger name

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name or __name__)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
