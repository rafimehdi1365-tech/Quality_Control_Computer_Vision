import logging
import sys

class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

def get_logger(name):
    """ایجاد یک logger رنگی برای خروجی"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
        handler.setFormatter(ColorFormatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
