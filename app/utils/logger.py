import sys
from loguru import logger

# Remove default logger
logger.remove()

# Console logger — colored, readable
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
           "<level>{message}</level>",
    level="DEBUG"
)

# File logger — saves all logs to file
logger.add(
    "logs/app.log",
    rotation="10 MB",       # New file after 10MB
    retention="7 days",     # Keep logs for 7 days
    compression="zip",      # Compress old logs
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
)

def get_logger(name: str):
    return logger.bind(name=name)