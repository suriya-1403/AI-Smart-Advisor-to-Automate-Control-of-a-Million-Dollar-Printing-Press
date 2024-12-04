import logging
import os

def get_logger(name: str, log_file: str = "app.log"):
    """
    Configure and return a logger instance.

    :param name: Name of the logger, usually __name__ from the calling module.
    :param log_file: Path to the log file. Defaults to 'app.log'.
    :return: Configured logger instance.
    """
    # Ensure the logs directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the default logging level

    # Create handlers
    file_handler = logging.FileHandler(log_file, mode="a")  # Log to a file
    console_handler = logging.StreamHandler()  # Log to the console

    # Set log formats
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.hasHandlers():  # Avoid adding multiple handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
