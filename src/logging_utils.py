import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple

def make_logger(outputs_dir: Path, verbose: bool = True) -> Tuple[logging.Logger, Path]:
    """Create a logger that writes to a timestamped file and optionally stdout.

    The logger always records DEBUG-level detail to a file. If verbose is enabled,
    an INFO-level stream handler is also attached for console visibility.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = outputs_dir / f"log_{ts}.txt"

    logger = logging.getLogger("bmp_model")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # File handler writes all DEBUG and above messages to the timestamped log file.
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.debug("Logger initialized")
    return logger, log_path
