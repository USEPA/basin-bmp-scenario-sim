import logging
from pathlib import Path
from typing import Optional, Tuple


def make_logger(
    outputs_dir: Path,
    verbose: bool = True,
    scenario_id: Optional[int] = None,
) -> Tuple[logging.Logger, Optional[Path]]:
    """Create the driver logger that optionally writes to stdout.

    If a scenario_id is provided, the logger writes to that scenario's log file.
    Otherwise it only logs to stdout and does not create a generic file.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_path = None
    if scenario_id is not None:
        log_path = outputs_dir / f"log_s{scenario_id}.txt"

    logger = logging.getLogger("bmp_model")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    if log_path is not None:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.debug("Driver logger initialized")
    return logger, log_path


def make_worker_logger(outputs_dir: Path, scenario_id: int) -> logging.Logger:
    """Create a per-scenario logger that writes all DEBUG lines to its own file.

    Workers call this to log into outputs/log_s{scenario_id}.txt.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"bmp_model.worker.scenario_{scenario_id}")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(outputs_dir / f"log_s{scenario_id}.txt", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Workers do not log to console to avoid interleaving lines.
    logger.propagate = False
    logger.debug(f"Worker logger initialized for scenario {scenario_id}")
    return logger