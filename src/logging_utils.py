"""
Logging helpers for driver and worker processes.

- Driver logger optionally writes to console and/or a file.
- Worker loggers write a dedicated file per scenario under outputs/logs/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple


def make_logger(outputs_dir: Path, verbose: bool = True, scenario_id: Optional[int] = None) -> Tuple[logging.Logger, Optional[Path]]:
    """Create a driver logger.

    Parameters
    ----------
    outputs_dir : Path
        Root outputs directory.
    verbose : bool, default True
        If True, also log to console at INFO level.
    scenario_id : Optional[int]
        If provided, the driver also logs to outputs/logs/s{scenario_id}.txt.

    Returns
    -------
    (logging.Logger, Optional[Path])
        The logger and optional log file path if scenario_id is provided.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_path = None
    if scenario_id is not None:
        logs_dir = outputs_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"s{scenario_id}.txt"

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
    """Create a per-scenario logger writing into outputs/logs/s{scenario_id}.txt.

    Parameters
    ----------
    outputs_dir : Path
        Root outputs directory.
    scenario_id : int
        1-based scenario id.

    Returns
    -------
    logging.Logger
        Logger instance dedicated to this scenario.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"bmp_model.worker.scenario_{scenario_id}")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logs_dir / f"s{scenario_id}.txt", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Workers do not log to console to avoid interleaving lines.
    logger.propagate = False
    logger.debug(f"Worker logger initialized for scenario {scenario_id}")
    return logger