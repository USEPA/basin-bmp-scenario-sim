"""Run the model from the command line using a YAML settings file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.constants import CFG_OUTPUTS, CFG_RANDOM_SEED, CFG_VERBOSE
from src.io_utils import read_config
from src.input_config import load_and_validate_all, normalize_config
from src.input_validation import validate_config
from src.logging_utils import make_logger
from src.model import Model
from src.plotting import make_summary_plots


def parse_args() -> argparse.Namespace:
    """Read command-line options for this script.

        Returns
        -------
        argparse.Namespace
            Parsed command-line arguments.
        
    """
    parser = argparse.ArgumentParser(
        description="Run the BMP scenario model using a YAML configuration file."
    )
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument("--outputs", help="Override the outputs directory from config")
    parser.add_argument("--seed", type=int, help="Override random seed from config")
    parser.add_argument("--quiet", action="store_true", help="Disable console logging")
    parser.add_argument("--version", action="version", version="basin-bmp-sim 0.1.0")
    return parser.parse_args()


def main() -> None:
    """Load settings, run scenarios, and make summary plots."""
    args = parse_args()
    cfg_path = Path(args.config)
    try:
        cfg = normalize_config(read_config(cfg_path))
        validate_config(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if args.outputs is not None:
        cfg[CFG_OUTPUTS] = args.outputs
    if args.seed is not None:
        cfg[CFG_RANDOM_SEED] = args.seed

    outputs_dir = Path(cfg[CFG_OUTPUTS])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 'verbose' controls file verbosity; --quiet only disables console output.
    verbose = bool(cfg[CFG_VERBOSE])
    logger, log_path = make_logger(outputs_dir, verbose=verbose, console=not args.quiet)
    logger.info("Starting model run")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Logging to: {log_path}")

    data = load_and_validate_all(cfg, logger)

    sim = Model(cfg, data, logger)
    scenario_records = sim.run_all_scenarios()

    make_summary_plots(cfg, data, scenario_records, outputs_dir, logger)

    logger.info("Model run complete")


if __name__ == "__main__":
    main()
