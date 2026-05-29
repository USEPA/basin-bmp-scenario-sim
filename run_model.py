"""
Command-line runner for the BMP scenario model.

Supports:
- Running scenarios based on a YAML config
- Generating per-scenario CSVs and plots
- Optionally consolidating all transposed summaries into a single CSV
- A consolidate-only mode that performs only the consolidation step
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

from src.constants import CFG_OUTPUTS, CFG_RANDOM_SEED, CFG_VERBOSE
from src.io_utils import load_and_validate_all, consolidate_transposed_summaries
from src.logging_utils import make_logger
from src.model import Model
from src.plotting import make_summary_plots


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Run the BMP scenario model using a YAML configuration file.")
    p.add_argument("config", help="Path to the YAML configuration file")
    p.add_argument("--outputs", help="Override the outputs directory from config")
    p.add_argument("--seed", type=int, help="Override random seed from config")
    p.add_argument("--quiet", action="store_true", help="Disable console logging")
    p.add_argument(
        "--consolidate",
        action="store_true",
        help="After running, consolidate per-scenario transposed summaries into outputs/summaries/all_scenarios.csv",
    )
    p.add_argument(
        "--consolidate-only",
        action="store_true",
        help="Only consolidate transposed per-scenario summaries and exit (no scenarios executed)",
    )
    p.add_argument("--version", action="version", version="basin-bmp-sim 0.1.0")
    return p.parse_args()


def main() -> None:
    """Entrypoint: load config, run scenarios or consolidate summaries."""
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: input yaml file not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = {str(k).lower(): v for k, v in cfg.items()}

    outputs_dir = Path(args.outputs or cfg.get(CFG_OUTPUTS, "./outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    verbose = False if args.quiet else bool(cfg.get(CFG_VERBOSE, True))
    logger, log_path = make_logger(outputs_dir, verbose=verbose)
    logger.info("Starting model run")
    logger.info(f"Config: {cfg_path}")
    if log_path is not None:
        logger.info(f"Logging to: {log_path}")
    else:
        logger.info("Logging only to console; per-scenario log files will be created.")

    if args.consolidate_only:
        consolidate_transposed_summaries(outputs_dir, logger)
        logger.info("Consolidation-only mode complete")
        return

    if args.seed is not None:
        cfg[CFG_RANDOM_SEED] = args.seed

    data = load_and_validate_all(cfg, logger)

    sim = Model(cfg, data, logger)
    scenario_records = sim.run_all_scenarios()

    make_summary_plots(cfg, data, scenario_records, outputs_dir, logger)

    if args.consolidate:
        consolidate_transposed_summaries(outputs_dir, logger)

    logger.info("Model run complete")


if __name__ == "__main__":
    main()