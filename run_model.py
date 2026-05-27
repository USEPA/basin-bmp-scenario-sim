import argparse
from pathlib import Path
import sys
import yaml

from src.constants import CFG_OUTPUTS, CFG_RANDOM_SEED, CFG_VERBOSE
from src.logging_utils import make_logger
from src.io_utils import load_and_validate_all
from src.model import Model
from src.plotting import make_summary_plots


def parse_args() -> argparse.Namespace:
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
    """Entry point for the BMP simulation model."""
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: input yaml file not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # normalize keys to lowercase
    cfg = {str(k).lower(): v for k, v in cfg.items()}

    if args.seed is not None:
        cfg[CFG_RANDOM_SEED] = args.seed

    outputs_dir = Path(args.outputs or cfg.get(CFG_OUTPUTS, "./outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    verbose = False if args.quiet else bool(cfg.get(CFG_VERBOSE, True))
    logger, log_path = make_logger(outputs_dir, verbose=verbose)
    logger.info("Starting model run")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Logging to: {log_path}")

    data = load_and_validate_all(cfg, logger)

    # Run simulation (workers will write per-scenario CSV + log files)
    sim = Model(cfg, data, logger)
    scenario_records = sim.run_all_scenarios()

    # Plots
    make_summary_plots(cfg, data, scenario_records, outputs_dir, logger)

    logger.info("Model run complete")


if __name__ == "__main__":
    main()