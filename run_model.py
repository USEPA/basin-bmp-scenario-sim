#!/usr/bin/env python3
import sys
import yaml
from pathlib import Path

from src.logging_utils import make_logger
from src.io_utils import load_and_validate_all
from src.simulate import Simulator
from src.plotting import make_summary_plots

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_model.py path/to/config.yaml")
        sys.exit(1)
    cfg_path = Path(sys.argv[1])
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # normalize keys to lowercase (do not enforce case sensitivity)
    cfg = {str(k).lower(): v for k, v in cfg.items()}

    outputs_dir = Path(cfg.get("outputs", "./outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = make_logger(outputs_dir, verbose=bool(cfg.get("verbose", True)))
    logger.info("Starting model run")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Logging to: {log_path}")

    data = load_and_validate_all(cfg, logger)
    sim = Simulator(cfg, data, logger)
    scenario_records = sim.run_all_scenarios()

    # Produce summary plots per spec
    make_summary_plots(cfg, data, scenario_records, outputs_dir, logger)

    logger.info("Model run complete")

if __name__ == "__main__":
    main()
