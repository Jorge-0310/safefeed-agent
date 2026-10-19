"""Calibra la penalización de publicación en una semilla de validación simulada."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MODEL_PATH, REPORT_DIR, AgentConfig, UtilityConfig
from .metrics import calibrate_utility_policy
from .runner import run_simulation
from .train import ensure_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_DIR / "policy_calibration.json")
    args = parser.parse_args()

    validation = run_simulation(
        n_steps=args.steps,
        seed=args.seed,
        model=ensure_model(args.model_path),
        agent_config=AgentConfig(learning_enabled=False),
        utility_config=UtilityConfig(),
        log_dir=None,
    )
    result = calibrate_utility_policy(validation["records"])
    result["seed"] = args.seed
    result["n_steps"] = args.steps
    result["configured_publish_harmful"] = UtilityConfig().publish_harmful
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
