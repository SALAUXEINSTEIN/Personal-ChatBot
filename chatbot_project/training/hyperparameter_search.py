"""
Targeted hyperparameter search — Section 3.6.3.

Grid (kept small deliberately, per the dissertation's compute-constraint
justification):
    learning_rate            in {1e-5, 5e-5, 1e-4}
    persona_consistency_wt   in {0.1, 0.3, 0.5}
    mixture_ratio (PC:DD)    in {70:30, 80:20}

Each configuration is trained briefly and evaluated on validation
perplexity + persona consistency; logs go to Weights & Biases if enabled.

This script only orchestrates config combinations and shells out to the
existing training entry points — it does not duplicate training logic.
"""

from __future__ import annotations
import itertools
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import SEARCH_CFG


def run_search(train_file: str, val_file: str, personachat_processed: str,
                dailydialog_processed: str, base_output_dir: str = "./checkpoints/search"):
    os.makedirs(base_output_dir, exist_ok=True)
    results = []

    for lr, persona_w, ratio in itertools.product(
        SEARCH_CFG.learning_rates, SEARCH_CFG.persona_loss_weights, SEARCH_CFG.mixture_ratios
    ):
        run_name = f"lr{lr}_pw{persona_w}_ratio{ratio}"
        run_dir = os.path.join(base_output_dir, run_name)
        print(f"\n=== Running configuration: {run_name} ===")

        combined_path = os.path.join(run_dir, "combined_train.jsonl")
        os.makedirs(run_dir, exist_ok=True)
        subprocess.run([
            sys.executable, "-m", "data.preprocess_dailydialog", "combine",
            "--personachat", personachat_processed,
            "--dailydialog", dailydialog_processed,
            "--out", combined_path,
            "--ratio", str(ratio),
        ], check=True)

        # NOTE: in a real run, override STAGE1_CFG.learning_rate / STAGE2_CFG.persona_consistency_weight
        # via environment variables or a config-override mechanism before invoking training.
        env = os.environ.copy()
        env["OVERRIDE_LR"] = str(lr)
        env["OVERRIDE_PERSONA_WEIGHT"] = str(persona_w)

        subprocess.run([
            sys.executable, "-m", "training.train_stage1",
            "--train_file", combined_path,
            "--val_file", val_file,
            "--output_dir", run_dir,
            "--use_wandb" if SEARCH_CFG.use_wandb else "",
        ], check=True, env=env)

        results.append({"run_name": run_name, "lr": lr, "persona_weight": persona_w,
                         "ratio": ratio, "output_dir": run_dir})

    print("\nHyperparameter search complete. Configurations run:")
    for r in results:
        print(r)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--personachat_processed", required=True)
    parser.add_argument("--dailydialog_processed", required=True)
    parser.add_argument("--output_dir", default="./checkpoints/search")
    args = parser.parse_args()

    run_search(args.train_file, args.val_file, args.personachat_processed,
               args.dailydialog_processed, args.output_dir)
