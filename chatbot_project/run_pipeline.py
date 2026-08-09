"""
End-to-end pipeline runner — ties together every stage described in
Chapter 3, for a single command that reproduces the full project on a
small sample (use --limit to control size; omit it for the full
dissertation-scale run, which requires a GPU with the Colab-tier specs
mentioned in Section 3.6.1).

    python run_pipeline.py --limit 500

Steps:
    1. Pre-process PersonaChat (train/val/test)
    2. Pre-process DailyDialog (train/val/test)
    3. Build the combined 70:30 training corpus
    4. Stage 1: fine-tune the backbone
    5. Stage 2: jointly train the UPE + DST
    6. Run the ablation study (B, V1, V2, V3, FS) on the test set
"""

from __future__ import annotations
import argparse
import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap examples per split for a fast local smoke test")
    parser.add_argument("--skip_training", action="store_true",
                         help="Skip Stages 1/2 (useful if checkpoints already exist)")
    args = parser.parse_args()

    limit_args = ["--limit", str(args.limit)] if args.limit else []

    # 1 & 2. Pre-process both datasets
    for split in ["train", "validation", "test"]:
        run([sys.executable, "-m", "data.preprocess_personachat", "--split", split] + limit_args)
        run([sys.executable, "-m", "data.preprocess_dailydialog", "preprocess", "--split", split] + limit_args)

    # 3. Build combined corpus (train + validation)
    for split, out_name in [("train", "combined_train.jsonl"), ("validation", "combined_val.jsonl")]:
        run([
            sys.executable, "-m", "data.preprocess_dailydialog", "combine",
            "--personachat", f"data/processed/personachat_{split}.jsonl",
            "--dailydialog", f"data/processed/dailydialog_{split}.jsonl",
            "--out", f"data/processed/{out_name}",
        ])

    if not args.skip_training:
        # 4. Stage 1
        run([
            sys.executable, "-m", "training.train_stage1",
            "--train_file", "data/processed/combined_train.jsonl",
            "--val_file", "data/processed/combined_val.jsonl",
        ])

        # 5. Stage 2
        run([
            sys.executable, "-m", "training.train_stage2",
            "--backbone_checkpoint", "checkpoints/stage1_backbone",
            "--train_file", "data/processed/personachat_train.jsonl",
            "--val_file", "data/processed/personachat_validation.jsonl",
        ])

    # 6. Ablation study
    run([
        sys.executable, "-m", "evaluation.ablation_study",
        "--backbone_checkpoint", "checkpoints/stage1_backbone",
        "--stage2_checkpoint", "checkpoints/stage2_full_system",
        "--test_file", "data/processed/personachat_test.jsonl",
    ] + (["--limit", str(args.limit)] if args.limit else []))

    print("\nPipeline complete. Launch the demo with:\n"
          "  python -m app.gradio_app --backbone_checkpoint checkpoints/stage1_backbone "
          "--stage2_checkpoint checkpoints/stage2_full_system")


if __name__ == "__main__":
    main()
