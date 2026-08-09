"""
Ablation study — Section 3.6.4.

Compares five system variants on the same held-out PersonaChat test
set:
    B  - zero-shot DialoGPT-medium (no fine-tuning)
    V1 - fine-tuned backbone only
    V2 - fine-tuned backbone + DST only
    V3 - fine-tuned backbone + UPE only
    FS - fine-tuned backbone + UPE + DST (full system)

Usage:
    python -m evaluation.ablation_study \
        --backbone_checkpoint checkpoints/stage1_backbone \
        --stage2_checkpoint checkpoints/stage2_full_system \
        --test_file data/processed/personachat_test.jsonl \
        --limit 200
"""

from __future__ import annotations
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import load_backbone_and_tokenizer, load_finetuned_backbone
from models.personalised_chatbot import build_variant
from evaluation.automatic_metrics import evaluate_variant
from configs.config import EVAL_CFG


def load_test_records(path: str, limit: int = None):
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records[:limit] if limit else records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_checkpoint", required=True,
                         help="Stage-1 fine-tuned backbone directory")
    parser.add_argument("--stage2_checkpoint", required=False,
                         help="Directory with upe.pt / dst.pt from Stage 2 (needed for V2/V3/FS)")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--limit", type=int, default=200,
                         help="Number of test examples (dissertation uses full 14,878; "
                              "reduce for quick local runs)")
    parser.add_argument("--output_json", default="./evaluation/ablation_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_records = load_test_records(args.test_file, args.limit)

    nli_model_name = EVAL_CFG.nli_model_name
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
    nli_model.eval()

    all_results = {}

    for variant in EVAL_CFG.ablation_variants:
        print(f"\n=== Evaluating variant: {variant} ===")

        if variant == "B":
            backbone, tokenizer = load_backbone_and_tokenizer()
        else:
            backbone, tokenizer = load_finetuned_backbone(args.backbone_checkpoint, device)

        system = build_variant(variant, backbone, tokenizer).to(device)

        if variant in ("V2", "V3", "FS") and args.stage2_checkpoint:
            if system.use_upe:
                system.upe.load_state_dict(
                    torch.load(os.path.join(args.stage2_checkpoint, "upe.pt"), map_location=device))
            if system.use_dst:
                system.dst.load_state_dict(
                    torch.load(os.path.join(args.stage2_checkpoint, "dst.pt"), map_location=device))

        system.eval()
        result = evaluate_variant(system, tokenizer, test_records, device,
                                   nli_model=nli_model, nli_tokenizer=nli_tokenizer)
        all_results[variant] = result
        print(json.dumps(result, indent=2))

        del backbone, system
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAblation results written to {args.output_json}")


if __name__ == "__main__":
    main()
