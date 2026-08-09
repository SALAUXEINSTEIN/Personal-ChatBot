"""
Exploratory Data Analysis — Section 3.4.4.

Produces:
    1. Utterance length distributions (tokens per turn)
    2. Vocabulary analysis (type-token ratio, most frequent tokens)
    3. Turn-count distribution across dialogues
    4. Persona sentence content analysis (topic distribution, length)
    5. Cross-dataset comparison of linguistic register / formality

Usage:
    python -m data.eda --personachat data/processed/personachat_train.jsonl \
                        --dailydialog data/processed/dailydialog_train.jsonl \
                        --out_dir ./eda_report
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.annotation import classify_formality


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def utterance_length_stats(records, text_key="response"):
    lengths = [len(r[text_key].split()) for r in records if r.get(text_key)]
    if not lengths:
        return {}
    lengths.sort()
    n = len(lengths)
    return {
        "count": n,
        "mean": sum(lengths) / n,
        "median": lengths[n // 2],
        "min": lengths[0],
        "max": lengths[-1],
    }


def vocabulary_stats(records, text_key="response", top_k=20):
    all_tokens = []
    for r in records:
        text = r.get(text_key, "")
        all_tokens.extend(text.lower().split())
    counter = Counter(all_tokens)
    ttr = len(counter) / max(len(all_tokens), 1)
    return {"type_token_ratio": ttr, "vocab_size": len(counter),
            "total_tokens": len(all_tokens), "top_tokens": counter.most_common(top_k)}


def turn_count_stats(records, history_key="history"):
    counts = [len(r.get(history_key, [])) for r in records]
    if not counts:
        return {}
    counts.sort()
    n = len(counts)
    return {"mean": sum(counts) / n, "median": counts[n // 2], "min": counts[0], "max": counts[-1]}


def persona_stats(personachat_records):
    lengths, all_words = [], []
    for r in personachat_records:
        sentences = r.get("persona_sentences", [])
        lengths.append(len(sentences))
        for s in sentences:
            all_words.extend(s.lower().split())
    counter = Counter(all_words)
    return {
        "avg_num_persona_sentences": sum(lengths) / max(len(lengths), 1),
        "top_persona_words": counter.most_common(20),
    }


def formality_comparison(personachat_records, dailydialog_records):
    def ratio(records, key):
        labels = [classify_formality(r.get(key, "")) for r in records if r.get(key)]
        informal = sum(1 for l in labels if l == "informal")
        return informal / max(len(labels), 1)

    return {
        "personachat_informal_ratio": ratio(personachat_records, "response"),
        "dailydialog_informal_ratio": ratio(dailydialog_records, "response"),
    }


def run_eda(personachat_path: str, dailydialog_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    pc = load_jsonl(personachat_path)
    dd = load_jsonl(dailydialog_path)

    report = {
        "personachat": {
            "utterance_length": utterance_length_stats(pc),
            "vocabulary": vocabulary_stats(pc),
            "turn_counts": turn_count_stats(pc),
            "persona_analysis": persona_stats(pc),
        },
        "dailydialog": {
            "utterance_length": utterance_length_stats(dd),
            "vocabulary": vocabulary_stats(dd),
            "turn_counts": turn_count_stats(dd),
        },
        "cross_dataset_formality": formality_comparison(pc, dd),
    }

    out_path = os.path.join(out_dir, "eda_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"EDA report written to {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--personachat", required=True)
    parser.add_argument("--dailydialog", required=True)
    parser.add_argument("--out_dir", default="./eda_report")
    args = parser.parse_args()
    run_eda(args.personachat, args.dailydialog, args.out_dir)
