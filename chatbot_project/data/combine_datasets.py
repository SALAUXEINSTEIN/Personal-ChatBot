"""
Combine preprocessed PersonaChat and DailyDialog JSONL files
for Stage 1 backbone training.
"""

import json
import os
import random


SEED = 42


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def combine(split_name, persona_file, daily_file, output_file):

    print("=" * 70)
    print(f"Combining {split_name} datasets")
    print("=" * 70)

    persona = load_jsonl(persona_file)
    daily = load_jsonl(daily_file)

    print(f"PersonaChat: {len(persona):,}")
    print(f"DailyDialog: {len(daily):,}")

    combined = persona + daily

    random.seed(SEED)
    random.shuffle(combined)

    save_jsonl(combined, output_file)

    print(f"Combined:    {len(combined):,}")
    print(f"Written to:  {output_file}")
    print()


def main():

    combine(
        "train",
        "data/processed/personachat_train.jsonl",
        "data/processed/dailydialog_train.jsonl",
        "data/processed/combined_train.jsonl",
    )

    combine(
        "validation",
        "data/processed/personachat_validation.jsonl",
        "data/processed/dailydialog_validation.jsonl",
        "data/processed/combined_val.jsonl",
    )


if __name__ == "__main__":
    main()