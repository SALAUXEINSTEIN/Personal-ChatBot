import json
from pathlib import Path

FILES = [
    "data/processed/personachat_train.jsonl",
    "data/processed/personachat_validation.jsonl",
    "data/processed/dailydialog_train.jsonl",
    "data/processed/dailydialog_validation.jsonl",
    "data/processed/dailydialog_test.jsonl",
]

for file_path in FILES:
    path = Path(file_path)

    print("\n" + "=" * 70)
    print(f"FILE: {file_path}")

    if not path.exists():
        print("❌ FILE NOT FOUND")
        continue

    count = 0

    with open(path, "r", encoding="utf-8") as f:
        first_record = None

        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if first_record is None:
                first_record = record

            count += 1

    print(f"Records: {count}")
    print(f"Fields: {list(first_record.keys())}")

    print("\nFirst record:")
    print(json.dumps(first_record, indent=2, ensure_ascii=False)[:2000])

print("\n" + "=" * 70)
print("Verification complete.")