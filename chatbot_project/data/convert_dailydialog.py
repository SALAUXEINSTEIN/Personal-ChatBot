import json
import os
import pandas as pd


RAW_DIR = "data/raw/dailydialog"
OUTPUT_DIR = "data/raw/dailydialog/json"


SPLITS = {
    "train": "dailydialog-train.parquet",
    "validation": "dailydialog-validation.parquet",
    "test": "dailydialog-test.parquet",
}


def convert_split(split, filename):
    input_path = os.path.join(RAW_DIR, filename)
    output_path = os.path.join(
        OUTPUT_DIR,
        f"dailydialog_{split}.json"
    )

    print(f"\nLoading: {input_path}")

    df = pd.read_parquet(input_path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    records = []

    for _, row in df.iterrows():

        # Convert NumPy arrays to normal Python lists
        utterances = [str(x) for x in row["utterances"]]
        acts = [int(x) for x in row["acts"]]
        emotions = [int(x) for x in row["emotions"]]

        # Basic validation
        if len(utterances) == 0:
            continue

        if len(utterances) != len(acts):
            print(
                f"WARNING: utterance/act mismatch "
                f"for {row['id']}"
            )
            continue

        if len(utterances) != len(emotions):
            print(
                f"WARNING: utterance/emotion mismatch "
                f"for {row['id']}"
            )
            continue

        record = {
            "id": str(row["id"]),
            "dialog": utterances,
            "act": acts,
            "emotion": emotions
        }

        records.append(record)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Written: {output_path}")
    print(f"Records: {len(records)}")


def main():

    print("=" * 60)
    print("DailyDialog Parquet → JSON conversion")
    print("=" * 60)

    for split, filename in SPLITS.items():
        convert_split(split, filename)

    print("\nConversion completed successfully.")


if __name__ == "__main__":
    main()