"""
DailyDialog pre-processing pipeline — Section 3.4.3.

Reads the locally converted DailyDialog JSON files instead of relying
on the deprecated Hugging Face dataset script.

Expected raw files:

data/raw/dailydialog/json/
    dailydialog_train.json
    dailydialog_validation.json
    dailydialog_test.json

Outputs:

data/processed/
    dailydialog_train.jsonl
    dailydialog_validation.jsonl
    dailydialog_test.jsonl

The processed examples contain:
    history
    response
    dialogue_acts
    emotions
    input_text
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from configs.config import DATA_CFG
from data.preprocess_personachat import build_tokenizer


DIALOGUE_ACT_MAP = {
    0: "dummy",
    1: "inform",
    2: "question",
    3: "directive",
    4: "commissive",
}

EMOTION_MAP = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}


RAW_JSON_DIR = os.path.join(
    "data",
    "raw",
    "dailydialog",
    "json"
)


def load_local_split(split: str):
    """
    Load a locally converted DailyDialog JSON split.
    """

    filename = f"dailydialog_{split}.json"

    path = os.path.join(
        RAW_JSON_DIR,
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nDailyDialog file not found:\n"
            f"{path}\n\n"
            f"Please make sure you have run:\n"
            f"python data\\convert_dailydialog.py"
        )

    print(f"Loading local DailyDialog file:")
    print(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} dialogues.")

    return data


def sliding_windows(
    turns,
    tokenizer,
    max_tokens: int,
    overlap: float
):
    """
    Generate overlapping windows of dialogue turns.

    Each window is constrained to max_tokens.
    """

    lengths = [
        len(tokenizer.encode(t))
        for t in turns
    ]

    n = len(turns)

    windows = []

    start = 0

    while start < n:

        end = start
        running = 0

        while (
            end < n
            and running + lengths[end] <= max_tokens
        ):
            running += lengths[end]
            end += 1

        # Guarantee progress for an unusually long turn
        if end == start:
            end = start + 1

        windows.append(
            (
                start,
                end,
                turns[start:end]
            )
        )

        step = max(
            1,
            int(
                (end - start)
                * (1 - overlap)
            )
        )

        start += step

    return windows


def process_split(
    split: str,
    out_path: str,
    limit: int | None = None
):

    tokenizer = build_tokenizer()

    dataset = load_local_split(split)

    os.makedirs(
        os.path.dirname(out_path),
        exist_ok=True
    )

    n_written = 0

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f_out:

        for i, example in enumerate(dataset):

            if limit is not None and i >= limit:
                break

            turns = example["dialog"]

            acts = example.get(
                "act",
                []
            )

            emotions = example.get(
                "emotion",
                []
            )

            if len(turns) < 2:
                continue

            windows = sliding_windows(
                turns,
                tokenizer,
                max_tokens=DATA_CFG.max_seq_length - 64,
                overlap=DATA_CFG.sliding_window_overlap,
            )

            for start, end, window in windows:

                # Need at least history + response
                if len(window) < 2:
                    continue

                history = window[:-1]

                response = window[-1]

                window_acts = acts[
                    start:end
                ]

                window_emotions = emotions[
                    start:end
                ]

                act_labels = [
                    DIALOGUE_ACT_MAP.get(
                        int(a),
                        "dummy"
                    )
                    for a in window_acts
                ]

                emotion_labels = [
                    EMOTION_MAP.get(
                        int(e),
                        "neutral"
                    )
                    for e in window_emotions
                ]

                input_text = " ".join(
                    f"<speaker{1 + (idx % 2)}> {turn}"
                    for idx, turn in enumerate(history)
                )

                record = {
                    "source": "dailydialog",

                    "history": history,

                    "response": response,

                    "dialogue_acts": act_labels,

                    "emotions": emotion_labels,

                    "input_text": input_text,
                }

                f_out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                n_written += 1

    print(
        f"[DailyDialog/{split}] "
        f"written={n_written}"
    )


def build_combined_corpus(
    personachat_path: str,
    dailydialog_path: str,
    out_path: str,
    ratio: float | None = None
):
    """
    Combine PersonaChat and DailyDialog.

    Default ratio:
        PersonaChat = 70%
        DailyDialog = 30%
    """

    ratio = (
        ratio
        if ratio is not None
        else DATA_CFG.persona_dailydialog_ratio
    )

    def load_jsonl(path):

        with open(
            path,
            encoding="utf-8"
        ) as f:

            return [
                json.loads(line)
                for line in f
                if line.strip()
            ]

    pc = load_jsonl(personachat_path)

    dd = load_jsonl(dailydialog_path)

    print(
        f"Loaded PersonaChat: {len(pc)}"
    )

    print(
        f"Loaded DailyDialog: {len(dd)}"
    )

    random.shuffle(pc)
    random.shuffle(dd)

    n_total = len(pc) + len(dd)

    n_pc_target = int(
        n_total * ratio
    )

    n_dd_target = (
        n_total - n_pc_target
    )

    # Sample without replacement where possible.
    pc_sample = (
        pc[:n_pc_target]
        if len(pc) >= n_pc_target
        else pc
    )

    dd_sample = (
        dd[:n_dd_target]
        if len(dd) >= n_dd_target
        else dd
    )

    # If one dataset is smaller than its requested share,
    # repeat examples to maintain the requested ratio.
    if len(pc_sample) < n_pc_target and pc:

        repetitions = (
            n_pc_target // len(pc)
        ) + 1

        pc_sample = (
            pc * repetitions
        )[:n_pc_target]

    if len(dd_sample) < n_dd_target and dd:

        repetitions = (
            n_dd_target // len(dd)
        ) + 1

        dd_sample = (
            dd * repetitions
        )[:n_dd_target]

    combined = (
        [
            {
                "source": "personachat",
                **record
            }
            for record in pc_sample
        ]
        +
        [
            {
                "source": "dailydialog",
                **record
            }
            for record in dd_sample
        ]
    )

    random.shuffle(combined)

    os.makedirs(
        os.path.dirname(out_path),
        exist_ok=True
    )

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f_out:

        for record in combined:

            f_out.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"[Combined corpus] "
        f"personachat={len(pc_sample)} "
        f"({ratio:.0%}) "
        f"dailydialog={len(dd_sample)} "
        f"({1 - ratio:.0%})"
    )

    print(
        f"Output: {out_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="cmd",
        required=True
    )

    # -------------------------
    # PREPROCESS
    # -------------------------

    p1 = sub.add_parser(
        "preprocess"
    )

    p1.add_argument(
        "--split",
        default="train",
        choices=[
            "train",
            "validation",
            "test"
        ]
    )

    p1.add_argument(
        "--out",
        default=None
    )

    p1.add_argument(
        "--limit",
        type=int,
        default=None
    )

    # -------------------------
    # COMBINE
    # -------------------------

    p2 = sub.add_parser(
        "combine"
    )

    p2.add_argument(
        "--personachat",
        required=True
    )

    p2.add_argument(
        "--dailydialog",
        required=True
    )

    p2.add_argument(
        "--out",
        required=True
    )

    p2.add_argument(
        "--ratio",
        type=float,
        default=None
    )

    args = parser.parse_args()

    if args.cmd == "preprocess":

        out_path = (
            args.out
            or
            f"{DATA_CFG.processed_dir}/"
            f"dailydialog_{args.split}.jsonl"
        )

        process_split(
            args.split,
            out_path,
            args.limit
        )

    elif args.cmd == "combine":

        build_combined_corpus(
            args.personachat,
            args.dailydialog,
            args.out,
            args.ratio
        )


if __name__ == "__main__":
    main()