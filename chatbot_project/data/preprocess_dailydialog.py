"""
DailyDialog pre-processing pipeline — Section 3.4.3.

- Same GPT-2 tokenisation as PersonaChat.
- Retains dialogue-act labels (inform, question, directive, commissive)
  and emotion labels (neutral, happiness, surprise, sadness, disgust,
  anger, fear) as auxiliary DST supervision.
- Segments conversations into 512-token windows with 50% sliding-window
  overlap to maximise data utilisation.

Usage:
    python -m data.preprocess_dailydialog --split train --out data/processed/dailydialog_train.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DATA_CFG
from data.preprocess_personachat import build_tokenizer

DIALOGUE_ACT_MAP = {1: "inform", 2: "question", 3: "directive", 4: "commissive"}
EMOTION_MAP = {
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
    4: "happiness", 5: "sadness", 6: "surprise",
}


def sliding_windows(turns, tokenizer, max_tokens: int, overlap: float):
    """
    Yields lists of consecutive turns ("windows") such that each window's
    token count <= max_tokens, advancing by (1 - overlap) fraction of the
    window each step (Section 3.4.3).
    """
    lengths = [len(tokenizer.encode(t)) for t in turns]
    n = len(turns)
    windows = []

    start = 0
    while start < n:
        end, running = start, 0
        while end < n and running + lengths[end] <= max_tokens:
            running += lengths[end]
            end += 1
        if end == start:
            end = start + 1  # guarantee progress even for a single long turn
        windows.append(turns[start:end])
        step = max(1, int((end - start) * (1 - overlap)))
        start += step

    return windows


def process_split(split: str, out_path: str, limit: int = None):
    from datasets import load_dataset

    tokenizer = build_tokenizer()
    ds = load_dataset(DATA_CFG.daily_dialog_name, split=split, trust_remote_code=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_written = 0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for i, example in enumerate(ds):
            if limit and i >= limit:
                break

            turns = example["dialog"]
            acts = example.get("act", [])
            emotions = example.get("emotion", [])

            if not (DATA_CFG.min_turns <= len(turns) <= DATA_CFG.max_turns * 3):
                # DailyDialog windows are later segmented, so allow a looser cap upstream
                pass

            windows = sliding_windows(
                turns, tokenizer,
                max_tokens=DATA_CFG.max_seq_length - 64,
                overlap=DATA_CFG.sliding_window_overlap,
            )

            for w_idx, window in enumerate(windows):
                if len(window) < 2:
                    continue
                history, response = window[:-1], window[-1]
                start = w_idx  # approximate index offset for act/emotion alignment
                act_labels = [DIALOGUE_ACT_MAP.get(a, "inform") for a in acts[start:start + len(window)]]
                emo_labels = [EMOTION_MAP.get(e, "neutral") for e in emotions[start:start + len(window)]]

                record = {
                    "history": history,
                    "response": response,
                    "dialogue_acts": act_labels,
                    "emotions": emo_labels,
                    "input_text": " ".join(
                        f"<speaker{1 + (idx % 2)}> {turn}" for idx, turn in enumerate(history)
                    ),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"[DailyDialog/{split}] written={n_written}")


def build_combined_corpus(personachat_path: str, dailydialog_path: str, out_path: str,
                           ratio: float = None):
    """
    Interleaves PersonaChat and DailyDialog examples at the configured
    ratio (Section 3.4.3, default 70:30) for backbone fine-tuning.
    """
    import random

    ratio = ratio if ratio is not None else DATA_CFG.persona_dailydialog_ratio

    def load(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    pc = load(personachat_path)
    dd = load(dailydialog_path)
    random.shuffle(pc)
    random.shuffle(dd)

    n_total = len(pc) + len(dd)
    n_pc_target = int(n_total * ratio)
    n_dd_target = n_total - n_pc_target

    pc_sample = (pc * (n_pc_target // max(len(pc), 1) + 1))[:n_pc_target]
    dd_sample = (dd * (n_dd_target // max(len(dd), 1) + 1))[:n_dd_target]

    combined = [{"source": "personachat", **r} for r in pc_sample] + \
               [{"source": "dailydialog", **r} for r in dd_sample]
    random.shuffle(combined)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f_out:
        for r in combined:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Combined corpus] personachat={len(pc_sample)} ({ratio:.0%}) "
          f"dailydialog={len(dd_sample)} ({1 - ratio:.0%}) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("preprocess")
    p1.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p1.add_argument("--out", default=None)
    p1.add_argument("--limit", type=int, default=None)

    p2 = sub.add_parser("combine")
    p2.add_argument("--personachat", required=True)
    p2.add_argument("--dailydialog", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--ratio", type=float, default=None)

    args = parser.parse_args()

    if args.cmd == "preprocess":
        out_path = args.out or f"{DATA_CFG.processed_dir}/dailydialog_{args.split}.jsonl"
        process_split(args.split, out_path, args.limit)
    else:
        build_combined_corpus(args.personachat, args.dailydialog, args.out, args.ratio)
