"""
PersonaChat pre-processing pipeline — Section 3.4.2.

Steps implemented (numbered to match the dissertation):
    1. Tokenisation with GPT-2 BPE tokenizer + special tokens
    2. Persona sentence encoding ([PERSONA]-delimited block prepended to history)
    3. Conversation history windowing (max 512 tokens, most-recent-turns kept)
    4. User preference annotation (formality / topic / sentiment)
    5. Data filtering (turn count, duplicate persona sets)

Usage:
    python -m data.preprocess_personachat --split train --out data/processed/personachat_train.jsonl
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Tokenizer
from configs.config import DATA_CFG
from utils.annotation import annotate_utterance


def build_tokenizer() -> GPT2Tokenizer:
    """Step 1: GPT-2 tokenizer with the special tokens required by Section 3.4.2."""
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens_dict = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "additional_special_tokens": ["<speaker1>", "<speaker2>", "<persona>"],
    }
    tok.add_special_tokens(special_tokens_dict)
    return tok


def _persona_hash(persona_sentences) -> str:
    joined = "||".join(sorted(s.strip().lower() for s in persona_sentences))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def build_persona_block(persona_sentences) -> str:
    """Step 2: concatenate persona sentences delimited by <persona>, ConvAI2-style."""
    return " <persona> ".join(s.strip() for s in persona_sentences)


def window_history(turns, tokenizer, max_tokens: int) -> list:
    """
    Step 3: keep the most recent turns whose cumulative token length
    fits within `max_tokens` (matching GPT-2's max input length).
    """
    kept, running_len = [], 0
    for turn in reversed(turns):
        n_tokens = len(tokenizer.encode(turn))
        if running_len + n_tokens > max_tokens:
            break
        kept.append(turn)
        running_len += n_tokens
    return list(reversed(kept))


def process_split(split: str, out_path: str, limit: int = None):
    from datasets import load_dataset

    tokenizer = build_tokenizer()
    ds = load_dataset(DATA_CFG.persona_chat_name, split=split)

    seen_persona_hashes = set()
    n_written, n_skipped_len, n_skipped_dup = 0, 0, 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f_out:
        for i, example in enumerate(ds):
            if limit and i >= limit:
                break

            persona = example.get("personality") or example.get("persona") or []
            history = example.get("history") or example.get("utterances") or []
            candidates = example.get("candidates") or []
            response = example.get("response") or (candidates[-1] if candidates else None)

            if not persona or not history or not response:
                continue

            # Step 5: filter by turn count
            if not (DATA_CFG.min_turns <= len(history) <= DATA_CFG.max_turns):
                n_skipped_len += 1
                continue

            # Step 5: de-duplicate on persona sentence set
            p_hash = _persona_hash(persona)
            if p_hash in seen_persona_hashes:
                n_skipped_dup += 1
                continue
            seen_persona_hashes.add(p_hash)

            persona_block = build_persona_block(persona)
            windowed_history = window_history(history, tokenizer, DATA_CFG.max_seq_length - 64)

            # Step 4: user-preference annotation, computed over the full history
            joined_history = " ".join(windowed_history)
            annotation = annotate_utterance(joined_history)

            record = {
                "persona_sentences": persona,
                "persona_block": persona_block,
                "history": windowed_history,
                "response": response,
                "formality": annotation["formality"],
                "topic_distribution": annotation["topic_distribution"],
                "sentiment": annotation["sentiment"],
                "input_text": f"<persona> {persona_block} <persona> " + " ".join(
                    f"<speaker{1 + (idx % 2)}> {turn}" for idx, turn in enumerate(windowed_history)
                ),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[PersonaChat/{split}] written={n_written} "
          f"skipped_length={n_skipped_len} skipped_duplicate_persona={n_skipped_dup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None, help="optional cap for quick local testing")
    args = parser.parse_args()

    out_path = args.out or f"{DATA_CFG.processed_dir}/personachat_{args.split}.jsonl"
    process_split(args.split, out_path, args.limit)
