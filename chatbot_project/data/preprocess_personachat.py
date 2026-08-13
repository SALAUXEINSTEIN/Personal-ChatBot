"""
PersonaChat pre-processing pipeline — Section 3.4.2.

PersonaChat JSON structure used here:

{
    "personality": [...],
    "utterances": [
        {
            "history": [...],
            "candidates": [...]
        },
        ...
    ]
}

Steps:
1. GPT-2 tokenisation + special tokens
2. Persona block construction
3. Conversation history windowing
4. User preference annotation
5. Filtering and duplicate persona removal

Usage:
    python -m data.preprocess_personachat --split train --limit 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from transformers import GPT2Tokenizer

from configs.config import DATA_CFG
from utils.annotation import annotate_utterance


# ============================================================
# TOKENIZER
# ============================================================

def build_tokenizer() -> GPT2Tokenizer:
    """Build GPT-2 tokenizer with project-specific special tokens."""

    tok = GPT2Tokenizer.from_pretrained("gpt2")

    special_tokens_dict = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "additional_special_tokens": [
            "<speaker1>",
            "<speaker2>",
            "<persona>",
        ],
    }

    tok.add_special_tokens(special_tokens_dict)

    return tok


# ============================================================
# PERSONA UTILITIES
# ============================================================

def _persona_hash(persona_sentences) -> str:
    """Create a stable hash for a persona sentence set."""

    joined = "||".join(
        sorted(
            str(s).strip().lower()
            for s in persona_sentences
        )
    )

    return hashlib.md5(
        joined.encode("utf-8")
    ).hexdigest()


def build_persona_block(persona_sentences) -> str:
    """
    Construct the persona block.

    Example:

    <persona> I like football.
    <persona> I have two dogs.
    """

    return " <persona> ".join(
        str(s).strip()
        for s in persona_sentences
        if str(s).strip()
    )


# ============================================================
# HISTORY WINDOWING
# ============================================================

def window_history(
    turns,
    tokenizer,
    max_tokens: int
) -> list:
    """
    Keep the most recent dialogue turns that fit
    within the token budget.
    """

    kept = []
    running_len = 0

    for turn in reversed(turns):

        n_tokens = len(
            tokenizer.encode(
                str(turn),
                add_special_tokens=False
            )
        )

        if running_len + n_tokens > max_tokens:
            break

        kept.append(str(turn))
        running_len += n_tokens

    return list(reversed(kept))


# ============================================================
# PROCESS DATASET
# ============================================================

def process_split(
    split: str,
    out_path: str,
    limit: int | None = None
):

    tokenizer = build_tokenizer()

    # --------------------------------------------------------
    # Load our LOCAL JSON file
    # --------------------------------------------------------

    split_to_file = {
        "train": (
            "data/raw/personachat/"
            "personachat_truecased_full_train.json"
        ),
        "validation": (
            "data/raw/personachat/"
            "personachat_truecased_full_valid.json"
        ),
        "test": (
            "data/raw/personachat/"
            "personachat_truecased_full_test.json"
        ),
    }

    input_path = split_to_file[split]

    print("Loading local PersonaChat file:")
    print(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"\nPersonaChat file not found:\n{input_path}\n"
            f"\nPlease check that the file exists."
        )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:
        dataset = json.load(f)

    print(
        f"Loaded {len(dataset)} top-level records."
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(out_path),
        exist_ok=True
    )

    seen_persona_hashes = set()

    n_written = 0
    n_skipped_len = 0
    n_skipped_dup = 0
    n_skipped_invalid = 0

    # --------------------------------------------------------
    # Iterate through top-level PersonaChat records
    # --------------------------------------------------------

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f_out:

        for record_index, example in enumerate(dataset):

            if limit is not None and record_index >= limit:
                break

            # ------------------------------------------------
            # Persona
            # ------------------------------------------------

            persona = example.get(
                "personality",
                []
            )

            # ------------------------------------------------
            # PersonaChat utterances
            # ------------------------------------------------

            utterances = example.get(
                "utterances",
                []
            )

            if not persona or not utterances:
                n_skipped_invalid += 1
                continue

            # ------------------------------------------------
            # Remove duplicate persona sets
            # ------------------------------------------------

            p_hash = _persona_hash(persona)

            if p_hash in seen_persona_hashes:
                n_skipped_dup += 1
                continue

            seen_persona_hashes.add(p_hash)

            # ------------------------------------------------
            # Each utterance entry represents a dialogue point
            # ------------------------------------------------

            for dialogue in utterances:

                history = dialogue.get(
                    "history",
                    []
                )

                candidates = dialogue.get(
                    "candidates",
                    []
                )

                if not history or not candidates:
                    n_skipped_invalid += 1
                    continue

                # ------------------------------------------------
                # PersonaChat candidate convention:
                #
                # The final candidate is the correct response
                # in the original PersonaChat format.
                # ------------------------------------------------

                response = candidates[-1]

                if not response:
                    n_skipped_invalid += 1
                    continue

                # ------------------------------------------------
                # Turn count filtering
                # ------------------------------------------------

                if not (
                    DATA_CFG.min_turns
                    <= len(history)
                    <= DATA_CFG.max_turns
                ):
                    n_skipped_len += 1
                    continue

                # ------------------------------------------------
                # History window
                # ------------------------------------------------

                windowed_history = window_history(
                    history,
                    tokenizer,
                    DATA_CFG.max_seq_length - 64
                )

                if not windowed_history:
                    n_skipped_invalid += 1
                    continue

                # ------------------------------------------------
                # User preference annotation
                # ------------------------------------------------

                # Step 4: user-preference annotation
                #
                # DeBERTa zero-shot topic classification is computationally expensive
                # on CPU. During local preprocessing we therefore use lightweight
                # annotations. The DeBERTa topic annotation can be performed later
                # on GPU before final training.

                joined_history = " ".join(windowed_history)

                # Lightweight annotation for local preprocessing
                from utils.annotation import (
                    classify_formality,
                    classify_sentiment,
                    DEFAULT_TOPICS,
                )

                formality = classify_formality(joined_history)
                sentiment = classify_sentiment(joined_history)

                # Temporary uniform topic distribution.
                # This will be replaced by DeBERTa topic probabilities during
                # the GPU annotation stage.
                uniform_probability = 1.0 / len(DEFAULT_TOPICS)

                topic_distribution = {
                    topic: uniform_probability
                    for topic in DEFAULT_TOPICS
                }

                annotation = {
                    "text": joined_history,
                    "formality": formality,
                    "topic_distribution": topic_distribution,
                    "sentiment": sentiment,
                }

                # ------------------------------------------------
                # Input representation
                # ------------------------------------------------

                speaker_history = " ".join(
                    f"<speaker{1 + (idx % 2)}> {turn}"
                    for idx, turn
                    in enumerate(windowed_history)
                )

                persona_block = build_persona_block(
                    persona
                )

                input_text = (
                    f"<persona> "
                    f"{persona_block} "
                    f"<persona> "
                    f"{speaker_history}"
                )

                # ------------------------------------------------
                # Output record
                # ------------------------------------------------

                output_record = {
                    "persona_sentences": persona,
                    "persona_block": persona_block,
                    "history": windowed_history,
                    "response": response,

                    "formality": annotation[
                        "formality"
                    ],

                    "topic_distribution": annotation[
                        "topic_distribution"
                    ],

                    "sentiment": annotation[
                        "sentiment"
                    ],

                    "input_text": input_text,
                }

                f_out.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False
                    ) + "\n"
                )

                n_written += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"\n[PersonaChat/{split}] "
        f"written={n_written} "
        f"skipped_length={n_skipped_len} "
        f"skipped_duplicate_persona={n_skipped_dup} "
        f"skipped_invalid={n_skipped_invalid}"
    )

    print(
        f"Output: {out_path}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="train",
        choices=[
            "train",
            "validation",
            "test"
        ]
    )

    parser.add_argument(
        "--out",
        default=None
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for quick local testing."
    )

    args = parser.parse_args()

    out_path = (
        args.out
        or f"{DATA_CFG.processed_dir}/"
           f"personachat_{args.split}.jsonl"
    )

    process_split(
        args.split,
        out_path,
        args.limit
    )