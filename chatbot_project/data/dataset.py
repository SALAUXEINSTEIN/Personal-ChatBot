"""
PyTorch Dataset / collator for Stage 1 (backbone-only) and Stage 2
(backbone + UPE + DST) training, consuming the JSONL files produced by
preprocess_personachat.py and preprocess_dailydialog.py.
"""

from __future__ import annotations
import json
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATA_CFG, DST_CFG


FORMALITY_LABELS = ["formal", "informal"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
DIALOGUE_ACT_LABELS = ["inform", "question", "directive", "commissive"]
TOPIC_LABELS = ["family", "work", "hobbies", "food", "travel",
                "health", "education", "entertainment", "sports", "technology"]


def _index(label_list, value, default=0):
    try:
        return label_list.index(value)
    except (ValueError, TypeError):
        return default


class DialogueCLMDataset(Dataset):
    """Stage 1: causal-LM dataset over the combined PersonaChat+DailyDialog corpus."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length or DATA_CFG.max_seq_length
        with open(jsonl_path, encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx) -> Dict:
        r = self.records[idx]
        prompt = r["input_text"]
        target = r["response"]
        full_text = f"{prompt} <speaker2> {target} <eos>"

        enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Labels: mask out the prompt portion so loss is only on the response
        prompt_len = len(self.tokenizer(prompt, truncation=True,
                                         max_length=self.max_length)["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class PersonalisedDialogueDataset(Dataset):
    """
    Stage 2: adds persona sentences (for the UPE) and DST supervision
    labels (formality / topic / sentiment / dialogue-act) on top of the
    Stage-1 fields.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length or DATA_CFG.max_seq_length
        with open(jsonl_path, encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx) -> Dict:
        r = self.records[idx]
        prompt = r["input_text"]
        target = r["response"]
        full_text = f"{prompt} <speaker2> {target} <eos>"

        enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        prompt_len = len(self.tokenizer(prompt, truncation=True,
                                         max_length=self.max_length)["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        persona_sentences = r.get("persona_sentences", [])
        if not persona_sentences:
            persona_sentences = ["I am a conversational assistant."]

        topic_dist = r.get("topic_distribution", {})
        if isinstance(topic_dist, dict) and topic_dist:
            top_topic = max(topic_dist, key=topic_dist.get)
        else:
            top_topic = "family"

        formality_label = _index(FORMALITY_LABELS, r.get("formality", "formal"))
        sentiment_label = _index(SENTIMENT_LABELS, r.get("sentiment", {}).get("label", "neutral")
                                  if isinstance(r.get("sentiment"), dict) else "neutral")
        topic_label = _index(TOPIC_LABELS, top_topic)
        act_label = _index(DIALOGUE_ACT_LABELS,
                            (r.get("dialogue_acts") or ["inform"])[-1])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "persona_sentences": persona_sentences,
            "formality_label": torch.tensor(formality_label, dtype=torch.long),
            "topic_label": torch.tensor(topic_label, dtype=torch.long),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
            "dialogue_act_label": torch.tensor(act_label, dtype=torch.long),
        }


def personalised_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collator needed because persona_sentences is a variable-length list of strings."""
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "formality_label": torch.stack([b["formality_label"] for b in batch]),
        "topic_label": torch.stack([b["topic_label"] for b in batch]),
        "sentiment_label": torch.stack([b["sentiment_label"] for b in batch]),
        "dialogue_act_label": torch.stack([b["dialogue_act_label"] for b in batch]),
        "persona_sentences": [b["persona_sentences"] for b in batch],  # list-of-lists, kept as-is
    }
    return out
