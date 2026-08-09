"""
Dialogue State Tracker (DST) — Section 3.5.4.

A lightweight BERT-base-uncased classifier that, at each turn, consumes
the concatenation of the last three utterances and predicts probability
distributions over:
    - formality state        (formal / informal)
    - topic interest          (10-way)
    - sentiment / emotional tone (negative / neutral / positive)
    - dialogue act            (inform / question / directive / commissive)

Tier-1 (task) state: current topic, identified intent, unresolved needs.
Tier-2 (user) state: formality preference (updated via EMA), topic
interests, emotional tone, explicit stated preferences.

The DST's outputs are used to refresh the UPE's annotation feature
vector at inference time (Section 3.5.4, final paragraph) — see
`DialogueState.update()` and `to_annotation_features()`.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DST_CFG
from data.dataset import FORMALITY_LABELS, SENTIMENT_LABELS, DIALOGUE_ACT_LABELS, TOPIC_LABELS


class DialogueStateTracker(nn.Module):
    """Multi-head classification on top of a shared BERT encoder."""

    def __init__(self, base_model: str = None):
        super().__init__()
        base_model = base_model or DST_CFG.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size

        self.formality_head = nn.Linear(hidden, DST_CFG.formality_classes)
        self.topic_head = nn.Linear(hidden, DST_CFG.topic_classes)
        self.sentiment_head = nn.Linear(hidden, DST_CFG.sentiment_classes)
        self.dialogue_act_head = nn.Linear(hidden, DST_CFG.dialogue_act_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # [CLS] representation
        return {
            "formality_logits": self.formality_head(pooled),
            "topic_logits": self.topic_head(pooled),
            "sentiment_logits": self.sentiment_head(pooled),
            "dialogue_act_logits": self.dialogue_act_head(pooled),
        }

    @torch.no_grad()
    def predict_from_turns(self, turns: List[str], device="cpu", context_window: int = None):
        """Runs inference on the last `context_window` utterances (default 3, Section 3.5.4)."""
        window = context_window or DST_CFG.context_window_turns
        context = " [SEP] ".join(turns[-window:])
        enc = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=256).to(device)
        self.to(device)
        logits = self.forward(enc["input_ids"], enc["attention_mask"])
        return {
            "formality": FORMALITY_LABELS[logits["formality_logits"].argmax(-1).item()],
            "topic": TOPIC_LABELS[logits["topic_logits"].argmax(-1).item()],
            "sentiment": SENTIMENT_LABELS[logits["sentiment_logits"].argmax(-1).item()],
            "dialogue_act": DIALOGUE_ACT_LABELS[logits["dialogue_act_logits"].argmax(-1).item()],
        }


@dataclass
class DialogueState:
    """
    Session-level running state (Section 3.5.4).

    Tier 1 (task-oriented): current_topic, current_intent, unresolved_needs
    Tier 2 (user-specific, novel contribution): formality_score (EMA),
        topic_interests (running tally), emotional_tone, explicit_preferences
    """
    current_topic: Optional[str] = None
    current_intent: Optional[str] = None
    unresolved_needs: List[str] = field(default_factory=list)

    formality_score: float = 0.5  # 0 = fully informal, 1 = fully formal, EMA-updated
    topic_interests: dict = field(default_factory=dict)  # topic -> running count
    emotional_tone: Optional[str] = None
    explicit_preferences: List[str] = field(default_factory=list)

    def update(self, dst_prediction: dict, alpha: float = None):
        """Exponential-moving-average update of user-specific state (Section 3.5.4)."""
        alpha = alpha if alpha is not None else DST_CFG.ema_alpha

        new_formality_obs = 1.0 if dst_prediction["formality"] == "formal" else 0.0
        self.formality_score = alpha * new_formality_obs + (1 - alpha) * self.formality_score

        topic = dst_prediction["topic"]
        self.topic_interests[topic] = self.topic_interests.get(topic, 0) + 1
        self.current_topic = topic
        self.current_intent = dst_prediction["dialogue_act"]
        self.emotional_tone = dst_prediction["sentiment"]

    def add_explicit_preference(self, statement: str):
        self.explicit_preferences.append(statement)

    def dominant_topic(self) -> str:
        if not self.topic_interests:
            return "family"
        return max(self.topic_interests, key=self.topic_interests.get)

    def to_annotation_indices(self):
        """
        Converts the running state into the four class indices consumed
        by `models.upe.normalise_annotation_features`, closing the
        DST <-> UPE feedback loop described at the end of Section 3.5.4.
        """
        formality_idx = 0 if self.formality_score >= 0.5 else 1  # 0=formal,1=informal
        topic_idx = TOPIC_LABELS.index(self.dominant_topic()) if self.dominant_topic() in TOPIC_LABELS else 0
        sentiment_idx = SENTIMENT_LABELS.index(self.emotional_tone) if self.emotional_tone in SENTIMENT_LABELS else 1
        act_idx = DIALOGUE_ACT_LABELS.index(self.current_intent) if self.current_intent in DIALOGUE_ACT_LABELS else 0
        return formality_idx, topic_idx, sentiment_idx, act_idx
