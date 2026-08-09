"""
Full Personalised Chatbot System — Section 3.5.1 (architectural overview).

Wires together the three core components:
    1. Transformer Dialogue Backbone (DialoGPT-medium)
    2. User Preference Encoder (soft-prompt prefix injected as past_key_values)
    3. Dialogue State Tracker (session-level user state, feeding back into the UPE)

Also supports the five ablation configurations from Section 3.6.4
(B, V1, V2, V3, FS) via `use_upe` / `use_dst` flags, so the same class
implements every variant needed for evaluation/ablation_study.py.
"""

from __future__ import annotations
import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import UPE_CFG, DST_CFG
from models.upe import UserPreferenceEncoder, normalise_annotation_features
from models.dst import DialogueStateTracker, DialogueState


class PersonalisedChatbotSystem(nn.Module):
    """
    Full System (FS) when use_upe=True and use_dst=True.
    Set either flag to False to reproduce ablation variants V1/V2/V3, or
    both False (and skip fine-tuning) to reproduce the Baseline (B).
    """

    def __init__(self, backbone, tokenizer, use_upe: bool = True, use_dst: bool = True):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.use_upe = use_upe
        self.use_dst = use_dst

        self.upe = UserPreferenceEncoder() if use_upe else None
        self.dst = DialogueStateTracker() if use_dst else None

    def _build_prefix(self, persona_sentences: List[List[str]], annotation_indices, device):
        """Builds past_key_values from the UPE for a batch (or None if UPE disabled)."""
        if not self.use_upe:
            return None
        formality_idx = torch.tensor([a[0] for a in annotation_indices], device=device)
        topic_idx = torch.tensor([a[1] for a in annotation_indices], device=device)
        sentiment_idx = torch.tensor([a[2] for a in annotation_indices], device=device)
        act_idx = torch.tensor([a[3] for a in annotation_indices], device=device)

        annotation_features = normalise_annotation_features(
            formality_idx, topic_idx, sentiment_idx, act_idx,
            n_topic=DST_CFG.topic_classes,
        )
        prefix = self.upe(persona_sentences, annotation_features)
        return self.upe.to_past_key_values(prefix)

    def forward(self, input_ids, attention_mask, labels=None,
                persona_sentences: Optional[List[List[str]]] = None,
                annotation_indices: Optional[list] = None,
                dst_labels: Optional[dict] = None):
        device = input_ids.device
        past_key_values = None
        if self.use_upe and persona_sentences is not None and annotation_indices is not None:
            past_key_values = self._build_prefix(persona_sentences, annotation_indices, device)

        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, past_key_values=past_key_values,
            use_cache=False if past_key_values is None else True,
        )
        loss_dict = {"lm_loss": outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)}
        total_loss = loss_dict["lm_loss"]

        if self.use_dst and dst_labels is not None:
            dst_out = self.dst(dst_labels["dst_input_ids"], dst_labels["dst_attention_mask"])
            dst_loss = (
                F.cross_entropy(dst_out["formality_logits"], dst_labels["formality_label"]) +
                F.cross_entropy(dst_out["topic_logits"], dst_labels["topic_label"]) +
                F.cross_entropy(dst_out["sentiment_logits"], dst_labels["sentiment_label"]) +
                F.cross_entropy(dst_out["dialogue_act_logits"], dst_labels["dialogue_act_label"])
            ) / 4.0
            loss_dict["dst_loss"] = dst_loss

        return {"logits": outputs.logits, "loss_dict": loss_dict, "total_loss": total_loss}

    @torch.no_grad()
    def generate_response(self, persona_sentences: List[str], history: List[str],
                           dialogue_state: Optional[DialogueState] = None,
                           max_new_tokens: int = 60, temperature: float = 0.8,
                           top_p: float = 0.9) -> str:
        """Inference-time generation used by the Gradio app (Section 3.7)."""
        device = next(self.backbone.parameters()).device
        persona_block = " <persona> ".join(persona_sentences) if persona_sentences else ""
        prompt = f"<persona> {persona_block} <persona> " + " ".join(
            f"<speaker{1 + (i % 2)}> {turn}" for i, turn in enumerate(history)
        )
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=452).to(device)

        past_key_values = None
        if self.use_upe:
            if dialogue_state is not None:
                annotation_idx = dialogue_state.to_annotation_indices()
            else:
                annotation_idx = (0, 0, 1, 0)  # formal, first topic, neutral, inform
            past_key_values = self._build_prefix([persona_sentences], [annotation_idx], device)

        gen_kwargs = dict(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if past_key_values is not None:
            gen_kwargs["past_key_values"] = past_key_values

        try:
            output_ids = self.backbone.generate(**gen_kwargs)
        except ValueError as exc:
            if "past_key_values" in str(exc).lower() or "cache" in str(exc).lower():
                gen_kwargs.pop("past_key_values", None)
                output_ids = self.backbone.generate(**gen_kwargs)
            else:
                raise

        response_ids = output_ids[0][enc["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response.strip()


def build_variant(variant: str, backbone, tokenizer) -> PersonalisedChatbotSystem:
    """
    Factory for the five ablation variants defined in Section 3.6.4:
        B  - baseline, zero-shot DialoGPT (no fine-tuning done upstream, no UPE/DST)
        V1 - fine-tuned backbone only
        V2 - fine-tuned backbone + DST only
        V3 - fine-tuned backbone + UPE only
        FS - fine-tuned backbone + UPE + DST (full system)
    """
    variant = variant.upper()
    mapping = {
        "B": dict(use_upe=False, use_dst=False),
        "V1": dict(use_upe=False, use_dst=False),
        "V2": dict(use_upe=False, use_dst=True),
        "V3": dict(use_upe=True, use_dst=False),
        "FS": dict(use_upe=True, use_dst=True),
    }
    if variant not in mapping:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(mapping)}.")
    return PersonalisedChatbotSystem(backbone, tokenizer, **mapping[variant])
