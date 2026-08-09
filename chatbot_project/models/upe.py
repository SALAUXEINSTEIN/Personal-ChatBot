"""
User Preference Encoder (UPE) — Section 3.5.3.

Design (Li & Liang, 2021 prefix-tuning, adapted per the dissertation):
  1. Persona feature vector: mean-pooled Sentence-BERT embeddings of the
     user's persona sentences (frozen encoder).
  2. Annotation feature vector: formality / topic-distribution /
     sentiment / dialogue-act-distribution, normalised to [0, 1].
  3. Concatenation -> 2-layer MLP (ReLU) -> per-layer key/value prefix
     vectors, prepended to the transformer backbone's attention at every
     layer. The MLP is the *only* module trained from scratch.
"""

from __future__ import annotations
import os
import sys
from typing import List

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import UPE_CFG, BACKBONE_CFG


class UserPreferenceEncoder(nn.Module):
    def __init__(self,
                 sbert_dim: int = None,
                 annotation_dim: int = None,
                 mlp_hidden_dim: int = None,
                 prefix_length: int = None,
                 hidden_size: int = None,
                 n_layer: int = None,
                 n_head: int = None,
                 freeze_sentence_bert: bool = None):
        super().__init__()
        self.sbert_dim = sbert_dim or UPE_CFG.sbert_dim
        self.annotation_dim = annotation_dim or UPE_CFG.annotation_feature_dim
        self.mlp_hidden_dim = mlp_hidden_dim or UPE_CFG.mlp_hidden_dim
        self.prefix_length = prefix_length or UPE_CFG.prefix_length
        self.hidden_size = hidden_size or BACKBONE_CFG.hidden_size
        self.n_layer = n_layer or BACKBONE_CFG.n_layer
        self.n_head = n_head or BACKBONE_CFG.n_head
        self.head_dim = self.hidden_size // self.n_head

        # Frozen Sentence-BERT encoder for persona sentences
        from sentence_transformers import SentenceTransformer
        self.sentence_encoder = SentenceTransformer(UPE_CFG.sentence_bert_model)
        if freeze_sentence_bert if freeze_sentence_bert is not None else UPE_CFG.freeze_sentence_bert:
            for p in self.sentence_encoder.parameters():
                p.requires_grad = False

        # The only trainable module in the UPE (Section 3.5.3, para 3)
        input_dim = self.sbert_dim + self.annotation_dim
        output_dim = self.n_layer * 2 * self.n_head * self.head_dim * self.prefix_length
        # 2x -> key AND value prefixes, per layer, per head

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, output_dim),
        )

    @torch.no_grad()
    def encode_personas(self, persona_sentence_lists: List[List[str]], device) -> torch.Tensor:
        """Mean-pool Sentence-BERT embeddings across each user's persona sentences."""
        pooled = []
        for sentences in persona_sentence_lists:
            if not sentences:
                sentences = ["No stated preferences."]
            embs = self.sentence_encoder.encode(sentences, convert_to_tensor=True,
                                                 show_progress_bar=False)
            pooled.append(embs.mean(dim=0))
        return torch.stack(pooled).to(device)

    def forward(self, persona_sentence_lists: List[List[str]],
                annotation_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            persona_sentence_lists: batch of lists of persona sentences.
            annotation_features: [batch, annotation_dim] tensor already
                normalised to [0, 1] (formality, topic dist, sentiment,
                dialogue-act dist).
        Returns:
            past_key_values-style tensor of shape
            [n_layer, 2, batch, n_head, prefix_length, head_dim]
            ready to prepend to the backbone's attention keys/values.
        """
        device = annotation_features.device
        persona_vec = self.encode_personas(persona_sentence_lists, device)  # [B, sbert_dim]
        combined = torch.cat([persona_vec, annotation_features], dim=-1)
        raw_prefix = self.mlp(combined)  # [B, n_layer*2*n_head*prefix_len*head_dim]

        batch_size = raw_prefix.shape[0]
        prefix = raw_prefix.view(
            batch_size, self.n_layer, 2, self.n_head, self.prefix_length, self.head_dim
        )
        # reorder to [n_layer, 2, batch, n_head, prefix_length, head_dim] for GPT2 past_key_values
        prefix = prefix.permute(1, 2, 0, 3, 4, 5)
        return prefix

    def to_past_key_values(self, prefix: torch.Tensor):
        """Splits the prefix tensor into the tuple-of-tuples format HF GPT-2 expects."""
        return tuple(
            (prefix[layer, 0], prefix[layer, 1]) for layer in range(prefix.shape[0])
        )


def normalise_annotation_features(formality_idx, topic_idx, sentiment_idx, act_idx,
                                   n_formality=2, n_topic=10, n_sentiment=3,
                                   n_act=4) -> torch.Tensor:
    """
    Builds the fixed-dimensional annotation feature vector described in
    Section 3.5.3 (formality level, topic category, sentiment polarity,
    dialogue-act), one-hot encoded and concatenated. The resulting
    dimensionality (n_formality + n_topic + n_sentiment + n_act) must
    match UPE_CFG.annotation_feature_dim — see configs/config.py.
    """
    import torch.nn.functional as F
    formality_oh = F.one_hot(formality_idx, num_classes=n_formality).float()
    sentiment_oh = F.one_hot(sentiment_idx, num_classes=n_sentiment).float()
    act_oh = F.one_hot(act_idx, num_classes=n_act).float()
    topic_oh = F.one_hot(topic_idx, num_classes=n_topic).float()
    return torch.cat([formality_oh, sentiment_oh, act_oh, topic_oh], dim=-1)
