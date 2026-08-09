"""
Transformer Dialogue Backbone — Section 3.5.2.

Thin wrapper around Hugging Face's AutoModelForCausalLM / AutoTokenizer
loading DialoGPT-medium, with the special tokens added in
data/preprocess_personachat.py registered on the model's embedding table.
"""

from __future__ import annotations
import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import BACKBONE_CFG
from data.preprocess_personachat import build_tokenizer


def load_backbone_and_tokenizer(model_name: str = None, device: str = None):
    """
    Returns (model, tokenizer). Resizes the model's token embeddings to
    account for the added special tokens (<bos>, <eos>, <speaker1>,
    <speaker2>, <persona>, <pad>).
    """
    model_name = model_name or BACKBONE_CFG.model_name
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return model, tokenizer


def load_finetuned_backbone(checkpoint_dir: str, device: str = None):
    """Loads a backbone checkpoint saved after Stage 1 fine-tuning."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    model.to(device)
    return model, tokenizer
