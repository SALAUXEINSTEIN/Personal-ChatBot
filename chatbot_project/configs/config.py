"""
Central configuration for the Personalised Transformer Chatbot.

Every value here is traceable to a specific section of Chapter 3
(Research Methodology) of the dissertation:
  - Section 3.4  -> dataset / pre-processing settings
  - Section 3.5  -> architecture settings (backbone, UPE, DST)
  - Section 3.6  -> training hyperparameters (Tables 3.2 and 3.3)
  - Section 3.8  -> evaluation settings
"""

from dataclasses import dataclass, field
from typing import List

import torch


def _default_backbone_model() -> str:
    """Choose a CPU-friendly default when no GPU is available."""
    return "microsoft/DialoGPT-medium" if torch.cuda.is_available() else "microsoft/DialoGPT-small"


# --------------------------------------------------------------------------
# 3.4 Dataset / pre-processing configuration
# --------------------------------------------------------------------------
@dataclass
class DataConfig:
    persona_chat_name: str = "bavard/personachat_truecased"   # HF hub mirror of ConvAI2 PersonaChat
    daily_dialog_name: str = "daily_dialog"                    # HF hub DailyDialog

    max_seq_length: int = 512          # 3.4.2 (1) — GPT-2 max input length
    min_turns: int = 3                 # 3.4.2 (5) — filter short dialogues
    max_turns: int = 20                # 3.4.2 (5) — filter long dialogues
    sliding_window_overlap: float = 0.5  # 3.4.3 — DailyDialog 50% overlap windows

    persona_dailydialog_ratio: float = 0.70  # 3.4.3 — 70:30 mixing ratio (tunable, see 3.6.3)

    special_tokens: List[str] = field(default_factory=lambda: [
        "<bos>", "<eos>", "<speaker1>", "<speaker2>", "<persona>",
    ])

    cache_dir: str = "./data/cache"
    processed_dir: str = "./data/processed"


# --------------------------------------------------------------------------
# 3.5 Architecture configuration
# --------------------------------------------------------------------------
@dataclass
class BackboneConfig:
    # 3.5.2 — DialoGPT-medium is the dissertation default, but a smaller variant is
    # used automatically on CPU-only machines to keep local startup practical.
    model_name: str = field(default_factory=_default_backbone_model)
    hidden_size: int = 1024      # DialoGPT-medium hidden size (GPT-2 medium architecture)
    n_layer: int = 24
    n_head: int = 16


@dataclass
class UPEConfig:
    # 3.5.3 — User Preference Encoder (soft-prompt / prefix-tuning, Li & Liang 2021)
    sentence_bert_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # frozen S-BERT encoder
    sbert_dim: int = 384
    # one-hot concatenation: formality(2) + sentiment(3) + dialogue-act(4) + topic(10) = 19
    annotation_feature_dim: int = 19
    mlp_hidden_dim: int = 512
    prefix_length: int = 10              # number of virtual prefix tokens per attention layer
    freeze_sentence_bert: bool = True


@dataclass
class DSTConfig:
    # 3.5.4 — Dialogue State Tracker
    base_model: str = "bert-base-uncased"
    context_window_turns: int = 3        # last three utterances fed to classifier
    formality_classes: int = 2           # formal / informal
    topic_classes: int = 10              # zero-shot topic buckets (configurable)
    sentiment_classes: int = 3           # negative / neutral / positive (VADER buckets)
    dialogue_act_classes: int = 4        # inform, question, directive, commissive
    ema_alpha: float = 0.3               # exponential-moving-average update rate for formality state


# --------------------------------------------------------------------------
# 3.6 Training configuration
# --------------------------------------------------------------------------
@dataclass
class Stage1TrainingConfig:
    # Table 3.2
    learning_rate: float = 5e-5
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8   # effective batch size = 32
    num_epochs: int = 3
    early_stopping_patience: int = 2
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.10
    max_seq_length: int = 512
    weight_decay: float = 0.01
    fp16: bool = True
    gradient_checkpointing: bool = True
    output_dir: str = "./checkpoints/stage1_backbone"


@dataclass
class Stage2TrainingConfig:
    # 3.6.2 — joint UPE + DST training with weighted composite loss
    learning_rate: float = 5e-5
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    freeze_backbone_first_epoch: bool = True   # warm-up phase for MLP init
    lm_loss_weight: float = 1.0
    persona_consistency_weight: float = 0.3
    dst_loss_weight: float = 0.2
    fp16: bool = True
    gradient_checkpointing: bool = True
    output_dir: str = "./checkpoints/stage2_full_system"


@dataclass
class HyperparamSearchConfig:
    # 3.6.3 — targeted search grid
    learning_rates: List[float] = field(default_factory=lambda: [1e-5, 5e-5, 1e-4])
    persona_loss_weights: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    mixture_ratios: List[float] = field(default_factory=lambda: [0.70, 0.80])
    use_wandb: bool = True
    wandb_project: str = "personalised-chatbot-dissertation"


# --------------------------------------------------------------------------
# 3.8 Evaluation configuration
# --------------------------------------------------------------------------
@dataclass
class EvalConfig:
    nli_model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"  # persona-consistency NLI
    bootstrap_iterations: int = 10_000
    confidence_level: float = 0.95
    ablation_variants: List[str] = field(default_factory=lambda: ["B", "V1", "V2", "V3", "FS"])


DATA_CFG = DataConfig()
BACKBONE_CFG = BackboneConfig()
UPE_CFG = UPEConfig()
DST_CFG = DSTConfig()
STAGE1_CFG = Stage1TrainingConfig()
STAGE2_CFG = Stage2TrainingConfig()
SEARCH_CFG = HyperparamSearchConfig()
EVAL_CFG = EvalConfig()
