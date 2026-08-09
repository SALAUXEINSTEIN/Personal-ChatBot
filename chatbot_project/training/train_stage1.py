"""
Stage 1 training — Section 3.6.2, hyperparameters per Table 3.2.

Fine-tunes DialoGPT-medium as a causal LM on the combined
PersonaChat + DailyDialog corpus, before the UPE/DST are introduced.
This produces the "V1" ablation checkpoint used throughout Chapter 5.

Usage:
    python -m training.train_stage1 \
        --train_file data/processed/combined_train.jsonl \
        --val_file data/processed/combined_val.jsonl
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling,
)

from configs.config import STAGE1_CFG, DATA_CFG
from models.backbone import load_backbone_and_tokenizer
from data.dataset import DialogueCLMDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", default=STAGE1_CFG.output_dir)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_backbone_and_tokenizer()

    train_ds = DialogueCLMDataset(args.train_file, tokenizer, DATA_CFG.max_seq_length)
    val_ds = DialogueCLMDataset(args.val_file, tokenizer, DATA_CFG.max_seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=STAGE1_CFG.learning_rate,
        per_device_train_batch_size=STAGE1_CFG.per_device_batch_size,
        per_device_eval_batch_size=STAGE1_CFG.per_device_batch_size,
        gradient_accumulation_steps=STAGE1_CFG.gradient_accumulation_steps,
        num_train_epochs=STAGE1_CFG.num_epochs,
        lr_scheduler_type=STAGE1_CFG.lr_scheduler_type,
        warmup_ratio=STAGE1_CFG.warmup_ratio,
        weight_decay=STAGE1_CFG.weight_decay,
        fp16=STAGE1_CFG.fp16,
        gradient_checkpointing=STAGE1_CFG.gradient_checkpointing,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=["wandb"] if args.use_wandb else [],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=STAGE1_CFG.early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Stage 1 fine-tuned backbone saved to {args.output_dir}")


if __name__ == "__main__":
    main()
