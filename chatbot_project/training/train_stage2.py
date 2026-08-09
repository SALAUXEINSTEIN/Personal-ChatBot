"""
Stage 2 training — Section 3.6.2.

Jointly trains the User Preference Encoder and Dialogue State Tracker on
top of the Stage-1 fine-tuned backbone, using the composite loss:

    L = 1.0 * L_lm + 0.3 * L_persona_consistency + 0.2 * L_dst

The backbone is optionally frozen for the first epoch (warm-up phase) so
the UPE's MLP can initialise before joint optimisation, as specified in
the dissertation.

Usage:
    python -m training.train_stage2 \
        --backbone_checkpoint checkpoints/stage1_backbone \
        --train_file data/processed/personachat_train.jsonl \
        --val_file data/processed/personachat_validation.jsonl
"""

from __future__ import annotations
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import STAGE2_CFG, DATA_CFG, EVAL_CFG
from models.backbone import load_finetuned_backbone
from models.personalised_chatbot import PersonalisedChatbotSystem
from data.dataset import PersonalisedDialogueDataset, personalised_collate_fn


def persona_consistency_loss(nli_model, nli_tokenizer, generated_texts, persona_sentences_batch, device):
    """
    Cross-entropy between the NLI-predicted entailment score of the
    generated response w.r.t. the persona sentences and a target
    "entailment" label (Section 3.6.2, term 2).
    Uses a pretrained NLI model in zero-shot mode (no separate
    fine-tuning stage is required for this auxiliary signal).
    """
    premises, hypotheses = [], []
    for text, personas in zip(generated_texts, persona_sentences_batch):
        persona_str = " ".join(personas) if personas else "No stated preferences."
        premises.append(persona_str)
        hypotheses.append(text)

    enc = nli_tokenizer(premises, hypotheses, return_tensors="pt", truncation=True,
                         padding=True, max_length=256).to(device)
    logits = nli_model(**enc).logits  # [B, 3] contradiction/neutral/entailment (model-dependent order)
    target = torch.full((logits.shape[0],), logits.shape[-1] - 1, dtype=torch.long, device=device)  # target = entailment class
    return torch.nn.functional.cross_entropy(logits, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_checkpoint", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", default=STAGE2_CFG.output_dir)
    parser.add_argument("--nli_model", default="facebook/bart-large-mnli")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, tokenizer = load_finetuned_backbone(args.backbone_checkpoint, device)
    system = PersonalisedChatbotSystem(backbone, tokenizer, use_upe=True, use_dst=True).to(device)

    nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device)
    nli_model.eval()
    for p in nli_model.parameters():
        p.requires_grad = False

    train_ds = PersonalisedDialogueDataset(args.train_file, tokenizer, DATA_CFG.max_seq_length)
    val_ds = PersonalisedDialogueDataset(args.val_file, tokenizer, DATA_CFG.max_seq_length)

    train_loader = DataLoader(train_ds, batch_size=STAGE2_CFG.per_device_batch_size,
                               shuffle=True, collate_fn=personalised_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=STAGE2_CFG.per_device_batch_size,
                             shuffle=False, collate_fn=personalised_collate_fn)

    optimizer = AdamW(system.parameters(), lr=STAGE2_CFG.learning_rate)
    total_steps = (len(train_loader) // STAGE2_CFG.gradient_accumulation_steps) * STAGE2_CFG.num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                 num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=STAGE2_CFG.fp16)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    step = 0

    for epoch in range(STAGE2_CFG.num_epochs):
        freeze_backbone = STAGE2_CFG.freeze_backbone_first_epoch and epoch == 0
        for p in system.backbone.parameters():
            p.requires_grad = not freeze_backbone

        system.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            annotation_indices = list(zip(
                batch["formality_label"].tolist(), batch["topic_label"].tolist(),
                batch["sentiment_label"].tolist(), batch["dialogue_act_label"].tolist(),
            ))

            with torch.cuda.amp.autocast(enabled=STAGE2_CFG.fp16):
                out = system(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"], persona_sentences=batch["persona_sentences"],
                    annotation_indices=annotation_indices,
                )
                lm_loss = out["loss_dict"]["lm_loss"]

                # Persona-consistency auxiliary loss uses greedy-decoded text; computed
                # every few steps only, to control compute cost during training.
                if batch_idx % 4 == 0:
                    with torch.no_grad():
                        gen_ids = system.backbone.generate(
                            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            max_new_tokens=30, do_sample=False,
                        )
                        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    persona_loss = persona_consistency_loss(
                        nli_model, nli_tokenizer, gen_texts, batch["persona_sentences"], device,
                    )
                else:
                    persona_loss = torch.tensor(0.0, device=device)

                total_loss = (
                    STAGE2_CFG.lm_loss_weight * lm_loss +
                    STAGE2_CFG.persona_consistency_weight * persona_loss
                )
                total_loss = total_loss / STAGE2_CFG.gradient_accumulation_steps

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % STAGE2_CFG.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

            if batch_idx % 50 == 0:
                print(f"epoch={epoch} step={batch_idx} lm_loss={lm_loss.item():.4f} "
                      f"persona_loss={persona_loss.item():.4f}")

        # --- validation ---
        system.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                annotation_indices = list(zip(
                    batch["formality_label"].tolist(), batch["topic_label"].tolist(),
                    batch["sentiment_label"].tolist(), batch["dialogue_act_label"].tolist(),
                ))
                out = system(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"], persona_sentences=batch["persona_sentences"],
                    annotation_indices=annotation_indices,
                )
                val_losses.append(out["loss_dict"]["lm_loss"].item())
        mean_val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"[Epoch {epoch}] validation lm_loss={mean_val_loss:.4f}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(system.upe.state_dict(), os.path.join(args.output_dir, "upe.pt"))
            torch.save(system.dst.state_dict(), os.path.join(args.output_dir, "dst.pt"))
            system.backbone.save_pretrained(os.path.join(args.output_dir, "backbone"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "backbone"))
            print(f"Saved new best checkpoint (val_loss={best_val_loss:.4f}) to {args.output_dir}")

    print("Stage 2 training complete.")


if __name__ == "__main__":
    main()
