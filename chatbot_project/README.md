# Personalised Transformer Chatbot — Reference Implementation

This is a working implementation of the system designed in **Chapters 1–3**
of the dissertation *"Development of a Chatbot with Personalised Dialogue
Management Using Transformers"*. Every module is written to match a specific
methodological decision made in Chapter 3, so the codebase can be used
directly to produce the results reported in Chapter 4 (System Development)
and Chapter 5 (Evaluation).

## 1. Architecture recap (Section 3.5)

```
                 ┌─────────────────────────┐
   persona +     │  User Preference        │   prefix (past_key_values)
   annotations ─▶│  Encoder (UPE)          │──────────────┐
                 │  frozen S-BERT + MLP    │              │
                 └─────────────────────────┘              ▼
                                                  ┌──────────────────┐
   dialogue history ───────────────────────────▶ │ DialoGPT-medium   │──▶ response
                                                  │ (backbone)        │
                                                  └──────────────────┘
                                                            ▲
                 ┌─────────────────────────┐               │
   last 3 turns ▶│ Dialogue State Tracker  │───────────────┘
                 │ (BERT-base classifier)  │   feeds updated state back
                 └─────────────────────────┘   into the UPE next turn
```

## 2. File map -> dissertation section

| File | Dissertation section |
|---|---|
| `configs/config.py` | All hyperparameters, Tables 3.1–3.3 |
| `utils/annotation.py` | 3.4.2 step 4 — formality / topic / sentiment annotation |
| `data/preprocess_personachat.py` | 3.4.2 — PersonaChat pipeline (tokenisation, persona block, windowing, filtering) |
| `data/preprocess_dailydialog.py` | 3.4.3 — DailyDialog pipeline + 70:30 corpus mixing |
| `data/dataset.py` | PyTorch `Dataset`/collator feeding both training stages |
| `data/eda.py` | 3.4.4 — exploratory data analysis |
| `models/backbone.py` | 3.5.2 — DialoGPT-medium backbone |
| `models/upe.py` | 3.5.3 — User Preference Encoder (prefix-tuning) |
| `models/dst.py` | 3.5.4 — Dialogue State Tracker + EMA session state |
| `models/personalised_chatbot.py` | 3.5.1 — full system wiring + ablation variant factory |
| `training/train_stage1.py` | 3.6.2, Table 3.2 — backbone fine-tuning |
| `training/train_stage2.py` | 3.6.2 — joint UPE+DST training, weighted composite loss |
| `training/hyperparameter_search.py` | 3.6.3 — targeted grid search |
| `evaluation/automatic_metrics.py` | 3.8.1 — PPL, BLEU, ROUGE-L, C-Score, Distinct-1/2, bootstrap CIs |
| `evaluation/ablation_study.py` | 3.6.4 — B / V1 / V2 / V3 / FS comparison |
| `evaluation/bias_audit.py` | 3.9.4 — Winogender/BBQ-adapted bias probes |
| `app/gradio_app.py` | 3.7 — Gradio interface (chat window, profile panel, persona toggle, feedback button, disclaimer) |
| `run_pipeline.py` | Orchestrates all of the above end-to-end |

## 3. Setup

```bash
python -m venv venv && source venv/bin/activate     # or use Colab directly
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

A CUDA GPU is strongly recommended (the dissertation used Colab Pro
A100/V100 instances — Section 3.6.1). CPU-only execution works for
small `--limit` smoke tests but is not practical for full training.

## 4. Quick start (small smoke test, ~500 examples per split)

```bash
python run_pipeline.py --limit 500
```

This will: pre-process both datasets → build the combined 70:30 corpus →
fine-tune the backbone (Stage 1) → jointly train the UPE+DST (Stage 2) →
run the five-variant ablation study → write `evaluation/ablation_results.json`.

## 5. Full dissertation-scale run

```bash
# 1. Pre-process (no --limit = full dataset sizes from Section 3.4.2)
python -m data.preprocess_personachat --split train
python -m data.preprocess_personachat --split validation
python -m data.preprocess_personachat --split test
python -m data.preprocess_dailydialog preprocess --split train
python -m data.preprocess_dailydialog preprocess --split validation
python -m data.preprocess_dailydialog preprocess --split test

# 2. Combine (70:30 ratio, Section 3.4.3)
python -m data.preprocess_dailydialog combine \
    --personachat data/processed/personachat_train.jsonl \
    --dailydialog data/processed/dailydialog_train.jsonl \
    --out data/processed/combined_train.jsonl
python -m data.preprocess_dailydialog combine \
    --personachat data/processed/personachat_validation.jsonl \
    --dailydialog data/processed/dailydialog_validation.jsonl \
    --out data/processed/combined_val.jsonl

# 3. Stage 1 — backbone fine-tuning (Table 3.2)
python -m training.train_stage1 \
    --train_file data/processed/combined_train.jsonl \
    --val_file data/processed/combined_val.jsonl \
    --use_wandb

# 4. Stage 2 — joint UPE + DST training
python -m training.train_stage2 \
    --backbone_checkpoint checkpoints/stage1_backbone \
    --train_file data/processed/personachat_train.jsonl \
    --val_file data/processed/personachat_validation.jsonl

# 5. (Optional) targeted hyperparameter search — Section 3.6.3
python -m training.hyperparameter_search \
    --train_file data/processed/combined_train.jsonl \
    --val_file data/processed/combined_val.jsonl \
    --personachat_processed data/processed/personachat_train.jsonl \
    --dailydialog_processed data/processed/dailydialog_train.jsonl

# 6. Ablation study — Section 3.6.4 / automatic metrics — Section 3.8.1
python -m evaluation.ablation_study \
    --backbone_checkpoint checkpoints/stage1_backbone \
    --stage2_checkpoint checkpoints/stage2_full_system \
    --test_file data/processed/personachat_test.jsonl

# 7. Bias audit — Section 3.9.4 (import run_bias_audit after loading a system; see file docstring)

# 8. Launch the demo interface — Section 3.7
python -m app.gradio_app \
    --backbone_checkpoint checkpoints/stage1_backbone \
    --stage2_checkpoint checkpoints/stage2_full_system \
    --share
```

## 6. Human evaluation (Section 3.8.2)

This codebase automates the *technical* pipeline. The human evaluation study
(20 participants, Latin-Square-counterbalanced sessions across variants B,
V1, V3, FS, 5-point Likert ratings, thematic analysis of open feedback) is a
**process**, not code — but the Gradio app's feedback button
(`app/gradio_app.py`) already logs participant ratings/comments to
`app/feedback_log.csv` in the schema you'll need for the Friedman/Wilcoxon
analysis in Section 3.8.3. A minimal analysis snippet:

```python
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

df = pd.read_csv("app/feedback_log.csv")
# pivot ratings by variant/participant, then:
# friedmanchisquare(*[group['rating'] for _, group in df.groupby('variant')])
```

You will need to extend the feedback schema with a `variant` column when
running the counterbalanced study (currently the toggle only distinguishes
persona-on/off, not the full B/V1/V3/FS set) — see the note in Section 8
below.

## 7. Ethical / deployment notes (Section 3.9)

- The Gradio app displays the mandatory disclaimer (3.9.5) on load.
- No participant data is written anywhere except the local
  `feedback_log.csv`, which stores only the participant code you type in
  (use anonymised codes P001, P002, ... per 3.9.3 — never real names).
- Obtain institutional ethics approval **before** recruiting any human
  evaluation participants (3.9.1) — this code does not gate on that, it's
  your responsibility as the researcher.

## 8. Known simplifications / what you'll still need to adapt

This is a complete, runnable reference implementation, but a few things are
intentionally left as configuration points rather than hard-coded, since
the dissertation treats them as tunable/empirical:

- **PersonaChat source**: `bavard/personachat_truecased` is used as a
  convenient Hugging Face mirror. If your ethics/data-access approval
  specifies obtaining it via ParlAI directly (as Section 3.4.2 states),
  swap the loader in `data/preprocess_personachat.py::process_split`.
- **Formality classifier**: a lexical heuristic is used to keep the
  pipeline runnable without extra training; swap in a trained classifier
  if your dissertation results require one.
- **NLI model for persona consistency**: `facebook/bart-large-mnli` /
  the configured zero-shot DeBERTa-v3 are used out of the box; the
  dissertation mentions a "fine-tuned DeBERTa-v3 NLI model" (3.8.1) —
  fine-tune one on persona-entailment pairs if you need that exact setup.
- **Gradio variant toggle**: currently a simple persona on/off switch;
  extend `app/gradio_app.py` with a hidden variant-selection mechanism
  (and a `variant` column in the feedback log) for the counterbalanced
  Latin-Square human study described in 3.8.2.
- **Hyperparameter overrides**: `training/hyperparameter_search.py`
  orchestrates the grid but stops short of wiring `OVERRIDE_LR` /
  `OVERRIDE_PERSONA_WEIGHT` into `train_stage1.py` — add a few lines
  reading those env vars into `STAGE1_CFG`/`STAGE2_CFG` before your
  actual search run.

## 9. Directory layout

```
chatbot_project/
├── configs/config.py
├── utils/annotation.py
├── data/
│   ├── preprocess_personachat.py
│   ├── preprocess_dailydialog.py
│   ├── dataset.py
│   └── eda.py
├── models/
│   ├── backbone.py
│   ├── upe.py
│   ├── dst.py
│   └── personalised_chatbot.py
├── training/
│   ├── train_stage1.py
│   ├── train_stage2.py
│   └── hyperparameter_search.py
├── evaluation/
│   ├── automatic_metrics.py
│   ├── ablation_study.py
│   └── bias_audit.py
├── app/gradio_app.py
├── run_pipeline.py
└── requirements.txt
```
