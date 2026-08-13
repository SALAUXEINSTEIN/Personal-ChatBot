"""
Automatic metric evaluation — Section 3.8.1.

Computes, for a given system variant on the PersonaChat test set:
    - Perplexity (PPL)
    - BLEU-1/2/4 (NLTK)
    - ROUGE-L (Hugging Face `evaluate` library)
    
    - Persona Consistency (C-Score) via NLI entailment
    - Distinct-1/2 (response diversity)
Bootstrap resampling (10,000 iterations) is used for 95% CIs.
"""

from __future__ import annotations
import json
import math
import os
import sys
from typing import List, Dict

import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import EVAL_CFG


def compute_perplexity(model, tokenizer, texts: List[str], device) -> float:
    model.eval()
    nlls, total_tokens = [], 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = model(**enc, labels=enc["input_ids"])
            n_tokens = enc["input_ids"].shape[1]
            nlls.append(out.loss.item() * n_tokens)
            total_tokens += n_tokens
    return math.exp(sum(nlls) / max(total_tokens, 1))


def compute_bleu(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    smoothie = SmoothingFunction().method4
    bleu1, bleu2, bleu4 = [], [], []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens, hyp_tokens = ref.split(), hyp.split()
        bleu1.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu2.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu4.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))
    return {"bleu1": float(np.mean(bleu1)), "bleu2": float(np.mean(bleu2)), "bleu4": float(np.mean(bleu4))}


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    import evaluate
    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=hypotheses, references=references, rouge_types=["rougeL"])
    return float(result["rougeL"])


def compute_distinct(hypotheses: List[str]) -> Dict[str, float]:
    unigrams, bigrams, total_uni, total_bi = set(), set(), 0, 0
    for h in hypotheses:
        toks = h.split()
        unigrams.update(toks)
        bigrams.update(zip(toks, toks[1:]))
        total_uni += len(toks)
        total_bi += max(len(toks) - 1, 0)
    return {
        "distinct1": len(unigrams) / max(total_uni, 1),
        "distinct2": len(bigrams) / max(total_bi, 1),
    }


def compute_persona_consistency(nli_model, nli_tokenizer, personas_batch: List[List[str]],
                                 hypotheses: List[str], device) -> float:
    """
    C-Score: proportion of generated responses entailed by or neutral
    w.r.t. the conditioning persona sentences (Section 3.8.1).
    """
    entailed_or_neutral = 0
    for personas, hyp in zip(personas_batch, hypotheses):
        premise = " ".join(personas) if personas else "No stated preferences."
        enc = nli_tokenizer(premise, hyp, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = nli_model(**enc).logits
        pred_class = logits.argmax(-1).item()
        # Convention: last two classes assumed {neutral, entailment}; first = contradiction.
        if pred_class != 0:
            entailed_or_neutral += 1
    return entailed_or_neutral / max(len(hypotheses), 1)


def bootstrap_ci(values: List[float], n_iterations: int = None, confidence: float = None):
    n_iterations = n_iterations or EVAL_CFG.bootstrap_iterations
    confidence = confidence or EVAL_CFG.confidence_level
    values = np.array(values)
    means = [np.mean(np.random.choice(values, size=len(values), replace=True))
             for _ in range(n_iterations)]
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    return float(np.mean(values)), float(lower), float(upper)


def evaluate_variant(system, tokenizer, test_records: List[dict], device,
                      nli_model=None, nli_tokenizer=None) -> Dict:
    """
    Runs the full automatic-metric suite for one system variant over a
    list of {persona_sentences, history, response} test records.
    """
    references, hypotheses, personas_batch = [], [], []

    for rec in test_records:
        hyp = system.generate_response(rec.get("persona_sentences", []), rec["history"])
        references.append(rec["response"])
        hypotheses.append(hyp)
        personas_batch.append(rec.get("persona_sentences", []))

    ppl = compute_perplexity(system.backbone, tokenizer, references, device)
    bleu = compute_bleu(references, hypotheses)
    rouge_l = compute_rouge_l(references, hypotheses)
    distinct = compute_distinct(hypotheses)

    result = {"perplexity": ppl, **bleu, "rouge_l": rouge_l, **distinct}

    if nli_model is not None:
        result["persona_consistency"] = compute_persona_consistency(
            nli_model, nli_tokenizer, personas_batch, hypotheses, device)

    # Bootstrap CI on BLEU-2 as the primary comparison metric (Section 3.8.1)
    bleu2_scores = []
    for ref, hyp in zip(references, hypotheses):
        smoothie = SmoothingFunction().method4
        bleu2_scores.append(sentence_bleu([ref.split()], hyp.split(),
                                           weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
    mean, lo, hi = bootstrap_ci(bleu2_scores)
    result["bleu2_ci95"] = [lo, hi]

    return result


if __name__ == "__main__":
    print("This module exposes evaluate_variant(); see evaluation/ablation_study.py "
          "for a full end-to-end runnable comparison across system variants.")
