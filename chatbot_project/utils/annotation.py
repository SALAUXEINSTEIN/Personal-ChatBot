"""
User-preference annotation utilities.

Implements Section 3.4.2, step 4 of the dissertation:
    (a) formality level        -> lexical formality classifier
    (b) topic category         -> zero-shot classification (DeBERTa-v3)
    (c) sentiment polarity     -> VADER (Hutto & Gilbert, 2014)

These annotations become the "annotation feature vector" consumed by the
User Preference Encoder (Section 3.5.3) and are also the supervision
targets for the Dialogue State Tracker (Section 3.5.4).
"""

from __future__ import annotations
import re
from functools import lru_cache
from typing import Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# (a) Formality classifier
# ---------------------------------------------------------------------------
# A lightweight lexical formality heuristic is used rather than a heavy
# trained classifier, consistent with the project's compute constraints
# (Section 3.6.1). It can be swapped for a fine-tuned classifier without
# changing any downstream interface.

_INFORMAL_MARKERS = {
    "gonna", "wanna", "gotta", "lol", "lmao", "omg", "yeah", "yep", "nah",
    "u", "ur", "btw", "kinda", "sorta", "haha", "hey", "cool", "dude",
}
_CONTRACTION_PATTERN = re.compile(r"\b\w+'(?:t|re|ve|ll|d|s|m)\b", re.IGNORECASE)


def classify_formality(text: str) -> str:
    """Returns 'formal' or 'informal' for a single utterance."""
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return "formal"
    informal_hits = sum(1 for t in tokens if t in _INFORMAL_MARKERS)
    contraction_hits = len(_CONTRACTION_PATTERN.findall(text))
    exclamations = text.count("!")
    informal_score = informal_hits + 0.5 * contraction_hits + 0.25 * exclamations
    ratio = informal_score / max(len(tokens), 1)
    return "informal" if ratio > 0.08 else "formal"


# ---------------------------------------------------------------------------
# (b) Zero-shot topic classifier (DeBERTa-v3, Laurer et al. 2022)
# ---------------------------------------------------------------------------
DEFAULT_TOPICS = [
    "family", "work", "hobbies", "food", "travel", "health",
    "education", "entertainment", "sports", "technology",
]


@lru_cache(maxsize=1)
def _get_zero_shot_pipeline():
    from transformers import pipeline
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    )


def classify_topic(text: str, candidate_topics: List[str] = None) -> Dict[str, float]:
    """
    Returns a probability distribution over candidate_topics for `text`.
    Falls back to a uniform distribution if the zero-shot model cannot be
    loaded (e.g. no network access in a constrained environment) so the
    rest of the pipeline can still run end-to-end.
    """
    candidate_topics = candidate_topics or DEFAULT_TOPICS
    try:
        clf = _get_zero_shot_pipeline()
        result = clf(text, candidate_labels=candidate_topics, multi_label=False)
        return dict(zip(result["labels"], result["scores"]))
    except Exception:
        uniform = 1.0 / len(candidate_topics)
        return {t: uniform for t in candidate_topics}


# ---------------------------------------------------------------------------
# (c) VADER sentiment
# ---------------------------------------------------------------------------
_vader = SentimentIntensityAnalyzer()


def classify_sentiment(text: str) -> Dict[str, float]:
    """Returns VADER compound/pos/neu/neg scores and a discrete label."""
    scores = _vader.polarity_scores(text)
    if scores["compound"] >= 0.05:
        label = "positive"
    elif scores["compound"] <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {**scores, "label": label}


# ---------------------------------------------------------------------------
# Convenience wrapper producing the full annotation used everywhere else
# ---------------------------------------------------------------------------
def annotate_utterance(text: str, candidate_topics: List[str] = None) -> Dict:
    return {
        "text": text,
        "formality": classify_formality(text),
        "topic_distribution": classify_topic(text, candidate_topics),
        "sentiment": classify_sentiment(text),
    }


if __name__ == "__main__":
    sample = "OMG I can't wait for the trip, gonna be so much fun!!"
    import json
    print(json.dumps(annotate_utterance(sample), indent=2))
