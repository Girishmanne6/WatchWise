"""Simple, explainable checks for reviews that look manipulated."""

import re

import numpy as np

_EXTREME = re.compile(
    r"\b(best movie ever|worst movie ever|masterpiece|absolute trash|10/10|0/10|10 out of 10|0 out of 10)\b",
    re.I,
)


def _words(text: str) -> int:
    return len((text or "").split())


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    # Vectors are expected to be normalized. Clip for safety.
    sims = vectors @ vectors.T
    return np.clip(sims, -1.0, 1.0)


def flag_suspicious(reviews: list[dict], embed_fn=None) -> dict:
    """Score a movie's reviews from 0 to 1 and list why.

    Signals: near-duplicate text from different authors (cosine > 0.9),
    a burst of reviews on one day, and very short extreme blurbs.
    """
    flagged = {id(review): [] for review in reviews}
    if not reviews:
        return {"score": 0.0, "reasons": [], "flagged_indexes": []}

    # Near duplicates.
    texts = [review.get("content") or "" for review in reviews]
    if embed_fn is not None and len(reviews) >= 2:
        vectors = np.asarray(embed_fn(texts), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        sims = _cosine_matrix(vectors)
        for i in range(len(reviews)):
            for j in range(i + 1, len(reviews)):
                if sims[i, j] <= 0.9:
                    continue
                author_i = (reviews[i].get("author") or "").strip().lower()
                author_j = (reviews[j].get("author") or "").strip().lower()
                if author_i and author_i == author_j:
                    continue
                flagged[id(reviews[i])].append("near-duplicate of another author")
                flagged[id(reviews[j])].append("near-duplicate of another author")

    # Bursts: one day holds several reviews and is well above the usual day.
    by_day: dict[str, list[int]] = {}
    for index, review in enumerate(reviews):
        day = str(review.get("date") or "")[:10]
        if len(day) == 10:
            by_day.setdefault(day, []).append(index)
    if by_day:
        counts = [len(indexes) for indexes in by_day.values()]
        typical = float(np.median(counts))
        cutoff = max(5, typical * 3)
        for indexes in by_day.values():
            if len(indexes) >= cutoff and len(indexes) >= 4:
                for index in indexes:
                    flagged[id(reviews[index])].append("burst of reviews on one day")

    # Very short and extreme.
    for review in reviews:
        content = review.get("content") or ""
        if _words(content) <= 12 and _EXTREME.search(content):
            flagged[id(review)].append("very short extreme review")

    reasons = []
    flagged_indexes = []
    for index, review in enumerate(reviews):
        why = list(dict.fromkeys(flagged[id(review)]))
        review["suspicious"] = bool(why)
        review["suspicion_reasons"] = why
        if why:
            flagged_indexes.append(index)
            for reason in why:
                if reason not in reasons:
                    reasons.append(reason)

    score = min(1.0, len(flagged_indexes) / len(reviews))
    return {"score": round(score, 3), "reasons": reasons, "flagged_indexes": flagged_indexes}
