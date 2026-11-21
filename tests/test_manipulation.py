"""The duplicate detector should catch copied reviews from different authors."""

import numpy as np

from src.manipulation import flag_suspicious


def _embed(texts):
    vectors = []
    for text in texts:
        # Same text, same vector. Different text, a different axis.
        code = abs(hash(text)) % 997
        vector = np.zeros(8, dtype=float)
        vector[code % 8] = 1.0
        vector[0] += code / 997
        vector = vector / np.linalg.norm(vector)
        vectors.append(vector)
    return np.vstack(vectors)


def test_catches_injected_duplicates():
    reviews = [
        {"author": "Ann", "content": "totally unique take on the lighting", "source": "audience", "date": "2020-01-01"},
        {"author": "Ben", "content": "worst movie ever 0/10", "source": "audience", "date": "2020-05-01"},
        {"author": "Cara", "content": "worst movie ever 0/10", "source": "audience", "date": "2020-05-01"},
    ]
    result = flag_suspicious(reviews, embed_fn=_embed)
    assert result["score"] > 0
    assert any("near-duplicate" in reason for reason in result["reasons"])
    assert reviews[1]["suspicious"] is True
    assert reviews[2]["suspicious"] is True
    assert reviews[0]["suspicious"] is False
