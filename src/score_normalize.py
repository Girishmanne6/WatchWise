"""Turn messy Rotten Tomatoes review scores into a 0 to 1 float."""

import math
import re

# Even steps from A+ = 1.0 down to F = 0.0.
_LETTER_GRADES = [
    "A+",
    "A",
    "A-",
    "B+",
    "B",
    "B-",
    "C+",
    "C",
    "C-",
    "D+",
    "D",
    "D-",
    "F",
]
_LETTER_TO_SCORE = {
    grade: 1.0 - (index / (len(_LETTER_GRADES) - 1))
    for index, grade in enumerate(_LETTER_GRADES)
}
_FRACTION = re.compile(r"^([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)$")
_PLAIN_NUMBER = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def normalize_score(raw) -> float | None:
    """Map a Rotten Tomatoes ``review_score`` value onto 0 to 1.

    Fractions such as ``3/5``, ``3.5/4``, ``8/10``, and ``78/100`` are
    divided. A zero denominator or a result above 1 returns None.

    Letter grades ``A+`` through ``F`` sit on an even scale with ``A+``
    at 1.0 and ``F`` at 0.0. Matching ignores case and spaces.

    A bare number with no denominator is ambiguous, so it returns None.
    Empty values, NaN, and anything else also return None.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None

    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None

    text = text.replace(" ", "").upper()

    fraction = _FRACTION.fullmatch(text)
    if fraction:
        numerator = float(fraction.group(1))
        denominator = float(fraction.group(2))
        if denominator == 0:
            return None
        value = numerator / denominator
        if value < 0 or value > 1:
            return None
        return value

    if text in _LETTER_TO_SCORE:
        return _LETTER_TO_SCORE[text]

    if _PLAIN_NUMBER.fullmatch(text):
        return None

    return None
