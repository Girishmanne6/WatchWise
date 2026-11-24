"""LLM judge. One JSON verdict per answer, from a model that is not in the comparison when possible."""

import json
import logging
import re

from src.config import JUDGE_PREFERENCE, TEST_MODELS
from src.llm import generate, model_is_configured

logger = logging.getLogger(__name__)

_SYSTEM = """You are a strict grader for a movie assistant. Use only the reviews and stats given.
Reply with JSON only, no markdown:
{"groundedness": <0-1>, "hallucination": false, "spoiler": false, "mentions_disagreement": false, "citations": [{"n": 1, "ok": true}]}
groundedness is a real number from 0 to 1, not only 0 or 1. Use a value in between when some claims are supported and some are not.
hallucination is true if the answer invents a director, actor, score, or plot that is not in the material.
spoiler is true only if the answer reveals plot points, twists, or the ending. A refusal is false.
mentions_disagreement is true if it says critics and audiences scored the movie differently.
citations: for each [n] attached to a claim, ok is true when review n actually supports that claim.
"""


def pick_judge(exclude: list[str] | None = None) -> str | None:
    """gpt-4o if that key works, otherwise gpt-oss-120b. Nothing else.

    A key in .env is not enough. The call has to succeed, because a bad key
    still looks configured. There is no fallback onto a model being tested,
    and no second Groq model, because those share one daily token cap.
    """
    blocked = set(exclude or [])
    blocked.update(TEST_MODELS)
    candidates = [key for key in JUDGE_PREFERENCE if key not in blocked and model_is_configured(key)]
    for key in candidates:
        try:
            generate(key, "Reply with JSON only.", '{"ok": true}', temperature=0)
            return key
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge %s unavailable: %s", key, type(exc).__name__)
    return None


def _parse(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


_HALLUCINATION = """You check one movie answer for invented facts.
Reply with JSON only: {"hallucination": false}
The reference is ground truth from our tables.
hallucination is true if the answer gives a year, director, actor, or score that contradicts the reference, or names a director or cast member who is not listed.
A score within 1 point of the reference is not a contradiction.
If directors or cast says "not in our tables", do not treat a person's name as a hallucination.
A plot claim is a hallucination only when it contradicts the dataset blurb. Refusing to spoil the plot is not a hallucination.
If you are unsure, answer false.
"""


def judge_hallucination(question: str, answer: str, facts: dict, judge_key: str) -> dict:
    """One hallucination label, using year, director, cast, and scores as the reference."""
    user = f"Question: {question}\n\nReference:\n{json.dumps(facts, default=str)}\n\nAnswer:\n{answer}"
    result = generate(judge_key, _HALLUCINATION, user, temperature=0)
    parsed = _parse(result["text"])
    return {
        "hallucination": bool(parsed.get("hallucination")),
        "_input_tokens": result["input_tokens"],
        "_output_tokens": result["output_tokens"],
        "_cost_usd": result["cost_usd"],
        "_latency_s": result["latency_s"],
        "_judge": judge_key,
    }


_GROUNDEDNESS = """You score how well one movie answer is supported by the reviews and stats.
Reply with JSON only, no markdown:
{"groundedness": <0-1>}
Replace <0-1> with a real number from 0 to 1.
0 means none of the claims are supported. 1 means every claim is supported.
Use a value in between when some claims are supported and some are not.
Do not answer only 0 or 1 unless the answer really is entirely unsupported or entirely supported.
"""


def _score_groundedness(parsed: dict) -> float:
    value = parsed.get("groundedness")
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("<") or value in {"0-1", ""}:
            raise ValueError("placeholder")
    score = float(value)
    if score < 0 or score > 1:
        raise ValueError("out of range")
    return score


def judge_groundedness(question: str, answer: str, reviews: list[dict], stats: dict, judge_key: str) -> dict:
    """One groundedness score. The reply has to be a real number from 0 to 1."""
    review_lines = []
    for review in reviews or []:
        snippet = (review.get("content") or "").replace("\n", " ")[:400]
        review_lines.append(f"[{review.get('n')}] {review.get('source')}: {snippet}")
    user = (
        f"Question: {question}\n\nStats: {json.dumps(stats, default=str)}\n\n"
        f"Reviews:\n{chr(10).join(review_lines) or '(none)'}\n\nAnswer:\n{answer}"
    )
    result = generate(judge_key, _GROUNDEDNESS, user, temperature=0)
    parsed = _parse(result["text"])
    return {
        "groundedness": _score_groundedness(parsed),
        "_input_tokens": result["input_tokens"],
        "_output_tokens": result["output_tokens"],
        "_cost_usd": result["cost_usd"],
        "_latency_s": result["latency_s"],
        "_judge": judge_key,
    }


def judge_answer(question: str, answer: str, reviews: list[dict], stats: dict, judge_key: str) -> dict:
    review_lines = []
    for review in reviews or []:
        snippet = (review.get("content") or "").replace("\n", " ")[:400]
        review_lines.append(f"[{review.get('n')}] {review.get('source')}: {snippet}")
    user = (
        f"Question: {question}\n\nStats: {json.dumps(stats, default=str)}\n\n"
        f"Reviews:\n{chr(10).join(review_lines) or '(none)'}\n\nAnswer:\n{answer}"
    )
    result = generate(judge_key, _SYSTEM, user, temperature=0)
    parsed = _parse(result["text"])
    parsed["_input_tokens"] = result["input_tokens"]
    parsed["_output_tokens"] = result["output_tokens"]
    parsed["_cost_usd"] = result["cost_usd"]
    parsed["_latency_s"] = result["latency_s"]
    parsed["_judge"] = judge_key
    return parsed
