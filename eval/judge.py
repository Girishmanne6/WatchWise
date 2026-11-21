"""LLM judge. One JSON verdict per answer, from a model that is not in the comparison when possible."""

import json
import logging
import re

from src.config import JUDGE_PREFERENCE, TEST_MODELS
from src.llm import generate, model_is_configured

logger = logging.getLogger(__name__)

_SYSTEM = """You are a strict grader for a movie assistant. Use only the reviews and stats given.
Reply with JSON only, no markdown:
{"groundedness": 0.0, "hallucination": false, "spoiler": false, "mentions_disagreement": false, "citations": [{"n": 1, "ok": true}]}
groundedness is 0 to 1: how well the claims are supported by the reviews and stats.
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
