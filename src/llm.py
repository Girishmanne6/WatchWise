"""One generate() function for every model, with a SQLite response cache."""

import hashlib
import logging
import re
import threading
import time

from src.cache import TMDBCache
from src.config import DATA_CACHE, MODELS

logger = logging.getLogger(__name__)

_CACHE = None
# Free-tier caps are easy to trip when several workers call at once.
_PROVIDER_LIMITS = {
    "google": threading.Semaphore(1),
    "groq": threading.Semaphore(1),
    "openai": threading.Semaphore(1),
    "anthropic": threading.Semaphore(1),
}
# Free-tier spacing. One call at a time is still too fast for the per-minute cap.
# Per model when one Groq model is capped and another is not.
_MIN_INTERVAL = {
    "google": 4.0,
    # Gemma 4 returns 503 when calls land close together.
    "gemma-4-26b-a4b-it": 8.0,
    "gemma-4-31b-it": 8.0,
    "openai/gpt-oss-20b": 4.5,
    "openai/gpt-oss-120b": 8.0,
    "qwen/qwen3.8-27b": 0.4,
}
_NEXT_OK = {}
_PACE_LOCK = threading.Lock()


def _pace(provider: str, model_id: str) -> None:
    gap = _MIN_INTERVAL.get(model_id, _MIN_INTERVAL.get(provider))
    if not gap:
        return
    key = model_id or provider
    with _PACE_LOCK:
        now = time.perf_counter()
        slot = max(now, _NEXT_OK.get(key, 0.0))
        _NEXT_OK[key] = slot + gap
        wait = slot - now
    if wait > 0:
        time.sleep(wait)


class ModelSkipped(RuntimeError):
    """The model has no API key in the environment."""


def _cache() -> TMDBCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = TMDBCache(DATA_CACHE / "llm.sqlite")
    return _CACHE


def _cache_key(model_key: str, model_id: str, system: str, user: str, temperature: float) -> str:
    raw = f"{model_key}\n{model_id}\n{temperature}\n{system}\n{user}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def model_is_configured(model_key: str) -> bool:
    spec = MODELS.get(model_key)
    if spec is None:
        return False
    import os

    return bool(os.getenv(spec["env"], "").strip())


def available_models(keys: list[str] | None = None) -> list[str]:
    """Models whose env var is set. Missing keys are skipped with a warning."""
    chosen = keys if keys is not None else list(MODELS)
    ready = []
    for key in chosen:
        spec = MODELS.get(key)
        if spec is None:
            logger.warning("unknown model %s", key)
            continue
        if model_is_configured(key):
            ready.append(key)
        else:
            logger.warning("skipping %s because %s is not set", key, spec["env"])
    return ready


def _cost(spec: dict, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * spec["input_per_million"]
        + output_tokens / 1_000_000 * spec["output_per_million"]
    )


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    # Daily quota errors and per-minute 429s use similar wording. Retry either
    # a few times; a real block still fails after the last attempt.
    return any(
        bit in text
        for bit in (
            "429",
            "rate limit",
            "rate_limit",
            "resource_exhausted",
            "too many",
            "500",
            "502",
            "503",
            "504",
            "deadline",
            "unavailable",
            "overloaded",
            "timeout",
            "empty completion",
        )
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait, if it named one."""
    match = re.search(r"try again in ([0-9.]+)\s*(h|hr|min|m|sec|s)\b", str(exc), re.I)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("h"):
        return amount * 3600
    if unit.startswith("m"):
        return amount * 60
    return amount


def _is_daily_limit(exc: Exception) -> bool:
    """True when the provider is done for the day, not for a few seconds."""
    text = str(exc).lower()
    if "tokens per day" in text or "tpd" in text or "daily" in text:
        return True
    wait = _retry_after_seconds(exc)
    # A 50400s wait is the daily token cap. Retrying it does not help.
    return wait is not None and wait >= 600


def _retry(fn):
    delay = 3.0
    last = None
    for attempt in range(8):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - providers raise their own types
            last = exc
            if _is_daily_limit(exc) or not _is_retryable(exc) or attempt == 7:
                raise
            wait = delay
            hinted = _retry_after_seconds(exc)
            if hinted is not None:
                wait = max(wait, hinted + 0.5)
            if "high demand" in str(exc).lower() or "unavailable" in str(exc).lower():
                wait = max(wait, 8.0)
            logger.warning(
                "retryable API error (%s), sleeping %.0fs before retry %s",
                type(exc).__name__,
                wait,
                attempt + 1,
            )
            time.sleep(wait)
            delay = min(delay * 2, 30)
    raise last


def _openai_style(spec: dict, system: str, user: str, temperature: float, groq: bool) -> dict:
    if groq:
        from groq import Groq
        import os

        client = Groq(api_key=os.environ[spec["env"]], timeout=60)
    else:
        from openai import OpenAI
        import os

        client = OpenAI(api_key=os.environ[spec["env"]], timeout=20)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    kwargs = {
        "model": spec["model_id"],
        "messages": messages,
        "temperature": temperature,
        # gpt-oss spends some of this budget on reasoning. Leave room for the answer.
        "max_tokens": 1500,
    }
    if groq and "gpt-oss" in spec["model_id"]:
        kwargs["reasoning_effort"] = "low"
    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    text = message.content or ""
    if not text.strip():
        raise RuntimeError("empty completion")
    usage = response.usage
    return {
        "text": text,
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _anthropic(spec: dict, system: str, user: str, temperature: float) -> dict:
    import os

    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ[spec["env"]], timeout=45)
    response = client.messages.create(
        model=spec["model_id"],
        max_tokens=800,
        temperature=temperature,
        system=system or "",
        messages=[{"role": "user", "content": user}],
    )
    parts = [block.text for block in response.content if getattr(block, "text", None)]
    return {
        "text": "\n".join(parts),
        "input_tokens": int(response.usage.input_tokens or 0),
        "output_tokens": int(response.usage.output_tokens or 0),
    }


def _gemini(spec: dict, system: str, user: str, temperature: float) -> dict:
    import os

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ[spec["env"]],
        # A long Gemma prompt can sit past the default client timeout.
        http_options=types.HttpOptions(timeout=120_000),
    )
    config_kwargs = {"temperature": temperature}
    if system:
        config_kwargs["system_instruction"] = system
    # gemini-3.5-flash-lite rejects thinking_config, so it stays off.
    response = client.models.generate_content(
        model=spec["model_id"],
        contents=user,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    usage = response.usage_metadata
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    text = response.text or ""
    if not text.strip():
        raise RuntimeError("empty completion")
    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _call_provider(spec: dict, system: str, user: str, temperature: float) -> dict:
    provider = spec["provider"]
    if provider == "openai":
        return _openai_style(spec, system, user, temperature, groq=False)
    if provider == "groq":
        return _openai_style(spec, system, user, temperature, groq=True)
    if provider == "anthropic":
        return _anthropic(spec, system, user, temperature)
    if provider == "google":
        return _gemini(spec, system, user, temperature)
    raise ValueError(f"unknown provider {provider}")


def generate(model_key: str, system: str, user: str, temperature: float = 0) -> dict:
    """Return text, token counts, latency, and estimated cost. Cached on disk."""
    spec = MODELS.get(model_key)
    if spec is None:
        raise ValueError(f"unknown model {model_key}")
    if not model_is_configured(model_key):
        logger.warning("skipping %s because %s is not set", model_key, spec["env"])
        raise ModelSkipped(f"{model_key} skipped, {spec['env']} is not set")

    key = _cache_key(model_key, spec["model_id"], system, user, temperature)
    cached = _cache().get(key)
    if cached is not None:
        payload = cached[0]
        payload = dict(payload)
        payload["cached"] = True
        return payload

    started = time.perf_counter()
    limit = _PROVIDER_LIMITS.get(spec["provider"])

    def _limited():
        _pace(spec["provider"], spec["model_id"])
        if limit is None:
            return _call_provider(spec, system, user, temperature)
        with limit:
            return _call_provider(spec, system, user, temperature)

    result = _retry(_limited)
    if not (result.get("text") or "").strip():
        raise RuntimeError("empty completion")
    latency = time.perf_counter() - started
    payload = {
        "text": result["text"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "latency_s": round(latency, 3),
        "cost_usd": round(_cost(spec, result["input_tokens"], result["output_tokens"]), 6),
        "cached": False,
    }
    _cache().set(key, payload)
    return payload
