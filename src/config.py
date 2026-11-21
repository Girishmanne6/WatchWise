"""Paths and environment settings for WatchWise."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

TMDB_READ_TOKEN = os.getenv("TMDB_READ_TOKEN", "")
CACHE_TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "7"))

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_CACHE = REPO_ROOT / "data" / "cache"

# Prices are USD per 1M tokens, list rates as of November 2026.
# The four keys in TEST_MODELS are the comparison. The others are judges only.
MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "env": "OPENAI_API_KEY",
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-5-5",
        "env": "ANTHROPIC_API_KEY",
        "input_per_million": 0.10,
        "output_per_million": 0.50,
    },
    "gemini-flash": {
        "provider": "google",
        # 3.8, 3.7, 3.6, and 3.5 Flash all returned a hard quota error on this key.
        # 3.5 Flash-Lite is the Google model that still answers. It rejects thinking_config.
        "model_id": "gemini-3.5-flash-lite",
        "env": "GEMINI_API_KEY",
        "input_per_million": 0.30,
        "output_per_million": 2.50,
    },
    "llama-3.1-8b": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "env": "GROQ_API_KEY",
        "input_per_million": 0.05,
        "output_per_million": 0.08,
    },
    # This Groq key does not have llama-3.1-8b-instant. gpt-oss-20b is the open model it can call.
    "gpt-oss-20b": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "env": "GROQ_API_KEY",
        "input_per_million": 0.075,
        "output_per_million": 0.30,
    },
    "gpt-oss-120b": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "env": "GROQ_API_KEY",
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    },
    # Not a judge and not in the comparison. Left here so the price is recorded.
    "qwen-27b": {
        "provider": "groq",
        "model_id": "qwen/qwen3.8-27b",
        "env": "GROQ_API_KEY",
        "input_per_million": 0.80,
        "output_per_million": 4.00,
    },
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "env": "OPENAI_API_KEY",
        "input_per_million": 2.50,
        "output_per_million": 10.00,
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-5-5",
        "env": "ANTHROPIC_API_KEY",
        "input_per_million": 2.00,
        "output_per_million": 10.00,
    },
    "llama-3.3-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "env": "GROQ_API_KEY",
        "input_per_million": 0.59,
        "output_per_million": 0.79,
    },
}

# Models that go in the comparison table. Judges are not in this list.
TEST_MODELS = ["gpt-4o-mini", "claude-haiku", "gemini-flash", "llama-3.1-8b", "gpt-oss-20b"]
# gpt-4o first. If that key fails, the only other judge is gpt-oss-120b.
# Do not add another Groq model here. It shares the daily token cap with the test model.
JUDGE_PREFERENCE = ["gpt-4o", "gpt-oss-120b"]
