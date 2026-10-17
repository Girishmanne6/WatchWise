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
