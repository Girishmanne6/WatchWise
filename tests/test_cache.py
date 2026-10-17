"""Checks for the SQLite TMDB cache."""

import pytest

from src.cache import TMDBCache


def test_set_then_get_returns_same_payload(tmp_path):
    cache = TMDBCache(tmp_path / "tmdb.sqlite")
    payload = {"id": 109439, "title": "Carrie", "results": [1, 2]}
    cache.set("search:carrie", payload)

    got = cache.get("search:carrie")
    assert got is not None
    body, age_days = got
    assert body == payload
    assert age_days == pytest.approx(0, abs=1e-3)


def test_missing_key_returns_none(tmp_path):
    cache = TMDBCache(tmp_path / "tmdb.sqlite")
    assert cache.get("does-not-exist") is None


def test_set_replaces_existing_key(tmp_path):
    cache = TMDBCache(tmp_path / "tmdb.sqlite")
    cache.set("movie:1", {"title": "old"})
    cache.set("movie:1", {"title": "new"})
    body, _age = cache.get("movie:1")
    assert body == {"title": "new"}
