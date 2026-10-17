"""TMDB client with a SQLite cache and a stale-cache fallback."""

import logging
import time

import requests

from src.cache import TMDBCache
from src.config import CACHE_TTL_DAYS, DATA_CACHE, TMDB_READ_TOKEN

logger = logging.getLogger(__name__)

BASE_URL = "https://api.themoviedb.org"


class TMDBError(Exception):
    """Raised when TMDB cannot be called and there is no cached copy to use."""

    def __init__(self, message: str, fallback_ok: bool = False):
        super().__init__(message)
        self.fallback_ok = fallback_ok


class TMDBClient:
    """Call TMDB search, movie details, and reviews.

    Every request is cached. A hit younger than ``CACHE_TTL_DAYS`` is
    returned as-is. If the network, a 5xx, or a 429 that survives retries
    gets in the way, a stale cached copy is used and a warning is logged.
    """

    def __init__(self, token: str | None = None, cache: TMDBCache | None = None, ttl_days: int | None = None):
        self.token = TMDB_READ_TOKEN if token is None else token
        self.ttl_days = CACHE_TTL_DAYS if ttl_days is None else ttl_days
        self.cache = cache if cache is not None else TMDBCache(DATA_CACHE / "tmdb.sqlite")

    @staticmethod
    def _cache_key(path: str, params: dict) -> str:
        parts = [f"{key}={params[key]}" for key in sorted(params)]
        if not parts:
            return path
        return f"{path}?{'&'.join(parts)}"

    def _get(self, path: str, params: dict | None = None) -> dict:
        params = {key: value for key, value in (params or {}).items() if value is not None}
        key = self._cache_key(path, params)
        cached = self.cache.get(key)

        if cached is not None and cached[1] < self.ttl_days:
            logger.info("cache hit for %s (age %.4f days)", key, cached[1])
            return cached[0]

        try:
            payload = self._request(path, params)
        except TMDBError as exc:
            if cached is not None and exc.fallback_ok:
                logger.warning(
                    "TMDB request failed (%s); using stale cache for %s (age %.1f days)",
                    exc,
                    key,
                    cached[1],
                )
                return cached[0]
            if cached is None:
                raise TMDBError(
                    f"{exc} Nothing is cached for {key}, so there is no fallback."
                ) from exc
            raise

        self.cache.set(key, payload)
        logger.info("fetched from TMDB %s", key)
        return payload

    def _request(self, path: str, params: dict) -> dict:
        if not self.token:
            raise TMDBError(
                "TMDB_READ_TOKEN is not set. Copy .env.example to .env and paste "
                "the API Read Access Token."
            )

        url = f"{BASE_URL}{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        backoff = 1.0

        # One try plus up to 3 retries when TMDB answers 429.
        for attempt in range(4):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
            except requests.RequestException as exc:
                raise TMDBError(
                    f"Network error calling TMDB {path}: {exc}",
                    fallback_ok=True,
                ) from exc

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after is not None else backoff
                except ValueError:
                    wait = backoff
                if attempt == 3:
                    raise TMDBError(
                        f"TMDB still returned 429 for {path} after 3 retries.",
                        fallback_ok=True,
                    )
                logger.warning(
                    "TMDB 429 on %s, sleeping %.1fs before retry %s",
                    path,
                    max(wait, 0),
                    attempt + 1,
                )
                time.sleep(max(wait, 0))
                backoff *= 2
                continue

            if response.status_code >= 500:
                raise TMDBError(
                    f"TMDB server error {response.status_code} for {path}.",
                    fallback_ok=True,
                )

            if not response.ok:
                detail = response.text[:300].replace("\n", " ")
                raise TMDBError(
                    f"TMDB request failed ({response.status_code}) for {path}: {detail}"
                )

            try:
                return response.json()
            except ValueError as exc:
                raise TMDBError(
                    f"TMDB returned non-JSON for {path}.",
                    fallback_ok=True,
                ) from exc

        raise TMDBError(f"TMDB request failed for {path}.", fallback_ok=True)

    def search_movie(self, title: str, year: int | None = None) -> list[dict]:
        """Return the raw TMDB search results for a title and optional year."""
        params: dict = {"query": title}
        if year is not None:
            params["year"] = year
        payload = self._get("/3/search/movie", params)
        results = payload.get("results") or []
        return list(results)

    def get_movie(self, tmdb_id: int) -> dict:
        """Return the TMDB movie details object."""
        return self._get(f"/3/movie/{tmdb_id}")

    def get_reviews(self, tmdb_id: int, max_pages: int = 3) -> list[dict]:
        """Return audience reviews as author, content, rating, and created_at."""
        reviews: list[dict] = []
        for page in range(1, max_pages + 1):
            payload = self._get(f"/3/movie/{tmdb_id}/reviews", {"page": page})
            for item in payload.get("results") or []:
                author_details = item.get("author_details") or {}
                reviews.append(
                    {
                        "author": item.get("author"),
                        "content": item.get("content"),
                        "rating": author_details.get("rating"),
                        "created_at": item.get("created_at"),
                    }
                )
            total_pages = payload.get("total_pages") or 1
            if page >= total_pages:
                break
        return reviews

    def find_movie(self, title: str, year: int | None = None) -> dict | None:
        """Pick the best search hit.

        An exact title match (case-insensitive) whose release year matches
        wins. Titles repeat, so year matters (there are several movies
        called Carrie). If nothing matches exactly, the most popular
        result is used.
        """
        results = self.search_movie(title, year=year)
        if not results:
            return None

        wanted = title.strip().lower()
        year_text = None if year is None else str(year)
        exact = [
            result
            for result in results
            if (result.get("title") or "").strip().lower() == wanted
            and (year_text is None or _release_year(result) == year_text)
        ]
        pool = exact or results
        return max(pool, key=_popularity)


def _release_year(result: dict) -> str | None:
    release_date = result.get("release_date") or ""
    if len(release_date) >= 4 and release_date[:4].isdigit():
        return release_date[:4]
    return None


def _popularity(result: dict) -> float:
    try:
        return float(result.get("popularity") or 0)
    except (TypeError, ValueError):
        return 0.0
