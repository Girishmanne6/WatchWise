"""Find a movie, then rank only that movie's reviews.

Reviews are always about one title, and a title has at most a few hundred
critic blurbs, so there is no giant index over the whole corpus.
"""

import hashlib
import logging
import re
import threading

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.config import DATA_CACHE, DATA_PROCESSED
from src.manipulation import flag_suspicious

logger = logging.getLogger(__name__)

_MOVIES = None
_REVIEWS = None
_EMBEDDER = None
_EMBED_LOCK = threading.Lock()


def _norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    global _MOVIES, _REVIEWS
    if _MOVIES is None:
        _MOVIES = pd.read_parquet(DATA_PROCESSED / "movies.parquet")
        _REVIEWS = pd.read_parquet(DATA_PROCESSED / "reviews.parquet")
        _REVIEWS = _REVIEWS.set_index("rotten_tomatoes_link", drop=False)
    return _MOVIES, _REVIEWS


def _year(value) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value)
        return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else None


def _from_row(row) -> dict:
    year = _year(row.get("release_year"))
    tomato = row.get("tomatometer_rating")
    audience = row.get("audience_rating")
    gap = row.get("critic_audience_gap")
    return {
        "title": str(row.get("movie_title") or ""),
        "year": year,
        "source": "kaggle",
        "ambiguous": False,
        "candidates": [],
        "rotten_tomatoes_link": row.get("rotten_tomatoes_link"),
        "tmdb_id": None,
        "tomatometer_rating": None if pd.isna(tomato) else float(tomato),
        "audience_rating": None if pd.isna(audience) else float(audience),
        "tomatometer_count": None if pd.isna(row.get("tomatometer_count")) else float(row.get("tomatometer_count")),
        "audience_count": None if pd.isna(row.get("audience_count")) else float(row.get("audience_count")),
        "critic_audience_gap": None if pd.isna(gap) else float(gap),
        "overview": row.get("movie_info") if isinstance(row.get("movie_info"), str) else None,
        "genres": row.get("genres") if isinstance(row.get("genres"), str) else None,
    }


def _from_tmdb(hit: dict) -> dict:
    release = hit.get("release_date") or ""
    year = int(release[:4]) if len(release) >= 4 and release[:4].isdigit() else None
    vote = hit.get("vote_average")
    votes = hit.get("vote_count")
    audience = None if vote is None else float(vote) * 10.0
    return {
        "title": hit.get("title") or "",
        "year": year,
        "source": "tmdb",
        "ambiguous": False,
        "candidates": [],
        "rotten_tomatoes_link": None,
        "tmdb_id": hit.get("id"),
        "tomatometer_rating": None,
        "audience_rating": audience,
        "tomatometer_count": None,
        "audience_count": None if votes is None else float(votes),
        "critic_audience_gap": None,
        "overview": hit.get("overview"),
        "genres": None,
    }


def _exact_matches(movies: pd.DataFrame, title: str, year: int | None) -> pd.DataFrame:
    key = _norm(title)
    titles = movies["movie_title"].map(_norm)
    matched = movies[titles == key]
    if year is not None and "release_year" in matched.columns:
        years = matched["release_year"].map(_year)
        matched = matched[years == year]
    return matched


def _fuzzy_matches(movies: pd.DataFrame, title: str, year: int | None) -> pd.DataFrame:
    key = _norm(title)
    pool = movies
    if year is not None:
        years = movies["release_year"].map(_year)
        pool = movies[years == year]
        if pool.empty:
            pool = movies
    scored = []
    for idx, row in pool.iterrows():
        name = _norm(str(row["movie_title"]))
        score = max(fuzz.ratio(key, name), fuzz.WRatio(key, name))
        if score >= 85:
            scored.append((score, idx))
    if not scored:
        return movies.iloc[0:0]
    scored.sort(reverse=True)
    best = scored[0][0]
    # Keep near-ties so "Carrie" doesn't collapse to one film on a typo of a unique title.
    keep = [idx for score, idx in scored if score >= best - 3]
    if best >= 90 and (len(scored) == 1 or scored[1][0] < best - 5):
        keep = [scored[0][1]]
    return movies.loc[keep]


def resolve_movie(title: str, year: int | None = None, movies: pd.DataFrame | None = None, tmdb=None) -> dict | None:
    """Match title + year in the Kaggle table, then TMDB if that misses."""
    if movies is None:
        movies, _reviews = load_tables()
    year = _year(year)

    # Anything after the scrape is not in the Kaggle file.
    if year is not None and year > 2020:
        return _tmdb_lookup(title, year, tmdb)

    exact = _exact_matches(movies, title, year)
    if year is None:
        exact_any = _exact_matches(movies, title, None)
        years = {_year(value) for value in exact_any.get("release_year", [])}
        years.discard(None)
        if len(exact_any) > 1 and len(years) > 1:
            candidates = [_from_row(row) for _, row in exact_any.drop_duplicates("release_year").iterrows()]
            candidates.sort(key=lambda item: item["year"] or 0)
            return {
                "title": title,
                "year": None,
                "source": "kaggle",
                "ambiguous": True,
                "candidates": candidates,
                "rotten_tomatoes_link": None,
                "tmdb_id": None,
            }
    if len(exact) >= 1:
        # Same title and year can still have duplicate rows. Prefer more critic reviews.
        exact = exact.sort_values("tomatometer_count", ascending=False)
        return _from_row(exact.iloc[0])

    fuzzy = _fuzzy_matches(movies, title, year)
    if len(fuzzy) == 1:
        return _from_row(fuzzy.iloc[0])
    if len(fuzzy) > 1 and year is None:
        candidates = [_from_row(row) for _, row in fuzzy.iterrows()]
        return {
            "title": title,
            "year": None,
            "source": "kaggle",
            "ambiguous": True,
            "candidates": candidates,
            "rotten_tomatoes_link": None,
            "tmdb_id": None,
        }
    if len(fuzzy) > 1:
        fuzzy = fuzzy.sort_values("tomatometer_count", ascending=False)
        return _from_row(fuzzy.iloc[0])

    return _tmdb_lookup(title, year, tmdb)


def _tmdb_lookup(title: str, year: int | None, tmdb):
    client = tmdb
    if client is None:
        from src.tmdb_client import TMDBClient, TMDBError

        client = TMDBClient()
        try:
            hit = client.find_movie(title, year)
        except TMDBError as exc:
            logger.warning("TMDB lookup failed: %s", type(exc).__name__)
            return None
    else:
        hit = client.find_movie(title, year)
    if not hit:
        return None
    return _from_tmdb(hit)


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts with MiniLM. The same text list is reused from disk."""
    if not texts:
        return np.zeros((0, 384), dtype=float)
    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
    folder = DATA_CACHE / "embeddings"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest}.npy"
    if path.exists():
        return np.load(path)
    # MiniLM is one shared model. One encode at a time keeps it from crashing.
    with _EMBED_LOCK:
        if path.exists():
            return np.load(path)
        vectors = _embedder().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=float)
        np.save(path, vectors)
        return vectors


def _critic_reviews(movie: dict) -> list[dict]:
    link = movie.get("rotten_tomatoes_link")
    if not link:
        return []
    _movies, reviews = load_tables()
    try:
        rows = reviews.loc[link]
    except KeyError:
        return []
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    out = []
    for _, row in rows.iterrows():
        date = row.get("review_date")
        date_text = "" if pd.isna(date) else str(date)[:10]
        score = row.get("score_norm")
        out.append(
            {
                "author": None if pd.isna(row.get("critic_name")) else str(row.get("critic_name")),
                "content": str(row.get("review_content") or ""),
                "source": "critic",
                "date": date_text,
                "score": None if pd.isna(score) else float(score),
                "fresh": bool(row.get("is_fresh")),
            }
        )
    return out


def _audience_reviews(movie: dict, tmdb=None) -> list[dict]:
    client = tmdb
    tmdb_id = movie.get("tmdb_id")
    if tmdb_id is None and movie.get("source") == "kaggle":
        if client is None:
            from src.tmdb_client import TMDBClient, TMDBError

            client = TMDBClient()
            try:
                hit = client.find_movie(movie["title"], movie.get("year"))
            except TMDBError as exc:
                logger.warning("TMDB audience lookup failed: %s", type(exc).__name__)
                return []
        else:
            hit = client.find_movie(movie["title"], movie.get("year"))
        if hit:
            tmdb_id = hit.get("id")
            movie["tmdb_id"] = tmdb_id
    if tmdb_id is None:
        return []
    if client is None:
        from src.tmdb_client import TMDBClient, TMDBError

        client = TMDBClient()
        try:
            raw = client.get_reviews(tmdb_id, max_pages=3)
        except TMDBError as exc:
            logger.warning("TMDB reviews failed: %s", type(exc).__name__)
            return []
    else:
        raw = client.get_reviews(tmdb_id, max_pages=3)
    out = []
    for item in raw:
        rating = item.get("rating")
        score = None if rating is None else float(rating) / 10.0
        created = item.get("created_at") or ""
        out.append(
            {
                "author": item.get("author"),
                "content": item.get("content") or "",
                "source": "audience",
                "date": str(created)[:10],
                "score": score,
                "fresh": None,
            }
        )
    return out


def get_reviews(movie: dict, tmdb=None, include_tmdb: bool = True) -> list[dict]:
    """Critic blurbs from the parquet, plus TMDB audience reviews when we can find them."""
    if movie.get("ambiguous"):
        return []
    reviews = _critic_reviews(movie)
    if include_tmdb:
        reviews.extend(_audience_reviews(movie, tmdb=tmdb))
    return reviews


def _rank_embed(reviews: list[dict], question: str) -> list[float]:
    texts = [review.get("content") or "" for review in reviews]
    review_vectors = embed_texts(texts)
    query = embed_texts([question])[0]
    return (review_vectors @ query).tolist()


def _rank_bm25(reviews: list[dict], question: str) -> list[float]:
    from rank_bm25 import BM25Okapi

    tokens = [(review.get("content") or "").lower().split() or ["empty"] for review in reviews]
    scores = BM25Okapi(tokens).get_scores(question.lower().split())
    return [float(score) for score in scores]


def movie_stats(movie: dict, reviews: list[dict]) -> dict:
    critic_texts = sum(1 for review in reviews if review.get("source") == "critic")
    audience_texts = sum(1 for review in reviews if review.get("source") == "audience")
    return {
        "title": movie.get("title"),
        "year": movie.get("year"),
        "source": movie.get("source"),
        "tomatometer": movie.get("tomatometer_rating"),
        "audience_score": movie.get("audience_rating"),
        "critic_review_count": movie.get("tomatometer_count") or critic_texts,
        "audience_review_count": movie.get("audience_count") if movie.get("audience_count") is not None else audience_texts,
        "critic_text_count": critic_texts,
        "audience_text_count": audience_texts,
        "gap": movie.get("critic_audience_gap"),
    }


def retrieve(movie: dict, question: str, k: int = 12, retriever: str = "embed", reviews: list[dict] | None = None, tmdb=None) -> dict:
    """Rank one movie's reviews. Keep about half critic and half audience when both exist."""
    if reviews is None:
        reviews = get_reviews(movie, tmdb=tmdb)
    reviews = [dict(review) for review in reviews if (review.get("content") or "").strip()]
    suspicion = flag_suspicious(reviews, embed_fn=embed_texts)
    if not reviews:
        return {"reviews": [], "stats": movie_stats(movie, reviews), "suspicion": suspicion}

    if retriever == "bm25":
        scores = _rank_bm25(reviews, question)
    else:
        scores = _rank_embed(reviews, question)
    for review, score in zip(reviews, scores):
        weight = 0.25 if review.get("suspicious") else 1.0
        review["rank_score"] = float(score) * weight

    def _top(source: str, limit: int) -> list[dict]:
        pool = [review for review in reviews if review.get("source") == source]
        pool.sort(key=lambda review: review["rank_score"], reverse=True)
        return pool[:limit]

    critics = [review for review in reviews if review.get("source") == "critic"]
    audience = [review for review in reviews if review.get("source") == "audience"]
    half = max(k // 2, 1)
    if critics and audience:
        chosen = _top("critic", half) + _top("audience", k - half)
    else:
        chosen = sorted(reviews, key=lambda review: review["rank_score"], reverse=True)[:k]
    chosen = sorted(chosen, key=lambda review: review["rank_score"], reverse=True)
    for index, review in enumerate(chosen, start=1):
        review["n"] = index
    return {"reviews": chosen, "stats": movie_stats(movie, reviews), "suspicion": suspicion}


# Touch the cache dir so embedding downloads have a home. The model itself is cached by the library.
DATA_CACHE.mkdir(parents=True, exist_ok=True)
