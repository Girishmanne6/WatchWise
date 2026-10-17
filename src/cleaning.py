"""Load the Kaggle Rotten Tomatoes CSVs and write cleaned tables."""

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW
from src.score_normalize import normalize_score

MOVIES_FILE = "rotten_tomatoes_movies.csv"
REVIEWS_FILE = "rotten_tomatoes_critic_reviews.csv"


def _count(label: str, movies: int | None = None, reviews: int | None = None) -> None:
    bits = [label]
    if movies is not None:
        bits.append(f"movies={movies:,}")
    if reviews is not None:
        bits.append(f"reviews={reviews:,}")
    print(" | ".join(bits))


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two raw CSVs. They join on ``rotten_tomatoes_link``."""
    movies_path = DATA_RAW / MOVIES_FILE
    reviews_path = DATA_RAW / REVIEWS_FILE
    movies = pd.read_csv(movies_path, low_memory=False)
    reviews = pd.read_csv(reviews_path, low_memory=False)
    _count("load_raw", len(movies), len(reviews))
    return movies, reviews


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Drop untitled rows, parse the release date, and remove exact duplicates."""
    before = len(df)
    _count("clean_movies before", movies=before)

    cleaned = df.dropna(subset=["movie_title"]).copy()
    cleaned["movie_title"] = cleaned["movie_title"].astype(str).str.strip()
    cleaned = cleaned[cleaned["movie_title"] != ""]
    _count("clean_movies after dropping missing titles", movies=len(cleaned))

    cleaned["original_release_date"] = pd.to_datetime(
        cleaned["original_release_date"], errors="coerce"
    )
    cleaned["release_year"] = cleaned["original_release_date"].dt.year.astype("Int64")

    cleaned = cleaned.drop_duplicates()
    _count("clean_movies after", movies=len(cleaned))
    print(f"clean_movies removed {before - len(cleaned):,} rows")
    return cleaned.reset_index(drop=True)


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Keep reviews that have text, normalize scores, and flag fresh/rotten."""
    before = len(df)
    _count("clean_reviews before", reviews=before)

    cleaned = df.dropna(subset=["review_content"]).copy()
    cleaned["review_content"] = cleaned["review_content"].astype(str).str.strip()
    cleaned = cleaned[cleaned["review_content"] != ""]
    cleaned = cleaned[~cleaned["review_content"].str.lower().isin({"nan", "none", "null"})]
    _count("clean_reviews after dropping empty text", reviews=len(cleaned))

    cleaned = cleaned.drop_duplicates(
        subset=["rotten_tomatoes_link", "critic_name", "review_content"]
    )
    _count("clean_reviews after dropping duplicate reviews", reviews=len(cleaned))

    cleaned["review_date"] = pd.to_datetime(cleaned["review_date"], errors="coerce")
    cleaned["score_norm"] = cleaned["review_score"].map(normalize_score)
    review_type = cleaned["review_type"].astype(str).str.strip().str.lower()
    cleaned["is_fresh"] = review_type.eq("fresh")

    _count("clean_reviews after", reviews=len(cleaned))
    print(f"clean_reviews removed {before - len(cleaned):,} rows")
    return cleaned.reset_index(drop=True)


def join_and_filter(
    movies: pd.DataFrame, reviews: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join on the Rotten Tomatoes link and keep movies that have a review.

    Adds ``critic_audience_gap`` (tomatometer minus audience rating) on movies.
    Positive means critics scored it higher than the audience.
    """
    _count("join_and_filter before", len(movies), len(reviews))

    movies = movies.copy()
    reviews = reviews.copy()
    movies["critic_audience_gap"] = movies["tomatometer_rating"] - movies["audience_rating"]

    shared = set(movies["rotten_tomatoes_link"]).intersection(reviews["rotten_tomatoes_link"])
    movies = movies[movies["rotten_tomatoes_link"].isin(shared)]
    reviews = reviews[reviews["rotten_tomatoes_link"].isin(shared)]

    _count("join_and_filter after", len(movies), len(reviews))
    print(f"movies with at least 1 review: {movies['rotten_tomatoes_link'].nunique():,}")
    return movies.reset_index(drop=True), reviews.reset_index(drop=True)


def save_processed(movies: pd.DataFrame, reviews: pd.DataFrame) -> None:
    """Write cleaned movies and reviews to ``data/processed`` as parquet."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    movies_path = DATA_PROCESSED / "movies.parquet"
    reviews_path = DATA_PROCESSED / "reviews.parquet"
    movies.to_parquet(movies_path, index=False)
    reviews.to_parquet(reviews_path, index=False)
    print(f"saved {len(movies):,} movies -> {movies_path}")
    print(f"saved {len(reviews):,} reviews -> {reviews_path}")
