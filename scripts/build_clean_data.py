"""Build the cleaned movie and review parquet files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleaning import (  # noqa: E402
    clean_movies,
    clean_reviews,
    join_and_filter,
    load_raw,
    save_processed,
)


def main() -> None:
    movies, reviews = load_raw()
    movies = clean_movies(movies)
    reviews = clean_reviews(reviews)
    movies, reviews = join_and_filter(movies, reviews)
    save_processed(movies, reviews)


if __name__ == "__main__":
    main()
