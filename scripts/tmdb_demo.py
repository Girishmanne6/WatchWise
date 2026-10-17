"""Look up one movie on TMDB and print a short review sample.

Usage:
    python scripts/tmdb_demo.py "Carrie" 2013

Run it twice. The second run should log cache hits instead of calling TMDB.
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tmdb_client import TMDBClient, TMDBError  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print('Usage: python scripts/tmdb_demo.py "TITLE" [YEAR]', file=sys.stderr)
        sys.exit(1)

    title = sys.argv[1]
    year = int(sys.argv[2]) if len(sys.argv) > 2 else None

    client = TMDBClient()
    try:
        match = client.find_movie(title, year)
    except TMDBError as exc:
        print(f"TMDB lookup failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if match is None:
        print(f"No TMDB match for {title!r}" + (f" ({year})" if year else ""))
        sys.exit(1)

    release = match.get("release_date") or ""
    release_year = release[:4] if len(release) >= 4 else "unknown"
    print(f"title: {match.get('title')}")
    print(f"year: {release_year}")
    print(f"tmdb_id: {match.get('id')}")
    print(f"vote_average: {match.get('vote_average')}")
    print("overview:")
    print(match.get("overview") or "(no overview)")

    try:
        reviews = client.get_reviews(match["id"], max_pages=3)
    except TMDBError as exc:
        print(f"Could not load reviews: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"reviews fetched: {len(reviews)}")
    for index, review in enumerate(reviews[:3], start=1):
        content = (review.get("content") or "").replace("\n", " ")
        print(f"--- review {index}: {review.get('author')} (rating {review.get('rating')})")
        print(content[:300])


if __name__ == "__main__":
    main()
