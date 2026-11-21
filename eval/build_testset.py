"""Build a fixed 50-movie question set. Seed is 42 so the file does not drift."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import resolve_movie  # noqa: E402

OUT = ROOT / "eval" / "testset.json"

# Well known releases after the October 2020 scrape.
POST_2020 = [
    ("Everything Everywhere All at Once", 2022),
    ("Top Gun: Maverick", 2022),
    ("The Batman", 2022),
    ("Barbie", 2023),
    ("Oppenheimer", 2023),
    ("Spider-Man: Across the Spider-Verse", 2023),
    ("Poor Things", 2023),
    ("Dune: Part Two", 2024),
    ("Inside Out 2", 2024),
    ("Wicked", 2024),
]

SECOND = ["agree", "family", "spoiler"]


def _year(value):
    if pd.isna(value):
        return None
    return int(value)


def _questions(title: str, year, kind: str) -> list[tuple[str, str]]:
    year_bit = f" ({year})" if year else ""
    verdict = ("verdict", f"Is {title}{year_bit} worth watching?")
    if kind == "agree":
        other = ("agree", f"Do critics and audiences agree on {title}?")
    elif kind == "family":
        other = ("family", f"Is {title} good for a family movie night?")
    else:
        other = ("spoiler", f"What happens at the end of {title}?")
    return [verdict, other]


def _record(group: str, movie: dict, kind: str, index: int) -> list[dict]:
    tomato = movie.get("tomatometer_rating")
    audience = movie.get("audience_rating")
    gap = movie.get("critic_audience_gap")
    if gap is None and tomato is not None and audience is not None:
        gap = tomato - audience
    rows = []
    for kind_name, question in _questions(movie["title"], movie.get("year"), kind):
        rows.append(
            {
                "id": f"{group}-{index}-{kind_name}",
                "group": group,
                "title": movie["title"],
                "year": movie.get("year"),
                "question": question,
                "kind": kind_name,
                "tomatometer": tomato,
                "audience": audience,
                "gap": gap,
                "worth_watching": None if tomato is None else bool(tomato >= 60),
                "audience_worth": None if audience is None else bool(audience >= 60),
                "has_gap": None if gap is None else bool(abs(gap) > 20),
                "source": movie.get("source"),
                "link": movie.get("rotten_tomatoes_link"),
            }
        )
    return rows


def _sample(frame: pd.DataFrame, rng: np.random.Generator, n: int, used: set[str]) -> pd.DataFrame:
    pool = frame[~frame["rotten_tomatoes_link"].isin(used)]
    if len(pool) <= n:
        return pool
    indexes = rng.choice(pool.index.to_numpy(), size=n, replace=False)
    return pool.loc[indexes]


def main() -> None:
    rng = np.random.default_rng(42)
    movies = pd.read_parquet(ROOT / "data" / "processed" / "movies.parquet")
    used: set[str] = set()
    items: list[dict] = []

    popular_pool = movies[movies["tomatometer_count"] >= 200]
    obscure_pool = movies[(movies["tomatometer_count"] < 20) & movies["tomatometer_count"].notna()]
    gap_pool = movies[(movies["critic_audience_gap"].abs() > 20) & (movies["tomatometer_count"] >= 30)]
    title_key = movies["movie_title"].str.strip().str.lower()
    year_n = movies.dropna(subset=["release_year"]).groupby(title_key)["release_year"].nunique()
    repeated_titles = set(year_n[year_n > 1].index)
    repeated_pool = movies[title_key.isin(repeated_titles) & movies["release_year"].notna()]

    groups = {
        "popular": _sample(popular_pool, rng, 10, used),
        "obscure": _sample(obscure_pool, rng, 10, used),
        "gap": _sample(gap_pool, rng, 10, used),
        "repeated": _sample(repeated_pool, rng, 10, used),
    }
    for name, frame in groups.items():
        for index, (_, row) in enumerate(frame.iterrows()):
            used.add(row["rotten_tomatoes_link"])
            movie = {
                "title": row["movie_title"],
                "year": _year(row["release_year"]),
                "tomatometer_rating": None if pd.isna(row["tomatometer_rating"]) else float(row["tomatometer_rating"]),
                "audience_rating": None if pd.isna(row["audience_rating"]) else float(row["audience_rating"]),
                "critic_audience_gap": None if pd.isna(row["critic_audience_gap"]) else float(row["critic_audience_gap"]),
                "source": "kaggle",
                "rotten_tomatoes_link": row["rotten_tomatoes_link"],
            }
            items.extend(_record(name, movie, SECOND[index % 3], index))

    for index, (title, year) in enumerate(POST_2020):
        found = resolve_movie(title, year=year)
        movie = {
            "title": title if found is None else found["title"],
            "year": year if found is None else found.get("year") or year,
            "tomatometer_rating": None if found is None else found.get("tomatometer_rating"),
            "audience_rating": None if found is None else found.get("audience_rating"),
            "critic_audience_gap": None if found is None else found.get("critic_audience_gap"),
            "source": None if found is None else found.get("source"),
            "rotten_tomatoes_link": None if found is None else found.get("rotten_tomatoes_link"),
        }
        # Post-2020 scores come from TMDB's vote average scaled to 0-100 when RT has no tomatometer.
        if movie["tomatometer_rating"] is None and movie["audience_rating"] is not None:
            movie["tomatometer_rating"] = movie["audience_rating"]
        items.extend(_record("post-2020", movie, SECOND[index % 3], index))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, indent=2))
    print(f"wrote {len(items)} questions to {OUT}")
    print(pd.Series([item["group"] for item in items]).value_counts().to_string())


if __name__ == "__main__":
    main()
