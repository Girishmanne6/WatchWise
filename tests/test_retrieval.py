"""Movie matching: exact year, a typo, and a title that is not in the table."""

import pandas as pd

from src.retrieval import resolve_movie


def _movies():
    return pd.DataFrame(
        [
            {
                "movie_title": "Inception",
                "release_year": 2010,
                "rotten_tomatoes_link": "m/inception",
                "tomatometer_rating": 87,
                "audience_rating": 91,
                "tomatometer_count": 360,
                "audience_count": 1000,
                "critic_audience_gap": -4,
                "movie_info": "A thief enters dreams.",
                "genres": "Action",
            },
            {
                "movie_title": "Carrie",
                "release_year": 1976,
                "rotten_tomatoes_link": "m/carrie_1976",
                "tomatometer_rating": 93,
                "audience_rating": 77,
                "tomatometer_count": 70,
                "audience_count": 500,
                "critic_audience_gap": 16,
                "movie_info": "",
                "genres": "Horror",
            },
            {
                "movie_title": "Carrie",
                "release_year": 2013,
                "rotten_tomatoes_link": "m/carrie_2013",
                "tomatometer_rating": 50,
                "audience_rating": 44,
                "tomatometer_count": 180,
                "audience_count": 400,
                "critic_audience_gap": 6,
                "movie_info": "",
                "genres": "Horror",
            },
        ]
    )


class _NoTMDB:
    def find_movie(self, title, year=None):
        return None

    def get_reviews(self, tmdb_id, max_pages=3):
        return []


def test_title_and_year():
    found = resolve_movie("Carrie", year=2013, movies=_movies(), tmdb=_NoTMDB())
    assert found["source"] == "kaggle"
    assert found["year"] == 2013
    assert found["rotten_tomatoes_link"] == "m/carrie_2013"


def test_typo():
    found = resolve_movie("Inceptoin", year=2010, movies=_movies(), tmdb=_NoTMDB())
    assert found is not None
    assert found["title"] == "Inception"
    assert not found["ambiguous"]


def test_not_found():
    found = resolve_movie("Zzxqpl the Unmade", year=1999, movies=_movies(), tmdb=_NoTMDB())
    assert found is None
