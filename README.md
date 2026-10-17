# WatchWise

WatchWise is a movie decision assistant for CMPE 259 (NLP) at San Jose State University, by Girish Chowdary Manne. It is meant to help someone decide if a movie is worth watching by reading real critic and audience reviews, pointing out when critics and audiences disagree, flagging reviews that look manipulated, and keeping spoilers out unless the person asks.

This repo is the first slice of that project: clean the Rotten Tomatoes dump, look at what is actually in it, and add a TMDB client that caches responses so later steps can cover movies from after September 2020.

## Data sources

- Kaggle, [Rotten Tomatoes movies and critic reviews](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) (Stefano Leone). Two CSVs join on `rotten_tomatoes_link`:
  - `rotten_tomatoes_movies.csv` has the movie record, Tomatometer, and audience score.
  - `rotten_tomatoes_critic_reviews.csv` has one row per critic review.
  - The scrape stops at the end of October 2020, so the tables do not cover newer releases.
- [TMDB](https://www.themoviedb.org/) (`https://api.themoviedb.org`) for newer movies and audience reviews. The client calls `GET /3/search/movie`, `GET /3/movie/{id}`, and `GET /3/movie/{id}/reviews`, with `Authorization: Bearer <TMDB_READ_TOKEN>`. TMDB is free for non-commercial use. The old hard cap of 40 requests per 10 seconds was removed in 2019. The current soft cap is about 40 requests per second.

The raw CSVs are not in git. Download them from Kaggle and put both files in `data/raw/`.

## Caching and fallback

- Every TMDB call is stored in SQLite (`data/cache/tmdb.sqlite`) under a key made from the path and the sorted query params.
- If a cached response is younger than `CACHE_TTL_DAYS` (default 7), it is returned and TMDB is not called.
- HTTP 429 uses the `Retry-After` header, then sleeps and retries up to 3 times with backoff.
- If the call fails because of a network error, a 5xx, or a 429 that is still failing after those retries, and any cached copy exists, that stale copy is returned and a warning is logged.
- If the call fails and nothing is cached, the client raises a clear error.
- `find_movie` prefers an exact title match (case-insensitive) with the same release year, because titles repeat. Otherwise it keeps the most popular search hit.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put the TMDB API Read Access Token (the long one, not the short API key) in `.env` as `TMDB_READ_TOKEN`. Place the two Kaggle CSVs in `data/raw/`.

```bash
python scripts/build_clean_data.py
jupyter notebook notebooks/01_cleaning_and_eda.ipynb
pytest
python scripts/tmdb_demo.py "Carrie" 2013
python scripts/tmdb_demo.py "Carrie" 2013   # second run should come from the cache
```

Cleaned tables land in `data/processed/movies.parquet` and `data/processed/reviews.parquet`. `.env` and everything under `data/raw/`, `data/processed/`, and `data/cache/` stay out of git.

## Project status

- [x] Data cleaning and EDA
- [x] TMDB client with cache and stale fallback
- [ ] Retrieval over reviews
- [ ] LLM answer prompt with spoiler control
- [ ] Manipulated review check
- [ ] Evaluation

## Layout

```
src/config.py            environment and data paths
src/score_normalize.py   review_score -> 0 to 1
src/cleaning.py          load, clean, join, save
src/cache.py             SQLite response cache
src/tmdb_client.py       search, details, reviews, best match
scripts/build_clean_data.py
scripts/tmdb_demo.py
notebooks/01_cleaning_and_eda.ipynb
```
