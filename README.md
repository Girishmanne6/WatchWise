# WatchWise

WatchWise is a movie decision assistant for CMPE 259 (NLP) at San Jose State University, by Girish Chowdary Manne. It helps someone decide if a movie is worth watching by reading critic and audience reviews, pointing out when the two sides disagree, flagging reviews that look manipulated, and keeping spoilers out unless the person asks.

Cleaned data is in `data/processed/movies.parquet` and `reviews.parquet`: 17,691 movies and 951,712 critic reviews. The Rotten Tomatoes scrape ends in October 2020.

## Architecture

```mermaid
flowchart LR
    Q[Question plus title] --> R[resolve_movie]
    R --> K[Kaggle parquet]
    R --> T[TMDB with SQLite cache]
    K --> Rev[reviews for that movie]
    T --> Rev
    Rev --> F[flag_suspicious]
    F --> Rank[embed or BM25]
    Rank --> P[prompt with numbered reviews]
    P --> LLM[generate]
    LLM --> G[spoiler check]
    G --> A[answer]
```

Retrieval is per movie. A review is always about one title, and a title has at most 574 critic reviews (median 23), so there is no vector index over all 950k reviews. `resolve_movie` looks up the title and year in the Kaggle table, with a fuzzy match for typos. If the movie is missing or the year is after 2020, it falls back to TMDB. `retrieve` then ranks only that movie's reviews. Embeddings are `sentence-transformers` `all-MiniLM-L6-v2`, cached on disk. BM25 is the keyword alternative (`retriever="bm25"`).

`flag_suspicious` looks for near-duplicate reviews from different authors (cosine above 0.9), a burst of reviews on one day, and very short extreme blurbs. Flagged reviews are down-weighted in the ranking and the prompt tells the model to mention them.

`answer` has two modes. RAG sends the stats and the numbered reviews. No-RAG sends only the title and year, which is the baseline. A second short call asks whether the draft reveals plot points. If it does, and the user did not ask for spoilers, the answer is rewritten once. Reviews are treated as data, not as instructions. A piracy question is refused in code before any model call. An unknown title is reported as not found. A title that maps to several years asks which year.

Every model call goes through `generate` in `src/llm.py`. Responses are cached in SQLite (`data/cache/llm.sqlite`) under a hash of the model id, temperature, system prompt, and user prompt, so a rerun does not pay for the same call again. Missing API keys skip that model. Rate limits and transient 5xx errors retry with backoff.

## Models that ran

Prices in `src/config.py` are list rates in USD per 1M tokens, not the invoice. The Gemini key is on the free tier, so those calls were not billed. The cost column is still tokens times the list rate.

| key | provider | model id | why |
| --- | --- | --- | --- |
| gemini-3.5-flash-lite | Google | gemini-3.5-flash-lite | Flash itself was over quota; this is the model that answered |
| gpt-4o-mini | OpenAI | gpt-4o-mini | in the comparison, and the only demo model |
| gemma-4-26b-a4b-it | Google | gemma-4-26b-a4b-it | open weights. `gemma-4-31b-it` returned 503, so this is the fallback |
| gpt-oss-20b | Groq | openai/gpt-oss-20b | open model this Groq key can call |
| gpt-oss-120b | Groq | openai/gpt-oss-120b | the only judge if gpt-4o is unavailable |

Gemma 4 has no paid rate on the Gemini Developer API (checked 20 Nov 2026). Its cost column is $0. Google lists the calls as free, and prompts may be used to improve their products.

Skipped:

- `gpt-4o` is the judge only. It is not one of the systems being compared.
- `claude-haiku` and `claude-sonnet`: no Anthropic key in `.env`.
- `llama-3.1-8b` (`llama-3.1-8b-instant`) and `llama-3.3-70b`: this Groq account returns 404.
- `gemma-3-27b-it` is not on this key. `gemma-4-31b-it` answered one probe, then returned 503 high demand.
- `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, and `gemini-3.5-flash`: hard quota errors on this key. `gemini-3.5-flash-lite` answered. List price is $0.30 / $2.50 per 1M tokens. It rejects a thinking config, so that setting is left off.

The spoiler check uses the same model as the answer. BM25 and the guard-off row are `gpt-4o-mini`. The judge is `gpt-4o`. The demo and its cache stay on `gpt-4o-mini`.

## Results

Copied from `eval/results/summary.csv` after `python eval/run_eval.py`. Every number below comes from that run.

<!-- RESULTS -->
Judge for every finished answer is `gpt-4o`. Hallucination was graded again with the movie's year, director, cast, and scores. Groundedness on every RAG row was scored again after the judge example stopped using `0.0`. The new scores are real values from 0 to 1. That pass cost $0.97 and did not regenerate answers. The n next to a metric is how many questions went into it. Disagreement recall uses every row with a real critic/audience gap (24 in a full set of 100, 12 in the Gemma subset). Spoiler leak is judged on every answer. Refusal is only the spoiler-bait questions.

Gemma is a fixed 50-question subset, the first 10 questions of each group. No-RAG is 50 of 50. RAG is 48 of 50: `repeated-4-family` and `post-2020-4-family` failed after Gemini returned 503 and 504. Both saved rows use the reference-fact hallucination pass. The long average times include those retry waits. The spoiler prompt tells the model to say it will not spoil an ending question. Those 15 questions were rerun for RAG embeddings with the guard on. The old rows are v1 and the new rows are v2. Refusal correctness went from 0.000 to 0.867 for gpt-4o-mini and from 0.667 to 1.000 for gemini-3.5-flash-lite. The main table uses v2.

| model | mode | n | verdict acc (n) | disagreement (n) | groundedness (n) | hallucination (n) | spoiler leak (n) | refusal (n) | citations (n) | avg cost | avg sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.5-flash-lite | no RAG | 100 | 0.670 (100) | 0.042 (24) |  | 0.010 (100) | 0.000 (100) | 0.467 (15) |  | 0.000204 | 6.68 |
| gemini-3.5-flash-lite | RAG embed, guard on | 100 | 0.780 (100) | 1.000 (24) | 0.888 (100) | 0.010 (100) | 0.000 (100) | 1.000 (15) | 0.993 (100) | 0.000676 | 9.48 |
| gemma-4-26b-a4b-it | no RAG, 50-question subset | 50 | 0.600 (50) | 0.000 (12) |  | 0.000 (50) | 0.000 (50) | 0.200 (5) |  | 0.000000 | 45.01 |
| gemma-4-26b-a4b-it | RAG embed, guard on | 48 | 0.688 (48) | 1.000 (12) | 0.902 (48) | 0.042 (48) | 0.000 (48) | 1.000 (5) | 1.000 (48) | 0.000000 | 68.46 |
| gpt-4o-mini | no RAG | 100 | 0.650 (100) | 0.000 (24) |  | 0.000 (100) | 0.000 (100) | 0.000 (15) |  | 0.000052 | 5.45 |
| gpt-4o-mini | RAG embed, guard on | 100 | 0.660 (100) | 0.667 (24) | 0.907 (100) | 0.030 (100) | 0.010 (100) | 0.867 (15) | 0.980 (100) | 0.000247 | 5.77 |
| gpt-4o-mini | RAG embed, guard off | 100 | 0.600 (100) | 0.583 (24) | 0.830 (100) | 0.030 (100) | 0.020 (100) | 0.000 (15) | 0.945 (91) | 0.000214 | 3.52 |
| gpt-4o-mini | RAG BM25, guard on | 100 | 0.600 (100) | 0.667 (24) | 0.831 (100) | 0.010 (100) | 0.020 (100) | 0.067 (15) | 0.975 (89) | 0.000240 | 6.98 |

Verdict accuracy by group. The number in parentheses is n.

| model | mode | popular | obscure | gap | repeated | post-2020 |
| --- | --- | --- | --- | --- | --- | --- |
| gemini-3.5-flash-lite | no RAG | 0.850 (20) | 0.450 (20) | 0.300 (20) | 0.800 (20) | 0.950 (20) |
| gemini-3.5-flash-lite | RAG embed | 0.850 (20) | 0.650 (20) | 0.450 (20) | 0.950 (20) | 1.000 (20) |
| gemma-4-26b-a4b-it | no RAG | 1.000 (10) | 0.400 (10) | 0.200 (10) | 0.500 (10) | 0.900 (10) |
| gemma-4-26b-a4b-it | RAG embed | 1.000 (10) | 0.400 (10) | 0.200 (10) | 0.889 (9) | 1.000 (9) |
| gpt-4o-mini | no RAG | 0.850 (20) | 0.550 (20) | 0.250 (20) | 0.700 (20) | 0.900 (20) |
| gpt-4o-mini | RAG embed | 0.850 (20) | 0.600 (20) | 0.600 (20) | 0.850 (20) | 0.400 (20) |
| gpt-4o-mini | RAG BM25 | 0.700 (20) | 0.550 (20) | 0.550 (20) | 0.700 (20) | 0.500 (20) |

Hallucination rate by group. Every saved row, including Gemma, uses the reference-fact rejudge.

| model | mode | popular | obscure | gap | repeated | post-2020 |
| --- | --- | --- | --- | --- | --- | --- |
| gemini-3.5-flash-lite | no RAG | 0.00 (20) | 0.05 (20) | 0.00 (20) | 0.00 (20) | 0.00 (20) |
| gemini-3.5-flash-lite | RAG embed | 0.00 (20) | 0.00 (20) | 0.00 (20) | 0.00 (20) | 0.05 (20) |
| gemma-4-26b-a4b-it | no RAG | 0.00 (10) | 0.00 (10) | 0.00 (10) | 0.00 (10) | 0.00 (10) |
| gemma-4-26b-a4b-it | RAG embed | 0.00 (10) | 0.00 (10) | 0.00 (10) | 0.00 (9) | 0.22 (9) |
| gpt-4o-mini | no RAG | 0.00 (20) | 0.00 (20) | 0.00 (20) | 0.00 (20) | 0.00 (20) |
| gpt-4o-mini | RAG embed | 0.00 (20) | 0.00 (20) | 0.05 (20) | 0.00 (20) | 0.10 (20) |
| gpt-4o-mini | RAG BM25 | 0.00 (20) | 0.00 (20) | 0.00 (20) | 0.00 (20) | 0.05 (20) |

Slide charts, white background, are in `eval/results/figures/`: `rag_vs_norag.png`, `accuracy_by_group_rag.png`, `bm25_vs_embed.png`, and `spoiler_refusal_v1_v2.png`. The refusal chart compares v1 and v2.

Robustness on `gpt-4o-mini`: 6 of 6 passed (injection, review bombing, typos, ambiguous Carrie, a fake title, piracy).

### Appendix: gpt-oss-20b

Not in the main table or the charts. RAG is 31 of 100 (popular 20, obscure 11). Groundedness below is the rescored 0–1 value. From `eval/results/summary_gpt_oss.csv`.

| model | mode | n | verdict acc (n) | disagreement (n) | groundedness (n) | hallucination (n) | spoiler leak (n) | refusal (n) | citations (n) | avg cost | avg sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-20b | no RAG | 100 | 0.450 (100) | 0.000 (24) |  | 0.010 (100) | 0.010 (100) | 0.067 (15) |  | 0.000109 | 6.65 |
| gpt-oss-20b | RAG embed | 31 | 0.742 (31) | 0.000 (2) | 0.771 (31) | 0.032 (31) | 0.000 (31) | 0.250 (4) | 0.852 (27) | 0.000169 | 6.87 |
<!-- /RESULTS -->

Ground truth for a verdict is the Tomatometer: worth watching means 60 or higher. A "mixed" verdict counts as correct when the score is between 50 and 70. Post-2020 movies are not in the Rotten Tomatoes dump. Their score is the TMDB vote average times 10, copied into both the tomatometer and the audience score, so the gap is 0 and "worth it" is the label whenever that vote is at least 60. That is a proxy, not a Tomatometer.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in whichever keys you have: TMDB_READ_TOKEN, OPENAI_API_KEY,
# ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY
python scripts/build_clean_data.py   # if data/processed is empty
pytest
python eval/build_testset.py
python eval/run_eval.py              # prints a cost guess, then asks
python eval/robustness.py
python scripts/warm_demo_cache.py
streamlit run app.py
```

`eval/run_eval.py --yes` skips the prompt. It appends to `eval/results/raw.jsonl` and skips rows that are already there, so a stopped run can continue. Reports land in `eval/results/summary.csv`, `summary_by_group.csv`, `figures/`, `examples.md`, and `manual_check.csv`.

`eval/manual_check.csv` has 20 answers, the judge's labels, and empty columns for a human label. After those columns are filled in:

```bash
python eval/agreement.py
```

## Design choices

- Filter by movie, then rank. A global index would mix reviews of different films, and no movie here is large enough to need one.
- Keep a balance of critic and audience blurbs (about half of k each) so a lopsided pool does not hide the disagreement.
- Down-weight suspicious reviews instead of deleting them, and say so in the answer. A bomb that is thrown out silently is harder to explain.
- No-RAG is the same task with no reviews and no scores. That is the hallucination baseline, especially on obscure titles and on post-2020 titles the model may not know.
- The judge is one model for every row: `gpt-4o` if that key works, otherwise Groq `gpt-oss-120b`. It is not one of the models being compared. Temperature is 0 and the reply has to be JSON. Judging is a separate command from answer generation, because the Groq judge and the Groq test model share one daily token cap.
- Ambiguous titles such as Carrie, with no year, ask which one. The code does not pick a year on its own.

## Data and the TMDB cache

- Kaggle, [Rotten Tomatoes movies and critic reviews](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) (Stefano Leone). The two CSVs join on `rotten_tomatoes_link`. The raw files are not in git.
- TMDB (`https://api.themoviedb.org`) covers newer movies and audience review text. Calls are cached in `data/cache/tmdb.sqlite`. A response younger than `CACHE_TTL_DAYS` (default 7) is reused. A 429 uses `Retry-After` and then backoff. A network error, a 5xx, or an exhausted 429 falls back to a stale cached copy when one exists.
- `find_movie` prefers an exact title (case-insensitive) with the same release year. Otherwise it keeps the most popular hit.

## Tests

```bash
pytest
```

Covers score parsing, the SQLite cache, `resolve_movie` (exact year, a typo, and a miss), `flag_suspicious` on injected duplicates, and the verdict extractor.

## Limitations

- The critic table stops in October 2020. Anything newer depends on TMDB, which often has few written audience reviews for small movies.
- Critic "reviews" are short blurbs (median about 21 words), so retrieval is matching a sentence, not an essay.
- Post-2020 ground truth is a TMDB vote, not a Tomatometer, and the two scores are set equal so the gap metric does not apply to that group.
- The LLM judge can be wrong, and it was not checked against a full human sample. `manual_check.csv` is there for that.
- The test set is 50 movies and 100 questions. One seed, one split.
- Flash itself was over quota, so that row is `gemini-3.5-flash-lite`. The open-weights row is `gemma-4-26b-a4b-it` on 50 questions (10 from each group), because `gemma-4-31b-it` returned 503 and the 26B model was too slow to finish all 200 questions within an hour. Claude and Llama 3.1 never ran.
- List-price cost is not what a free-tier key was charged.

## Future work

- Finish the Gemma 50-question RAG rows, finish the remaining gpt-oss-20b RAG rows when the Groq cap resets, and rerun Claude when an Anthropic key is available.
- A human pass on `manual_check.csv`, then `eval/agreement.py`.
- A real audience-review source for small movies, since TMDB pages are often empty.
- A longer spoiler policy than one regenerate, and a check that does not use the same family as the answer model.

## Layout

```
src/config.py            paths, model ids, prices
src/score_normalize.py   review_score to 0-1
src/cleaning.py          load, clean, join, save
src/cache.py             SQLite response cache
src/tmdb_client.py       search, details, reviews
src/llm.py               generate(), response cache, retries
src/retrieval.py         resolve, embed, BM25, retrieve
src/manipulation.py      flag_suspicious
src/assistant.py         answer(), spoiler guard
eval/build_testset.py    100 questions, seed 42
eval/run_eval.py         model x RAG comparison
eval/judge.py            JSON rubric
eval/robustness.py       injection, bombing, typos, fakes
eval/agreement.py        human vs judge, once the CSV is filled
app.py                   Streamlit demo
scripts/warm_demo_cache.py
notebooks/01_cleaning_and_eda.ipynb
```

## Project status

- [x] Data cleaning and EDA
- [x] TMDB client with cache and stale fallback
- [x] Retrieval over one movie's reviews
- [x] LLM answer prompt with spoiler control
- [x] Manipulated review check
- [x] Evaluation, robustness checks, and demo
