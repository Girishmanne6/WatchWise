"""Small Streamlit demo for WatchWise."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.assistant import answer
from src.config import TEST_MODELS
from src.llm import available_models

DEMOS = [
    ("Joker", 2019, "lots of reviews"),
    ("Late Bloomers", 1997, "fewer than 20 critic reviews"),
    ("Knock Down the House", 2019, "critics liked it much more than audiences"),
    ("Carrie", 2013, "the title is used for more than one movie"),
    ("Barbie", 2023, "after the 2020 dataset, so this comes from TMDB"),
    ("The Godfather", 1972, "a known title, useful for the typo check"),
]


def _score_bar(label: str, value, count) -> None:
    if value is None:
        st.write(f"{label}: not in the tables")
        return
    st.write(f"{label}: {value:.0f}  ({count if count is not None else 'n/a'} ratings)")
    st.progress(min(max(float(value) / 100.0, 0.0), 1.0))


def main() -> None:
    st.set_page_config(page_title="WatchWise", page_icon="🎬", layout="centered")
    st.title("WatchWise")
    st.caption("A movie decision assistant for CMPE 259. It reads critic and audience reviews before it answers.")

    models = available_models(TEST_MODELS)
    with st.sidebar:
        st.header("Demo movies")
        st.write("These are the ones the warm-up script caches.")
        for title, year, note in DEMOS:
            if st.button(f"{title} ({year})", key=f"demo-{title}-{year}"):
                st.session_state["title"] = title
                st.session_state["year"] = str(year)
                st.session_state["question"] = f"Is {title} ({year}) worth watching?"
            st.caption(note)

    if not models:
        st.warning("No model keys are set. Add one to .env and restart.")
        models = ["gpt-4o-mini"]

    title = st.text_input("Title", value=st.session_state.get("title", "The Godfather"))
    year_text = st.text_input("Year (optional)", value=st.session_state.get("year", "1972"))
    question = st.text_input(
        "Question",
        value=st.session_state.get("question", "Is The Godfather (1972) worth watching?"),
    )
    model_key = st.selectbox("Model", models)
    use_rag = st.toggle("Use reviews (RAG)", value=True)
    allow_spoilers = st.checkbox("Allow spoilers", value=False)

    if st.button("Ask", type="primary"):
        year = int(year_text) if year_text.strip().isdigit() else None
        with st.spinner("Reading reviews..."):
            result = answer(
                question,
                title,
                year=year,
                model_key=model_key,
                use_rag=use_rag,
                spoiler_guard=True,
                allow_spoilers=allow_spoilers,
            )
        verdict = result.get("verdict")
        if verdict:
            st.subheader(f"Verdict: {verdict}")
        st.write(result["text"])
        stats = result.get("stats") or {}
        if stats:
            left, right = st.columns(2)
            with left:
                _score_bar("Tomatometer", stats.get("tomatometer"), stats.get("critic_review_count"))
            with right:
                _score_bar("Audience", stats.get("audience_score"), stats.get("audience_review_count"))
            if stats.get("gap") is not None:
                st.caption(f"Critic minus audience: {stats['gap']:.0f}")
        suspicion = result.get("suspicion") or {}
        if suspicion.get("score"):
            reasons = ", ".join(suspicion.get("reasons") or [])
            st.warning(f"Some reviews look suspicious ({suspicion['score']:.0%}). {reasons}")
        reviews = result.get("retrieved") or []
        if reviews:
            with st.expander(f"Reviews used ({len(reviews)})"):
                for review in reviews:
                    tag = review.get("source")
                    flag = " · flagged" if review.get("suspicious") else ""
                    st.markdown(f"**[{review.get('n')}] {tag}**{flag} · {review.get('author') or 'unknown'}")
                    st.write(review.get("content") or "")


if __name__ == "__main__":
    main()
