"""Pre-run a few demo movies so the live demo is not waiting on the network."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assistant import answer  # noqa: E402
from src.llm import available_models  # noqa: E402

DEMOS = [
    ("Joker", 2019),
    ("Late Bloomers", 1997),
    ("Knock Down the House", 2019),
    ("Carrie", 2013),
    ("Barbie", 2023),
    ("The Godfather", 1972),
]


def main() -> None:
    # The live demo stays on gpt-4o-mini. Gemma is only in the comparison.
    models = available_models(["gpt-4o-mini"])
    if not models:
        print("gpt-4o-mini is not configured, so the demo answers were not cached")
        return
    model = "gpt-4o-mini"
    print("warming with", model)
    for title, year in DEMOS:
        question = f"Is {title} ({year}) worth watching?"
        print(title, year, flush=True)
        result = answer(question, title, year=year, model_key=model, use_rag=True)
        print(" ", (result.get("verdict") or "no verdict"), "found" if result["found"] else "missing")


if __name__ == "__main__":
    main()
