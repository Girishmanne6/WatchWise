"""Run every test question through every available model, with and without RAG.

Prints a rough cost, then waits for confirmation unless --yes is passed.
Raw rows are appended to eval/results/raw.jsonl so a stopped run can resume.
"""

import argparse
import json
import logging
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assistant import answer, extract_verdict  # noqa: E402
from src.config import MODELS, TEST_MODELS  # noqa: E402
from src.llm import ModelSkipped, _is_daily_limit, available_models, generate  # noqa: E402
from eval.judge import judge_answer, pick_judge  # noqa: E402

RESULTS = ROOT / "eval" / "results"
RAW = RESULTS / "raw.jsonl"
TESTSET = ROOT / "eval" / "testset.json"

_SPLIT = re.compile(r"critic|audience", re.I)
_REFUSAL = re.compile(
    r"spoiler|won't spoil|will not spoil|not going to spoil|won't reveal|will not reveal|ruin the ending|rather not",
    re.I,
)


def _run_key(item, model, mode, retriever, guard) -> str:
    return f"{item['id']}|{model}|{mode}|{retriever}|{int(guard)}"


def _load_done() -> tuple[list[dict], set[str]]:
    rows = []
    keys = set()
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            keys.add(row["run_key"])
    return rows, keys


def mentions_split(text: str) -> bool:
    lowered = (text or "").lower()
    if "critic" not in lowered or "audience" not in lowered:
        return False
    return any(word in lowered for word in ("disagree", "disagreement", "split", "gap", "differ", "than the", "points"))


def refused_spoiler(text: str) -> bool:
    return bool(_REFUSAL.search(text or ""))


def verdict_correct(predicted: str | None, score) -> bool | None:
    """Mixed counts as correct when the score is between 50 and 70."""
    if score is None or predicted is None:
        return None
    if predicted == "mixed" and 50 <= score <= 70:
        return True
    if predicted == "worth it" and score >= 60:
        return True
    if predicted == "skip" and score < 60:
        return True
    return False


def _configs(models: list[str]) -> list[dict]:
    configs = []
    for model in models:
        configs.append({"model": model, "mode": "no_rag", "retriever": "embed", "guard": True})
        configs.append({"model": model, "mode": "rag", "retriever": "embed", "guard": True})
    primary = "gpt-4o-mini" if "gpt-4o-mini" in models else models[0]
    configs.append({"model": primary, "mode": "rag", "retriever": "bm25", "guard": True})
    configs.append({"model": primary, "mode": "rag", "retriever": "embed", "guard": False})
    return configs


def _estimate(n_questions: int, configs: list[dict]) -> None:
    calls = n_questions * len(configs)
    # Guard adds a second call when it is on. Assume it does.
    guarded = sum(1 for config in configs if config["guard"]) * n_questions
    judge_calls = calls
    # Rough size of a prompt with 12 short reviews, plus a short answer.
    answer_in, answer_out = 900, 220
    judge_in, judge_out = 1100, 120
    print(f"questions: {n_questions}")
    print(f"configs: {len(configs)}")
    print(f"answer calls: {calls} plus about {guarded} spoiler checks")
    print(f"judge calls: {judge_calls}")
    print(
        "rough token guess: "
        f"{(calls + guarded) * answer_in + judge_calls * judge_in:,} input, "
        f"{(calls + guarded) * answer_out + judge_calls * judge_out:,} output"
    )
    print("Cached calls from a previous run are not sent to the API again.")
    for config in configs:
        prices = MODELS[config["model"]]
        unit = (answer_in / 1e6) * prices["input_per_million"] + (answer_out / 1e6) * prices["output_per_million"]
        print(
            f"  {config['model']} {config['mode']} retriever={config['retriever']} "
            f"guard={config['guard']}: about ${unit * n_questions:.2f} before the judge"
        )


def _safe_exc(exc: Exception) -> str:
    text = re.sub(r"(sk-|gsk_|key_|AQ\.)\S+", "[redacted]", str(exc))
    return f"{type(exc).__name__}: {text[:160]}"


def _one(item: dict, config: dict, judge_key: str | None, guard_model: str | None) -> dict:
    result = answer(
        item["question"],
        item["title"],
        year=item.get("year"),
        model_key=config["model"],
        use_rag=config["mode"] == "rag",
        spoiler_guard=config["guard"],
        retriever=config["retriever"],
        guard_model=guard_model,
    )
    row = {
        "run_key": _run_key(item, config["model"], config["mode"], config["retriever"], config["guard"]),
        "id": item["id"],
        "group": item["group"],
        "kind": item["kind"],
        "title": item["title"],
        "year": item.get("year"),
        "question": item["question"],
        "model": config["model"],
        "mode": config["mode"],
        "retriever": config["retriever"],
        "guard": config["guard"],
        "tomatometer": item.get("tomatometer"),
        "has_gap": item.get("has_gap"),
        "text": result["text"],
        "verdict": result.get("verdict") or extract_verdict(result["text"]),
        "found": result["found"],
        "ambiguous": result["ambiguous"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cost_usd": result["cost_usd"],
        "latency_s": result["latency_s"],
        "guard_fired": result["guard"]["fired"],
        "guard_leaked_first": result["guard"]["leaked"],
        "suspicion_score": (result.get("suspicion") or {}).get("score"),
        "retrieved": [
            {"n": review.get("n"), "source": review.get("source"), "content": (review.get("content") or "")[:400]}
            for review in result.get("retrieved") or []
        ],
        "stats": result.get("stats") or {},
    }
    if judge_key and result["found"] and not result["ambiguous"]:
        try:
            row["judge"] = judge_answer(
                item["question"],
                result["text"],
                result.get("retrieved") or [],
                result.get("stats") or {},
                judge_key,
            )
        except Exception as exc:  # noqa: BLE001
            row["judge_error"] = type(exc).__name__
    return row


def _citation_accuracy(judge: dict) -> float | None:
    citations = judge.get("citations") or []
    if not citations:
        return None
    ok = 0
    for item in citations:
        if item.get("ok") is True:
            ok += 1
    return ok / len(citations)


def _metrics(frame: pd.DataFrame) -> dict:
    def _rate(series):
        series = series.dropna()
        if series.empty:
            return None
        return float(series.mean())

    scores = []
    for _, row in frame.iterrows():
        scores.append(verdict_correct(row["verdict"], row["tomatometer"]))
    gap_rows = frame[frame["has_gap"] == True]  # noqa: E712
    spoiler_rows = frame[frame["kind"] == "spoiler"]
    rag_rows = frame[frame["mode"] == "rag"]
    grounded = []
    halluc = []
    leaks = []
    citations = []
    for _, row in frame.iterrows():
        judge = row["judge"] if isinstance(row["judge"], dict) else {}
        if row["mode"] == "rag" and judge.get("groundedness") is not None:
            grounded.append(float(judge["groundedness"]))
        if judge.get("hallucination") is not None:
            halluc.append(bool(judge["hallucination"]))
        if judge.get("spoiler") is not None:
            leaks.append(bool(judge["spoiler"]))
        if row["mode"] == "rag":
            accuracy = _citation_accuracy(judge)
            if accuracy is not None:
                citations.append(accuracy)
    refusal = []
    for _, row in spoiler_rows.iterrows():
        judge = row["judge"] if isinstance(row["judge"], dict) else {}
        leaked = bool(judge.get("spoiler")) if judge else False
        refusal.append(bool(refused_spoiler(row["text"]) and not leaked))
    return {
        "n": int(len(frame)),
        "verdict_accuracy": _rate(pd.Series(scores, dtype=float)),
        "disagreement_recall": _rate(gap_rows["text"].map(mentions_split)) if len(gap_rows) else None,
        "groundedness": _rate(pd.Series(grounded, dtype=float)) if grounded else None,
        "hallucination_rate": _rate(pd.Series(halluc, dtype=float)) if halluc else None,
        "spoiler_leak_rate": _rate(pd.Series(leaks, dtype=float)) if leaks else None,
        "refusal_correct": _rate(pd.Series(refusal, dtype=float)) if refusal else None,
        "citation_accuracy": _rate(pd.Series(citations, dtype=float)) if citations else None,
        "avg_cost_usd": float(frame["cost_usd"].mean()) if len(frame) else None,
        "avg_latency_s": float(frame["latency_s"].mean()) if len(frame) else None,
        "rag_rows": int(len(rag_rows)),
    }


def _label(row) -> str:
    guard = "guard" if row["guard"] else "no-guard"
    return f"{row['model']} | {row['mode']} | {row['retriever']} | {guard}"


def _write_reports(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("no rows to summarize")
        return
    if "judge" not in frame.columns:
        frame["judge"] = [{} for _ in range(len(frame))]
    frame["judge"] = frame["judge"].apply(lambda value: value if isinstance(value, dict) else {})
    summary_rows = []
    for keys, group in frame.groupby(["model", "mode", "retriever", "guard"], dropna=False):
        metrics = _metrics(group)
        metrics.update({"model": keys[0], "mode": keys[1], "retriever": keys[2], "guard": keys[3]})
        summary_rows.append(metrics)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)

    group_rows = []
    for keys, group in frame.groupby(["model", "mode", "retriever", "guard", "group"], dropna=False):
        metrics = _metrics(group)
        metrics.update(
            {"model": keys[0], "mode": keys[1], "retriever": keys[2], "guard": keys[3], "group": keys[4]}
        )
        group_rows.append(metrics)
    by_group = pd.DataFrame(group_rows)
    by_group.to_csv(RESULTS / "summary_by_group.csv", index=False)

    _figures(summary, by_group)
    _examples(frame)
    _manual(frame)
    print(summary.to_string(index=False))
    print(f"wrote {RESULTS / 'summary.csv'}")


def _figures(summary: pd.DataFrame, by_group: pd.DataFrame) -> None:
    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = summary.copy()
    summary["label"] = summary.apply(_label, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot = summary.dropna(subset=["verdict_accuracy"])
    ax.bar(plot["label"], plot["verdict_accuracy"], color="#1f4e79")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Verdict accuracy")
    ax.set_title("Verdict accuracy by config")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures / "accuracy_by_config.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot = summary.dropna(subset=["hallucination_rate"])
    ax.bar(plot["label"], plot["hallucination_rate"], color="#c4554a")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hallucination rate")
    ax.set_title("Hallucination rate by config")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures / "hallucination_by_config.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped = by_group.groupby("group")["hallucination_rate"].mean()
    ax.bar(grouped.index.astype(str), grouped.values, color="#6a4c93")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hallucination rate")
    ax.set_xlabel("Test group")
    ax.set_title("Hallucination rate by test group")
    fig.tight_layout()
    fig.savefig(figures / "hallucination_by_group.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot = summary.dropna(subset=["verdict_accuracy", "avg_cost_usd"])
    ax.scatter(plot["avg_cost_usd"], plot["verdict_accuracy"], s=60, color="#1f4e79")
    for _, row in plot.iterrows():
        ax.annotate(row["model"], (row["avg_cost_usd"], row["verdict_accuracy"]), fontsize=8)
    ax.set_xlabel("Average cost per question (USD)")
    ax.set_ylabel("Verdict accuracy")
    ax.set_title("Cost vs verdict accuracy")
    fig.tight_layout()
    fig.savefig(figures / "cost_vs_accuracy.png", dpi=140)
    plt.close(fig)


def _examples(frame: pd.DataFrame) -> None:
    rag = frame[(frame["mode"] == "rag") & (frame["retriever"] == "embed") & (frame["guard"] == True)]  # noqa: E712
    base = frame[(frame["mode"] == "no_rag") & (frame["guard"] == True)]  # noqa: E712
    merged = rag.merge(base, on=["id", "model"], suffixes=("_rag", "_norag"))
    if merged.empty:
        return
    merged["distance"] = [
        abs(len(a or "") - len(b or "")) + (0 if va == vb else 500)
        for a, b, va, vb in zip(merged["text_rag"], merged["text_norag"], merged["verdict_rag"], merged["verdict_norag"])
    ]
    picked = merged.sort_values("distance", ascending=False).head(4)
    lines = ["# No RAG vs RAG", ""]
    for _, row in picked.iterrows():
        lines.append(f"## {row['title_rag']} ({row['year_rag']}) — {row['model']}")
        lines.append("")
        lines.append(f"Question: {row['question_rag']}")
        lines.append("")
        lines.append("### No RAG")
        lines.append("")
        lines.append(row["text_norag"] or "")
        lines.append("")
        lines.append("### RAG")
        lines.append("")
        lines.append(row["text_rag"] or "")
        lines.append("")
    # The spec asked for no em dashes in the student notebook. This file is an eval note.
    text = "\n".join(lines).replace("—", "-")
    (RESULTS / "examples.md").write_text(text)


def _manual(frame: pd.DataFrame) -> None:
    pool = frame[frame["text"].astype(str).str.len() > 0]
    if pool.empty:
        return
    rng = random.Random(42)
    indexes = list(pool.index)
    rng.shuffle(indexes)
    take = pool.loc[indexes[:20]]
    rows = []
    for _, row in take.iterrows():
        judge = row["judge"] if isinstance(row["judge"], dict) else {}
        rows.append(
            {
                "id": row["id"],
                "model": row["model"],
                "mode": row["mode"],
                "question": row["question"],
                "answer": row["text"],
                "judge_verdict": row["verdict"],
                "judge_groundedness": judge.get("groundedness"),
                "judge_hallucination": judge.get("hallucination"),
                "judge_spoiler": judge.get("spoiler"),
                "human_verdict": "",
                "human_groundedness": "",
                "human_hallucination": "",
                "human_spoiler": "",
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS / "manual_check.csv", index=False)


def _write_raw(rows: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with RAW.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()


def _repair_judges(rows: list[dict], judge_key: str | None) -> list[dict]:
    """Relabel every answer with one judge. Each row is written before the next call."""
    if not judge_key:
        return rows
    for index, row in enumerate(rows, start=1):
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        same = judge.get("_judge") == judge_key and judge.get("groundedness") is not None and not row.get("judge_error")
        if same:
            continue
        if not row.get("found") or row.get("ambiguous"):
            continue
        try:
            row["judge"] = judge_answer(
                row["question"],
                row.get("text") or "",
                row.get("retrieved") or [],
                row.get("stats") or {},
                judge_key,
            )
            row.pop("judge_error", None)
            print(f"rejudged {index}/{len(rows)} {row['model']} {row['id']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            row["judge_error"] = type(exc).__name__
            print(f"rejudge failed {row['id']}: {_safe_exc(exc)}", flush=True)
            _write_raw(rows)
            if _is_daily_limit(exc):
                print("daily token limit. stopped so this judge is not mixed with another.", flush=True)
                break
            continue
        _write_raw(rows)
    return rows


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="run without the confirmation prompt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--rejudge-only",
        action="store_true",
        help="label saved answers with one judge and do not call the test models",
    )
    args = parser.parse_args()
    if args.rejudge_only:
        RESULTS.mkdir(parents=True, exist_ok=True)
        judge_key = pick_judge()
        print("judge:", judge_key or "(none)", flush=True)
        if judge_key is None:
            print("no judge is available")
            sys.exit(1)
        rows, _done = _load_done()
        # Rough size of one rubric call. Printed before any answer is relabeled.
        prices = MODELS[judge_key]
        judge_in, judge_out = 1100, 120
        unit = (judge_in / 1e6) * prices["input_per_million"] + (judge_out / 1e6) * prices["output_per_million"]
        print(f"rows: {len(rows)}")
        print(f"estimated judge cost at list rates: ${unit * len(rows):.2f}")
        print("Cached calls are not sent again. One judge is used for every row.")
        if not args.yes:
            typed = input("Rejudge with this model? [y/N] ").strip().lower()
            if typed != "y":
                print("stopped")
                return
        rows = _repair_judges(rows, judge_key)
        if rows:
            _write_reports(rows)
        return
    if not TESTSET.exists():
        print("missing eval/testset.json. Run eval/build_testset.py first.")
        sys.exit(1)
    items = json.loads(TESTSET.read_text())
    models = []
    for key in available_models(TEST_MODELS):
        try:
            generate(key, "Reply with exactly: ok", "ok", temperature=0)
            models.append(key)
        except Exception as exc:  # noqa: BLE001
            print(f"dropping {key}: {type(exc).__name__}")
    if not models:
        print("no model keys are set, so there is nothing to run")
        sys.exit(1)
    configs = _configs(models)
    print("models:", ", ".join(models))
    print("judge: skipped during answer generation. Run --rejudge-only afterward, with no test-model calls.")
    _estimate(len(items), configs)
    if not args.yes:
        typed = input("Run the eval? [y/N] ").strip().lower()
        if typed != "y":
            print("stopped")
            return

    RESULTS.mkdir(parents=True, exist_ok=True)
    # Load the tables and the embedding model on this thread before workers start.
    from src.retrieval import embed_texts, load_tables

    load_tables()
    embed_texts(["warmup"])
    rows, done = _load_done()
    # Answers are saved without a judge. Judging is python eval/run_eval.py --rejudge-only,
    # so the Groq daily cap is not shared with the test model in the same run.
    total = len(items) * len(configs)
    # The spoiler check is a cheap yes/no. Use the open Groq model when it works.
    guard_model = "gpt-oss-20b" if "gpt-oss-20b" in models else models[0]
    pending = []
    finished = 0
    for config in configs:
        for item in items:
            finished += 1
            key = _run_key(item, config["model"], config["mode"], config["retriever"], config["guard"])
            if key in done:
                continue
            pending.append((finished, item, config))
    print(f"todo: {len(pending)} of {total}, workers: {args.workers}, guard model: {guard_model}", flush=True)

    def _work(job):
        index, item, config = job
        row = _one(item, config, None, guard_model)
        return index, item, config, row

    try:
        with RAW.open("a") as handle:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(_work, job) for job in pending]
                for future in as_completed(futures):
                    try:
                        index, item, config, row = future.result()
                    except ModelSkipped as exc:
                        print(_safe_exc(exc), flush=True)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        print(f"failed: {_safe_exc(exc)}", flush=True)
                        continue
                    print(
                        f"[{index}/{total}] {config['model']} {config['mode']} {config['retriever']} {item['id']}",
                        flush=True,
                    )
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    rows.append(row)
    finally:
        saved, _keys = _load_done()
        if saved:
            _write_reports(saved)


if __name__ == "__main__":
    main()
