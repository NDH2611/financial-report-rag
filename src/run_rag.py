import argparse
import json
import time
from pathlib import Path

from llm import ask_rag, ask_rag_evidence
from retrieval import retrieve
from run_experiments import (
    MAX_RETRIES,
    REQUEST_INTERVAL,
    RETRY_429_WAIT,
    RETRY_503_WAIT,
    build_ordered_results,
    is_successful,
    load_questions,
)

EVAL_DIR = Path("data/evaluation")

CONFIGS = {
    "rag": (ask_rag, EVAL_DIR / "rag.json"),
    "rag_evidence": (ask_rag_evidence, EVAL_DIR / "rag_evidence.json"),
}


def load_results(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {x["id"]: x for x in data if isinstance(x, dict) and "id" in x}
    except (json.JSONDecodeError, OSError):
        return {}


def save_results(path: Path, results: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def call_with_retry(fn, question, contexts):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(question, contexts)
        except Exception as e:
            msg = str(e)

            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if ("GenerateRequestsPerDayPerProjectPerModel-FreeTier" in msg
                        or "requests per day limit exceeded" in msg.lower()):
                    raise RuntimeError("DAILY_QUOTA_EXHAUSTED") from e
                wait = RETRY_429_WAIT
            elif "503" in msg or "UNAVAILABLE" in msg:
                wait = RETRY_503_WAIT
            else:
                raise

            print(f"  Retry {attempt}/{MAX_RETRIES} after {wait}s ({msg[:60]})")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)


def run_config(name: str, k: int, mode: str) -> None:
    fn, out_path = CONFIGS[name]
    questions = load_questions()
    result_map = load_results(out_path)

    print(f"\nRunning {name} | k={k} | retrieval={mode} | {len(questions)} questions")

    for i, item in enumerate(questions, start=1):
        qid, question = item["id"], item["question"]
        print(f"[{i}/{len(questions)}] {qid}: {question}")

        prev = result_map.get(qid)
        if prev and is_successful(prev):
            print("  SKIP | already completed successfully")
            continue

        contexts = retrieve(question, k=k, mode=mode)

        try:
            result = call_with_retry(fn, question, contexts)
            result_map[qid] = {
                "id": qid,
                "category": item["category"],
                **result,
                # Used by evaluate.py -> retrieval_recall
                "retrieved_chunks": [
                    {"chunk_id": c["chunk_id"], "page": c["page"], "score": c["score"]}
                    for c in contexts
                ],
            }
            print(f"  OK | pages={[c['page'] for c in contexts]} "
                  f"| latency={result['latency']:.2f}s")

        except Exception as e:
            if "DAILY_QUOTA_EXHAUSTED" in str(e):
                save_results(out_path, build_ordered_results(questions, result_map))
                print("DAILY QUOTA EXHAUSTED: progress saved, rerun later.")
                return

            print(f"  FAILED: {e}")
            result_map[qid] = {
                "id": qid,
                "category": item["category"],
                "configuration": name,
                "answer": "",
                "calculation": None,
                "evidence": [],
                "retrieved_chunks": [
                    {"chunk_id": c["chunk_id"], "page": c["page"], "score": c["score"]}
                    for c in contexts
                ],
                "error": str(e),
            }

        save_results(out_path, build_ordered_results(questions, result_map))
        time.sleep(REQUEST_INTERVAL)

    final = build_ordered_results(questions, result_map)
    ok = sum(1 for r in final if is_successful(r))
    print(f"\n{name}: {ok}/{len(questions)} successful -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    args = parser.parse_args()

    for cfg in (CONFIGS if args.config == "all" else [args.config]):
        run_config(cfg, args.k, args.mode)