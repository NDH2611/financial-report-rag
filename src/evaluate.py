import argparse, csv, json, math, re, statistics
from pathlib import Path

CONFIGS = {
    "llm_only": "llm_only.json",
    "rag": "rag.json",
    "rag_evidence": "rag_evidence.json",
}

LABELS = {
    "llm_only": "LLM Only",
    "rag": "LLM + RAG",
    "rag_evidence": "LLM + RAG + Evidence",
}

ABSTENTION = [
    "not enough information", "insufficient information",
    "cannot be determined", "cannot determine",
    "not provided", "not disclosed", "not available",
    "not stated", "unable to determine",
    "the document does not", "the 10-k does not"
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(x):
    return re.sub(r"\s+", " ", str(x or "").strip().lower())


def to_float(x):
    try:
        return None if x is None or isinstance(x, bool) else float(x)
    except (TypeError, ValueError):
        return None


def numbers_in(x):
    if isinstance(x, dict):
        out = []
        for v in x.values():
            out += numbers_in(v)
        return out
    if isinstance(x, list):
        out = []
        for v in x:
            out += numbers_in(v)
        return out
    n = to_float(x)
    return [] if n is None else [n]


def extract_numbers(text):
    raw = re.findall(
        r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?:[.,]\d{3})*(?:\.\d+)?",
        normalize(text)
    )
    out = []
    for x in raw:
        if "," in x and "." in x:
            x = x.replace(",", "")
        elif "," in x:
            p = x.split(",")
            x = "".join(p) if len(p) == 2 and len(p[1]) == 3 else ".".join(p)
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def records(data):
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("outputs", "results", "responses", "data"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
        if "answer" in data:
            return [data]
        if data and all(isinstance(v, dict) for v in data.values()):
            result = []
            for key, value in data.items():
                row = dict(value)
                row.setdefault("id", key)
                result.append(row)
            return result
    raise ValueError("Unsupported JSON format")


def by_id(rows):
    result = {}
    for row in rows:
        qid = row.get("id") or row.get("question_id") or row.get("qid")
        if qid is not None:
            result[str(qid)] = row
    return result


def expected_numbers(gt):
    if gt.get("expected_value") is not None:
        n = to_float(gt["expected_value"])
        return [] if n is None else [n]
    return numbers_in(gt.get("correct_value"))


def numerical_accuracy(gt, answer):
    expected = expected_numbers(gt)
    if not expected:
        return None

    actual = extract_numbers(answer)
    if not actual:
        return False

    tolerance = abs(to_float(gt.get("tolerance")) or 0.1)

    if len(expected) == 1:
        return any(math.isclose(a, expected[0], abs_tol=tolerance, rel_tol=0)
                   for a in actual)

    return all(
        any(math.isclose(a, e, abs_tol=tolerance, rel_tol=0) for a in actual)
        for e in expected
    )


def is_abstention(answer):
    text = normalize(answer)
    return any(x in text for x in ABSTENTION)


def answer_accuracy(gt, output):
    answer = normalize(output.get("answer"))
    if not answer:
        return False

    category = normalize(gt.get("category"))

    if category == "unanswerable":
        return is_abstention(answer)

    numerical = numerical_accuracy(gt, answer)
    if numerical is not None:
        return numerical

    expected = set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", normalize(gt.get("ground_truth"))))
    actual = set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", answer))

    stop = {"the", "a", "an", "is", "was", "were", "of", "in", "to",
            "and", "or", "with", "for", "than", "from", "that", "this"}

    expected -= stop
    actual -= stop

    if not expected:
        return None

    return len(expected & actual) / len(expected) >= 0.35


def find_pages(value):
    pages = []

    def visit(x):
        if isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            pages.append(int(x))
        elif isinstance(x, str):
            for p in re.findall(r"\b(?:page|p\.?)\s*[:#]?\s*(\d{1,3})\b", x.lower()):
                pages.append(int(p))

    visit(value)
    return set(pages)


def evidence_accuracy(gt, output):
    expected = find_pages(gt.get("evidence_pages", []))
    if not expected:
        return None

    actual = find_pages(output.get("evidence", []))
    if not actual:
        actual = find_pages(output.get("page"))

    return bool(expected & actual) if actual else False


def retrieval_recall(gt, output):
    # Your current RAG output does not contain retrieved chunks/pages.
    # Therefore this returns None (N/A), rather than inventing a score.
    retrieved = None
    for key in ("retrieved_chunks", "retrieved", "top_k",
                "retrieval", "retrieved_pages"):
        if key in output:
            retrieved = output[key]
            break

    if retrieved is None:
        return None

    expected = find_pages(gt.get("evidence_pages", []))
    actual = find_pages(retrieved)

    if not expected:
        return None

    return bool(expected & actual)


def reasoning_accuracy(gt, output):
    category = normalize(gt.get("category"))
    answer = normalize(output.get("answer"))

    if category == "calculation":
        return numerical_accuracy(gt, answer)

    if category == "trend":
        label = normalize(gt.get("trend_label"))
        if label == "increasing":
            return any(x in answer for x in
                       ["increase", "increasing", "increased", "up", "rise"])
        if label == "decreasing":
            return any(x in answer for x in
                       ["decrease", "decreasing", "decreased", "down", "fall"])
        if label == "mixed":
            return (
                any(x in answer for x in ["fell", "fall", "declined", "decrease"])
                and any(x in answer for x in ["rose", "increase", "increased", "up"])
            )

    if category != "reasoning":
        return None

    expected = set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", normalize(gt.get("ground_truth"))))
    actual = set(re.findall(r"[a-zA-ZÀ-ỹ0-9]+", answer))

    stop = {"the", "a", "an", "is", "was", "were", "of", "in", "to",
            "and", "or", "with", "for", "than", "from", "that", "this"}

    expected -= stop
    actual -= stop

    return None if not expected else len(expected & actual) / len(expected) >= 0.30


def one_question(qid, gt, output):
    return {
        "id": qid,
        "category": gt.get("category", ""),
        "configuration": output.get("configuration", ""),
        "answer": output.get("answer", ""),
        "answer_accuracy": answer_accuracy(gt, output),
        "numerical_accuracy": numerical_accuracy(gt, output.get("answer", "")),
        "evidence_accuracy": evidence_accuracy(gt, output),
        "retrieval_recall": retrieval_recall(gt, output),
        "reasoning_accuracy": reasoning_accuracy(gt, output),
        "trend_accuracy": (
            reasoning_accuracy(gt, output)
            if normalize(gt.get("category")) == "trend" else None
        ),
        "abstention_accuracy": (
            is_abstention(output.get("answer", ""))
            if normalize(gt.get("category")) == "unanswerable" else None
        ),
        "parse_error": (
            None if "parse_error" not in output
            else not (output.get("parse_error") in (None, "", False))
        ),
        "latency": to_float(output.get("latency")),
        "input_tokens": to_float(output.get("input_tokens")),
        "output_tokens": to_float(output.get("output_tokens")),
        "model": output.get("model", "")
    }


def percentage(values):
    values = [x for x in values if x is not None]
    if not values:
        return None
    return round(sum(x is True for x in values) / len(values) * 100, 2)


def average(values):
    values = [x for x in values if x is not None]
    return None if not values else round(statistics.mean(values), 4)


def aggregate(rows, config):
    get = lambda key: [r[key] for r in rows]

    return {
        "configuration": config,
        "configuration_label": LABELS[config],
        "questions_evaluated": len(rows),
        "answer_accuracy_%": percentage(get("answer_accuracy")),
        "numerical_accuracy_%": percentage(get("numerical_accuracy")),
        "evidence_accuracy_%": percentage(get("evidence_accuracy")),
        "retrieval_recall_%": percentage(get("retrieval_recall")),
        "hallucination_rate_%": None,
        "reasoning_accuracy_%": percentage(get("reasoning_accuracy")),
        "trend_accuracy_%": percentage(get("trend_accuracy")),
        "abstention_accuracy_%": percentage(get("abstention_accuracy")),
        "parse_error_rate_%": percentage([
            False if x is None else x for x in get("parse_error")
        ]),
        "avg_latency_seconds": average(get("latency")),
        "avg_input_tokens": average(get("input_tokens")),
        "avg_output_tokens": average(get("output_tokens")),
        "cost": None
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/evaluation/questions.json")
    parser.add_argument("--input-dir", default="data/evaluation")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    questions = load_json(Path(args.questions))
    ground_truth = {
        str(q["id"]): q for q in questions
        if isinstance(q, dict) and q.get("id") is not None
    }

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    summary = []
    all_rows = []

    for config, filename in CONFIGS.items():
        path = input_dir / filename

        if not path.exists():
            print(f"[WARNING] Missing: {path}")
            continue

        try:
            output_rows = by_id(records(load_json(path)))
        except Exception as exc:
            print(f"[ERROR] Cannot read {path}: {exc}")
            continue

        rows = []

        for qid, gt in ground_truth.items():
            if qid not in output_rows:
                print(f"[WARNING] {config}: missing {qid}")
                continue

            row = one_question(qid, gt, output_rows[qid])
            rows.append(row)
            all_rows.append(row)

        write_csv(output_dir / f"{config}_evaluation.csv", rows)
        summary.append(aggregate(rows, config))

    write_csv(output_dir / "comparison.csv", summary)
    write_json(output_dir / "comparison.json", summary)
    write_csv(output_dir / "all_evaluation.csv", all_rows)

    print("\n=== EVALUATION SUMMARY ===")

    for row in summary:
        print(
            f"{row['configuration_label']}: "
            f"Answer={row['answer_accuracy_%']}%, "
            f"Numerical={row['numerical_accuracy_%']}%, "
            f"Evidence={row['evidence_accuracy_%']}%, "
            f"Retrieval={row['retrieval_recall_%']}%, "
            f"Reasoning={row['reasoning_accuracy_%']}%, "
            f"Trend={row['trend_accuracy_%']}%, "
            f"Abstention={row['abstention_accuracy_%']}%, "
            f"Latency={row['avg_latency_seconds']}s"
        )

    print("\nRetrieval Recall = N/A with the current RAG output schema.")
    print("Cost and Hallucination Rate = N/A until the project defines their input data.")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
