import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ============================================================
# Configuration
# ============================================================

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
    "not enough information",
    "insufficient information",
    "information is insufficient",
    "insufficient to answer",
    "cannot be determined",
    "cannot determine",
    "not provided",
    "not disclosed",
    "not available",
    "not stated",
    "unable to determine",
    "the document does not",
    "the 10-k does not",
    "does not provide",
    "doesn't provide",
    "not enough reliable information",
]

# IMPORTANT:
# Keep this model fixed for the entire hallucination evaluation.
#
# Your current quota error showed gemini-3.6-flash.
# Therefore use the same judge model consistently for all judgments.
JUDGE_MODEL = "gemini-3.6-flash"

# We have a request/day bottleneck, not primarily a token bottleneck.
# Batch 10 substantially reduces request count while remaining manageable.
JUDGE_BATCH_SIZE = 10

DEFAULT_JUDGMENT_CACHE = Path(
    "data/evaluation/hallucination_judgments.json"
)

VALID_JUDGMENTS = {
    "supported",
    "hallucinated",
    "uncertain",
}


# ============================================================
# Generic utilities
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def normalize(x):
    return re.sub(
        r"\s+",
        " ",
        str(x or "").strip().lower(),
    )


def to_float(x):
    try:
        if x is None or isinstance(x, bool):
            return None

        return float(x)

    except (TypeError, ValueError):
        return None


def numbers_in(x):
    if isinstance(x, dict):
        out = []

        for value in x.values():
            out += numbers_in(value)

        return out

    if isinstance(x, list):
        out = []

        for value in x:
            out += numbers_in(value)

        return out

    number = to_float(x)

    if number is None:
        return []

    return [number]


def extract_numbers(text):
    raw = re.findall(
        r"(?<![\w])[-+]?\d+(?:[.,]\d+)?"
        r"(?:[.,]\d{3})*(?:\.\d+)?",
        normalize(text),
    )

    out = []

    for value in raw:
        if "," in value and "." in value:
            value = value.replace(",", "")

        elif "," in value:
            parts = value.split(",")

            if (
                len(parts) == 2
                and len(parts[1]) == 3
            ):
                value = "".join(parts)
            else:
                value = ".".join(parts)

        try:
            out.append(float(value))

        except ValueError:
            pass

    return out


def records(data):
    if isinstance(data, list):
        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    if isinstance(data, dict):
        for key in (
            "outputs",
            "results",
            "responses",
            "data",
        ):
            if isinstance(data.get(key), list):
                return [
                    item
                    for item in data[key]
                    if isinstance(item, dict)
                ]

        if "answer" in data:
            return [data]

        if (
            data
            and all(
                isinstance(value, dict)
                for value in data.values()
            )
        ):
            result = []

            for key, value in data.items():
                row = dict(value)
                row.setdefault("id", key)
                result.append(row)

            return result

    raise ValueError(
        "Unsupported JSON format"
    )


def by_id(rows):
    result = {}

    for row in rows:
        qid = (
            row.get("id")
            or row.get("question_id")
            or row.get("qid")
        )

        if qid is not None:
            result[str(qid)] = row

    return result


# ============================================================
# Existing evaluation metrics
# ============================================================

def expected_numbers(gt):
    if gt.get("expected_value") is not None:
        number = to_float(
            gt["expected_value"]
        )

        if number is None:
            return []

        return [number]

    return numbers_in(
        gt.get("correct_value")
    )


def numerical_accuracy(gt, answer):
    expected = expected_numbers(gt)

    if not expected:
        return None

    actual = extract_numbers(answer)

    if not actual:
        return False

    tolerance = abs(
        to_float(
            gt.get("tolerance")
        )
        or 0.1
    )

    if len(expected) == 1:
        return any(
            math.isclose(
                actual_value,
                expected[0],
                abs_tol=tolerance,
                rel_tol=0,
            )
            for actual_value in actual
        )

    return all(
        any(
            math.isclose(
                actual_value,
                expected_value,
                abs_tol=tolerance,
                rel_tol=0,
            )
            for actual_value in actual
        )
        for expected_value in expected
    )


def is_abstention(answer):
    text = normalize(answer)

    return any(
        phrase in text
        for phrase in ABSTENTION
    )


def answer_accuracy(gt, output):
    answer = normalize(
        output.get("answer")
    )

    if not answer:
        return False

    category = normalize(
        gt.get("category")
    )

    if category == "unanswerable":
        return is_abstention(answer)

    numerical = numerical_accuracy(
        gt,
        answer,
    )

    if numerical is not None:
        return numerical

    expected = set(
        re.findall(
            r"[a-zA-ZÀ-ỹ0-9]+",
            normalize(
                gt.get("ground_truth")
            ),
        )
    )

    actual = set(
        re.findall(
            r"[a-zA-ZÀ-ỹ0-9]+",
            answer,
        )
    )

    stop = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "of",
        "in",
        "to",
        "and",
        "or",
        "with",
        "for",
        "than",
        "from",
        "that",
        "this",
    }

    expected -= stop
    actual -= stop

    if not expected:
        return None

    overlap = (
        len(expected & actual)
        / len(expected)
    )

    return overlap >= 0.35


def find_pages(value):
    pages = []

    def visit(x):
        if isinstance(x, dict):
            for value in x.values():
                visit(value)

        elif isinstance(x, list):
            for value in x:
                visit(value)

        elif (
            isinstance(x, (int, float))
            and not isinstance(x, bool)
        ):
            pages.append(int(x))

        elif isinstance(x, str):
            matches = re.findall(
                r"\b(?:page|p\.?)"
                r"\s*[:#]?\s*(\d{1,3})\b",
                x.lower(),
            )

            for page in matches:
                pages.append(int(page))

    visit(value)

    return set(pages)


def evidence_accuracy(gt, output):
    expected = find_pages(
        gt.get(
            "evidence_pages",
            [],
        )
    )

    if not expected:
        return None

    actual = find_pages(
        output.get(
            "evidence",
            [],
        )
    )

    if not actual:
        actual = find_pages(
            output.get("page")
        )

    if not actual:
        return False

    return bool(
        expected & actual
    )


def retrieval_recall(gt, output):
    retrieved = None

    for key in (
        "retrieved_chunks",
        "retrieved",
        "top_k",
        "retrieval",
        "retrieved_pages",
    ):
        if key in output:
            retrieved = output[key]
            break

    if retrieved is None:
        return None

    expected = find_pages(
        gt.get(
            "evidence_pages",
            [],
        )
    )

    actual = find_pages(
        retrieved
    )

    if not expected:
        return None

    return bool(
        expected & actual
    )


def reasoning_accuracy(gt, output):
    category = normalize(
        gt.get("category")
    )

    answer = normalize(
        output.get("answer")
    )

    if category == "calculation":
        return numerical_accuracy(
            gt,
            answer,
        )

    if category == "trend":
        label = normalize(
            gt.get("trend_label")
        )

        if label == "increasing":
            return any(
                word in answer
                for word in (
                    "increase",
                    "increasing",
                    "increased",
                    "up",
                    "rise",
                )
            )

        if label == "decreasing":
            return any(
                word in answer
                for word in (
                    "decrease",
                    "decreasing",
                    "decreased",
                    "down",
                    "fall",
                )
            )

        if label == "mixed":
            down = any(
                word in answer
                for word in (
                    "fell",
                    "fall",
                    "declined",
                    "decrease",
                )
            )

            up = any(
                word in answer
                for word in (
                    "rose",
                    "increase",
                    "increased",
                    "up",
                )
            )

            return down and up

    if category != "reasoning":
        return None

    expected = set(
        re.findall(
            r"[a-zA-ZÀ-ỹ0-9]+",
            normalize(
                gt.get("ground_truth")
            ),
        )
    )

    actual = set(
        re.findall(
            r"[a-zA-ZÀ-ỹ0-9]+",
            answer,
        )
    )

    stop = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "of",
        "in",
        "to",
        "and",
        "or",
        "with",
        "for",
        "than",
        "from",
        "that",
        "this",
    }

    expected -= stop
    actual -= stop

    if not expected:
        return None

    overlap = (
        len(expected & actual)
        / len(expected)
    )

    return overlap >= 0.30


# ============================================================
# Compact hallucination reference
# ============================================================

def compact_value(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = re.sub(
            r"\s+",
            " ",
            value.strip(),
        )

        return value or None

    if isinstance(
        value,
        (int, float, bool),
    ):
        return str(value)

    if isinstance(value, list):
        items = []

        for item in value:
            text = compact_value(item)

            if text:
                items.append(text)

        if not items:
            return None

        return "; ".join(items)

    if isinstance(value, dict):
        items = []

        for key, item in value.items():
            text = compact_value(item)

            if text:
                items.append(
                    f"{key}={text}"
                )

        if not items:
            return None

        return "; ".join(items)

    return str(value)


def build_reference(gt):
    category = normalize(
        gt.get("category")
    )

    lines = [
        f"Category: {category}"
    ]

    ground_truth = compact_value(
        gt.get("ground_truth")
    )

    if ground_truth:
        lines.append(
            f"Ground truth: {ground_truth}"
        )

    correct_value = compact_value(
        gt.get("correct_value")
    )

    if correct_value:
        lines.append(
            f"Correct value: {correct_value}"
        )

    unit = compact_value(
        gt.get("unit")
    )

    if unit:
        lines.append(
            f"Unit: {unit}"
        )

    expected_value = compact_value(
        gt.get("expected_value")
    )

    if expected_value:
        lines.append(
            f"Expected value: {expected_value}"
        )

    tolerance = compact_value(
        gt.get("tolerance")
    )

    if tolerance:
        lines.append(
            f"Tolerance: {tolerance}"
        )

    expected_calculation = compact_value(
        gt.get("expected_calculation")
    )

    if expected_calculation:
        lines.append(
            "Expected calculation: "
            f"{expected_calculation}"
        )

    key_points = compact_value(
        gt.get("key_points")
    )

    if key_points:
        lines.append(
            f"Key points: {key_points}"
        )

    reasoning = compact_value(
        gt.get("reasoning")
    )

    if reasoning:
        lines.append(
            f"Reference reasoning: {reasoning}"
        )

    trend_label = compact_value(
        gt.get("trend_label")
    )

    if trend_label:
        lines.append(
            f"Trend: {trend_label}"
        )

    values = compact_value(
        gt.get("values")
    )

    if values:
        lines.append(
            f"Reference values: {values}"
        )

    evidence_pages = compact_value(
        gt.get("evidence_pages")
    )

    if evidence_pages:
        lines.append(
            f"Evidence pages: {evidence_pages}"
        )

    if category == "unanswerable":
        lines.append(
            "Reference status: "
            "The question is unanswerable from "
            "the evaluated financial report."
        )

    return "\n".join(lines)


# ============================================================
# Free deterministic hallucination checks
# ============================================================

def deterministic_hallucination_judgment(
    gt,
    output,
):
    answer = str(
        output.get(
            "answer",
            "",
        )
        or ""
    ).strip()

    category = normalize(
        gt.get("category")
    )

    # Empty response has no factual claim.
    # It may be inaccurate/useless, but it is not hallucination.
    if not answer:
        return {
            "status": "supported",
            "hallucinated": False,
            "reason": (
                "Empty response; no factual claim "
                "was produced."
            ),
            "method": "deterministic",
        }

    # Correct abstention on an intentionally unanswerable
    # question is not hallucination.
    if (
        category == "unanswerable"
        and is_abstention(answer)
    ):
        return {
            "status": "supported",
            "hallucinated": False,
            "reason": (
                "Correct abstention on an "
                "unanswerable question."
            ),
            "method": "deterministic",
        }

    return None


# ============================================================
# Hallucination cache
# ============================================================

def judgment_key(
    configuration,
    qid,
):
    return f"{configuration}:{qid}"


def load_judgment_cache(path):
    if not path.exists():
        return {}

    try:
        data = load_json(path)

    except Exception as exc:
        print(
            "[WARNING] Cannot read hallucination "
            f"cache {path}: {exc}"
        )

        return {}

    if isinstance(data, dict):
        return data

    return {}


def save_judgment_cache(
    path,
    cache,
):
    write_json(
        path,
        cache,
    )


# ============================================================
# Gemini hallucination judge
# ============================================================

def get_judge_client():
    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env"
        )

    return genai.Client(
        api_key=api_key
    )


def build_judge_prompt(items):
    sections = []

    for item in items:
        sections.append(
            "\n".join([
                f"KEY: {item['key']}",
                "QUESTION:",
                item["question"],
                "REFERENCE:",
                item["reference"],
                "MODEL ANSWER:",
                item["answer"],
            ])
        )

    joined = "\n\n---\n\n".join(
        sections
    )

    return f"""
You are evaluating hallucination in a financial
question-answering benchmark.

Judge each item ONLY against its provided REFERENCE.
Do not use outside knowledge.

Definition:
A response is HALLUCINATED if it contains at least one
factual claim that is contradicted by the REFERENCE or
is not supported by the REFERENCE.

Rules:
1. Correct factual paraphrases are SUPPORTED.
2. Equivalent financial units are allowed.
3. A calculation using supported values is not hallucinated
   merely because it is expressed differently.
4. An incorrect factual or numerical claim is HALLUCINATED.
5. A factual claim that goes beyond what the REFERENCE
   supports is HALLUCINATED.
6. Missing information or an incomplete answer is NOT
   automatically hallucination.
7. An abstention is NOT hallucination merely because the
   benchmark question was answerable.
8. Do not judge style, verbosity, formatting, or usefulness.
9. If the REFERENCE is genuinely insufficient to determine
   whether a factual claim is supported, use UNCERTAIN.
10. Do not guess.
11. Evaluate every item independently.

Return ONLY valid JSON.
Do not use Markdown fences.

Return exactly a JSON array with one object per input item:

[
  {{
    "key": "configuration:question_id",
    "status": "supported",
    "reason": "short reason"
  }}
]

status must be exactly one of:
- supported
- hallucinated
- uncertain

Keep each reason to one short sentence.

ITEMS:

{joined}
""".strip()


def parse_judge_response(
    text,
    expected_keys,
):
    text = str(text or "").strip()

    # Defensive cleanup in case the model unexpectedly
    # returns Markdown JSON fences.
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    try:
        data = json.loads(text)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Judge returned invalid JSON: "
            f"{exc}\nRaw response:\n{text}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            "Judge response must be a JSON array."
        )

    result = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        key = str(
            item.get(
                "key",
                "",
            )
        ).strip()

        status = normalize(
            item.get("status")
        )

        reason = str(
            item.get(
                "reason",
                "",
            )
            or ""
        ).strip()

        if key not in expected_keys:
            continue

        if status not in VALID_JUDGMENTS:
            status = "uncertain"

        result[key] = {
            "status": status,
            "hallucinated": (
                True
                if status == "hallucinated"
                else False
                if status == "supported"
                else None
            ),
            "reason": reason,
            "method": "llm_judge",
            "judge_model": JUDGE_MODEL,
        }

    return result


def is_daily_quota_error(exc):
    """
    Detect quota errors that cannot be fixed by waiting
    15-60 seconds.

    We intentionally stop immediately so that:
    - no time is wasted on useless retries
    - successful cached judgments remain available
    - the next run can resume after quota reset
    """

    text = str(exc).lower()

    if (
        "429" not in text
        and "resource_exhausted" not in text
    ):
        return False

    daily_markers = (
        "requests per day",
        "perday",
        "per day limit",
        "generate_content_free_tier_requests",
        "generaterequestsperday",
    )

    return any(
        marker in text
        for marker in daily_markers
    )


def call_judge(
    client,
    items,
    max_retries=3,
):
    prompt = build_judge_prompt(
        items
    )

    expected_keys = {
        item["key"]
        for item in items
    }

    for attempt in range(
        1,
        max_retries + 1,
    ):
        try:
            start = time.perf_counter()

            response = (
                client.models.generate_content(
                    model=JUDGE_MODEL,
                    contents=prompt,
                )
            )

            latency = (
                time.perf_counter()
                - start
            )

            parsed = parse_judge_response(
                response.text,
                expected_keys,
            )

            usage = getattr(
                response,
                "usage_metadata",
                None,
            )

            input_tokens = (
                getattr(
                    usage,
                    "prompt_token_count",
                    None,
                )
                if usage
                else None
            )

            output_tokens = (
                getattr(
                    usage,
                    "candidates_token_count",
                    None,
                )
                if usage
                else None
            )

            return {
                "judgments": parsed,
                "latency": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        except Exception as exc:
            # ------------------------------------------------
            # IMPORTANT:
            # Daily quota exhaustion cannot recover after a
            # short retry. Stop immediately.
            # ------------------------------------------------
            if is_daily_quota_error(exc):
                print()
                print(
                    "  Daily Gemini quota exhausted."
                )
                print(
                    "  Successful previous batches "
                    "are already cached."
                )
                print(
                    "  No short retry will be attempted."
                )
                print(
                    "  Resume after quota reset by "
                    "running the same command."
                )

                raise RuntimeError(
                    "Gemini daily request quota exhausted. "
                    "Resume later; cached judgments "
                    "will be skipped."
                ) from exc

            if attempt >= max_retries:
                raise

            wait_seconds = min(
                10 * attempt,
                30,
            )

            print(
                f"  Temporary judge error "
                f"(attempt {attempt}/"
                f"{max_retries}): {exc}"
            )

            print(
                f"  Retrying in "
                f"{wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "Judge failed unexpectedly."
    )


# ============================================================
# Prepare hallucination judgments
# ============================================================

def prepare_hallucination_items(
    ground_truth,
    outputs_by_config,
    cache,
):
    pending = []

    deterministic_count = 0
    cached_count = 0

    for config, output_rows in (
        outputs_by_config.items()
    ):
        for qid, gt in (
            ground_truth.items()
        ):
            if qid not in output_rows:
                continue

            key = judgment_key(
                config,
                qid,
            )

            if key in cache:
                cached_count += 1
                continue

            output = output_rows[qid]

            deterministic = (
                deterministic_hallucination_judgment(
                    gt,
                    output,
                )
            )

            if deterministic is not None:
                cache[key] = {
                    "key": key,
                    "configuration": config,
                    "id": qid,
                    **deterministic,
                }

                deterministic_count += 1
                continue

            answer = str(
                output.get(
                    "answer",
                    "",
                )
                or ""
            ).strip()

            question = str(
                gt.get(
                    "question",
                    "",
                )
                or ""
            ).strip()

            pending.append({
                "key": key,
                "configuration": config,
                "id": qid,
                "question": question,
                "reference": build_reference(
                    gt
                ),
                "answer": answer,
            })

    return (
        pending,
        deterministic_count,
        cached_count,
    )


def run_hallucination_judge(
    ground_truth,
    outputs_by_config,
    cache_path,
    batch_size,
):
    cache = load_judgment_cache(
        cache_path
    )

    (
        pending,
        deterministic_count,
        cached_count,
    ) = prepare_hallucination_items(
        ground_truth,
        outputs_by_config,
        cache,
    )

    # Save free deterministic judgments before any API call.
    save_judgment_cache(
        cache_path,
        cache,
    )

    print()
    print(
        "=== HALLUCINATION JUDGE ==="
    )

    print(
        f"Judge model            : "
        f"{JUDGE_MODEL}"
    )

    print(
        f"Batch size             : "
        f"{batch_size}"
    )

    print(
        f"Cached judgments       : "
        f"{cached_count}"
    )

    print(
        f"New deterministic      : "
        f"{deterministic_count}"
    )

    print(
        f"Pending LLM judgments  : "
        f"{len(pending)}"
    )

    print(
        f"Total currently cached : "
        f"{len(cache)}"
    )

    if not pending:
        print()
        print(
            "All hallucination judgments "
            "are already available."
        )

        return cache

    client = get_judge_client()

    total_batches = math.ceil(
        len(pending)
        / batch_size
    )

    print(
        f"API requests required  : "
        f"{total_batches}"
    )

    total_input_tokens = 0
    total_output_tokens = 0
    completed_this_run = 0

    for batch_number, start in enumerate(
        range(
            0,
            len(pending),
            batch_size,
        ),
        start=1,
    ):
        batch = pending[
            start:start + batch_size
        ]

        print()
        print(
            f"[Judge batch "
            f"{batch_number}/"
            f"{total_batches}]"
        )

        print(
            "  "
            + ", ".join(
                item["key"]
                for item in batch
            )
        )

        try:
            result = call_judge(
                client,
                batch,
            )

        except RuntimeError as exc:
            # Cache from all successful earlier batches has
            # already been written to disk.
            print()
            print(
                "=== JUDGE STOPPED SAFELY ==="
            )

            print(
                f"Reason: {exc}"
            )

            print(
                f"Completed this run: "
                f"{completed_this_run}"
            )

            print(
                f"Cached total      : "
                f"{len(cache)}"
            )

            print(
                f"Still pending     : "
                f"{len(pending) - completed_this_run}"
            )

            print(
                f"Cache preserved at: "
                f"{cache_path}"
            )

            print()
            print(
                "After quota reset, run the "
                "same command again."
            )

            return cache

        judgments = result[
            "judgments"
        ]

        batch_saved = 0

        for item in batch:
            key = item["key"]

            judgment = judgments.get(
                key
            )

            # If the judge omitted an item, do NOT pretend
            # it was supported/hallucinated. Mark uncertain.
            if judgment is None:
                judgment = {
                    "status": "uncertain",
                    "hallucinated": None,
                    "reason": (
                        "Judge omitted this item "
                        "from the batch response."
                    ),
                    "method": "llm_judge",
                    "judge_model": JUDGE_MODEL,
                }

            cache[key] = {
                "key": key,
                "configuration": (
                    item["configuration"]
                ),
                "id": item["id"],
                **judgment,
            }

            batch_saved += 1

        # CRITICAL:
        # Persist immediately after every successful batch.
        save_judgment_cache(
            cache_path,
            cache,
        )

        completed_this_run += (
            batch_saved
        )

        input_tokens = (
            result["input_tokens"]
            or 0
        )

        output_tokens = (
            result["output_tokens"]
            or 0
        )

        total_input_tokens += (
            input_tokens
        )

        total_output_tokens += (
            output_tokens
        )

        print(
            f"  Saved judgments: "
            f"{batch_saved}"
        )

        print(
            f"  Latency        : "
            f"{result['latency']:.3f}s"
        )

        print(
            f"  Input tokens   : "
            f"{input_tokens}"
        )

        print(
            f"  Output tokens  : "
            f"{output_tokens}"
        )

        print(
            f"  Cached total   : "
            f"{len(cache)}"
        )

    print()
    print(
        "=== HALLUCINATION JUDGE COMPLETE ==="
    )

    print(
        f"New LLM judgments: "
        f"{completed_this_run}"
    )

    print(
        f"Total cached     : "
        f"{len(cache)}"
    )

    print(
        f"Input tokens     : "
        f"{total_input_tokens}"
    )

    print(
        f"Output tokens    : "
        f"{total_output_tokens}"
    )

    print(
        f"Total tokens     : "
        f"{total_input_tokens + total_output_tokens}"
    )

    return cache


# ============================================================
# Hallucination metric
# ============================================================

def get_hallucination_judgment(
    cache,
    config,
    qid,
):
    key = judgment_key(
        config,
        qid,
    )

    judgment = cache.get(
        key
    )

    if not isinstance(
        judgment,
        dict,
    ):
        return None, None, None

    return (
        judgment.get(
            "hallucinated"
        ),
        judgment.get(
            "status"
        ),
        judgment.get(
            "reason"
        ),
    )


def hallucination_rate(rows):
    """
    HR =
        HALLUCINATED
        -----------------------------
        SUPPORTED + HALLUCINATED
        x 100

    UNCERTAIN and missing judgments are excluded from
    the denominator and reported separately.
    """

    resolved = [
        row["hallucinated"]
        for row in rows
        if row.get("hallucinated")
        is not None
    ]

    if not resolved:
        return None

    hallucinated = sum(
        value is True
        for value in resolved
    )

    return round(
        hallucinated
        / len(resolved)
        * 100,
        2,
    )


def hallucination_counts(rows):
    supported = 0
    hallucinated = 0
    uncertain = 0
    missing = 0

    for row in rows:
        status = normalize(
            row.get(
                "hallucination_status"
            )
        )

        if status == "supported":
            supported += 1

        elif status == "hallucinated":
            hallucinated += 1

        elif status == "uncertain":
            uncertain += 1

        else:
            missing += 1

    return {
        "supported": supported,
        "hallucinated": hallucinated,
        "uncertain": uncertain,
        "missing": missing,
    }


# ============================================================
# Per-question evaluation
# ============================================================

def one_question(
    qid,
    gt,
    output,
    config,
    hallucination_cache,
):
    (
        hallucinated,
        hallucination_status,
        hallucination_reason,
    ) = get_hallucination_judgment(
        hallucination_cache,
        config,
        qid,
    )

    return {
        "id": qid,
        "category": gt.get(
            "category",
            "",
        ),
        "configuration": output.get(
            "configuration",
            "",
        ),
        "answer": output.get(
            "answer",
            "",
        ),
        "answer_accuracy": answer_accuracy(
            gt,
            output,
        ),
        "numerical_accuracy": numerical_accuracy(
            gt,
            output.get(
                "answer",
                "",
            ),
        ),
        "evidence_accuracy": evidence_accuracy(
            gt,
            output,
        ),
        "retrieval_recall": retrieval_recall(
            gt,
            output,
        ),
        "hallucinated": hallucinated,
        "hallucination_status": (
            hallucination_status
        ),
        "hallucination_reason": (
            hallucination_reason
        ),
        "reasoning_accuracy": reasoning_accuracy(
            gt,
            output,
        ),
        "trend_accuracy": (
            reasoning_accuracy(
                gt,
                output,
            )
            if normalize(
                gt.get("category")
            ) == "trend"
            else None
        ),
        "abstention_accuracy": (
            is_abstention(
                output.get(
                    "answer",
                    "",
                )
            )
            if normalize(
                gt.get("category")
            ) == "unanswerable"
            else None
        ),
        "parse_error": (
            None
            if "parse_error"
            not in output
            else not (
                output.get(
                    "parse_error"
                )
                in (
                    None,
                    "",
                    False,
                )
            )
        ),
        "latency": to_float(
            output.get("latency")
        ),
        "input_tokens": to_float(
            output.get(
                "input_tokens"
            )
        ),
        "output_tokens": to_float(
            output.get(
                "output_tokens"
            )
        ),
        "model": output.get(
            "model",
            "",
        ),
    }


# ============================================================
# Aggregation
# ============================================================

def percentage(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return round(
        sum(
            value is True
            for value in values
        )
        / len(values)
        * 100,
        2,
    )


def average(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return round(
        statistics.mean(values),
        4,
    )


def aggregate(
    rows,
    config,
):
    get = lambda key: [
        row[key]
        for row in rows
    ]

    counts = hallucination_counts(
        rows
    )

    resolved = (
        counts["supported"]
        + counts["hallucinated"]
    )

    return {
        "configuration": config,
        "configuration_label": (
            LABELS[config]
        ),
        "questions_evaluated": len(
            rows
        ),
        "answer_accuracy_%": percentage(
            get("answer_accuracy")
        ),
        "numerical_accuracy_%": percentage(
            get("numerical_accuracy")
        ),
        "evidence_accuracy_%": percentage(
            get("evidence_accuracy")
        ),
        "retrieval_recall_%": percentage(
            get("retrieval_recall")
        ),
        "hallucination_rate_%": (
            hallucination_rate(
                rows
            )
        ),
        "hallucination_resolved": (
            resolved
        ),
        "hallucination_supported": (
            counts["supported"]
        ),
        "hallucination_flagged": (
            counts["hallucinated"]
        ),
        "hallucination_uncertain": (
            counts["uncertain"]
        ),
        "hallucination_missing": (
            counts["missing"]
        ),
        "reasoning_accuracy_%": percentage(
            get("reasoning_accuracy")
        ),
        "trend_accuracy_%": percentage(
            get("trend_accuracy")
        ),
        "abstention_accuracy_%": percentage(
            get("abstention_accuracy")
        ),
        "parse_error_rate_%": percentage([
            False
            if value is None
            else value
            for value in get(
                "parse_error"
            )
        ]),
        "avg_latency_seconds": average(
            get("latency")
        ),
        "avg_input_tokens": average(
            get("input_tokens")
        ),
        "avg_output_tokens": average(
            get("output_tokens")
        ),
        "cost": None,
    }


# ============================================================
# Load experiment outputs
# ============================================================

def load_all_outputs(
    input_dir,
):
    outputs_by_config = {}

    for config, filename in (
        CONFIGS.items()
    ):
        path = (
            input_dir
            / filename
        )

        if not path.exists():
            print(
                f"[WARNING] Missing: {path}"
            )

            continue

        try:
            outputs_by_config[
                config
            ] = by_id(
                records(
                    load_json(path)
                )
            )

        except Exception as exc:
            print(
                "[ERROR] Cannot read "
                f"{path}: {exc}"
            )

    return outputs_by_config


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions",
        default=(
            "data/evaluation/"
            "questions.json"
        ),
    )

    parser.add_argument(
        "--input-dir",
        default="data/evaluation",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
    )

    parser.add_argument(
        "--hallucination-cache",
        default=str(
            DEFAULT_JUDGMENT_CACHE
        ),
    )

    parser.add_argument(
        "--judge-hallucination",
        action="store_true",
        help=(
            "Run the compact Gemini hallucination "
            "judge only for uncached responses."
        ),
    )

    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=JUDGE_BATCH_SIZE,
    )

    args = parser.parse_args()

    questions = load_json(
        Path(args.questions)
    )

    ground_truth = {
        str(question["id"]): question
        for question in questions
        if (
            isinstance(
                question,
                dict,
            )
            and question.get("id")
            is not None
        )
    }

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    cache_path = Path(
        args.hallucination_cache
    )

    outputs_by_config = (
        load_all_outputs(
            input_dir
        )
    )

    # --------------------------------------------------------
    # Hallucination judging
    # --------------------------------------------------------

    if args.judge_hallucination:
        hallucination_cache = (
            run_hallucination_judge(
                ground_truth,
                outputs_by_config,
                cache_path,
                max(
                    1,
                    args.judge_batch_size,
                ),
            )
        )

    else:
        hallucination_cache = (
            load_judgment_cache(
                cache_path
            )
        )

        # Add any deterministic judgments that are still
        # missing without calling Gemini.
        (
            _pending,
            _deterministic_count,
            _cached_count,
        ) = prepare_hallucination_items(
            ground_truth,
            outputs_by_config,
            hallucination_cache,
        )

        save_judgment_cache(
            cache_path,
            hallucination_cache,
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    summary = []
    all_rows = []

    for config, output_rows in (
        outputs_by_config.items()
    ):
        rows = []

        for qid, gt in (
            ground_truth.items()
        ):
            if qid not in output_rows:
                print(
                    f"[WARNING] "
                    f"{config}: "
                    f"missing {qid}"
                )

                continue

            row = one_question(
                qid,
                gt,
                output_rows[qid],
                config,
                hallucination_cache,
            )

            rows.append(row)
            all_rows.append(row)

        write_csv(
            output_dir
            / f"{config}_evaluation.csv",
            rows,
        )

        summary.append(
            aggregate(
                rows,
                config,
            )
        )

    write_csv(
        output_dir
        / "comparison.csv",
        summary,
    )

    write_json(
        output_dir
        / "comparison.json",
        summary,
    )

    write_csv(
        output_dir
        / "all_evaluation.csv",
        all_rows,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print(
        "=== EVALUATION SUMMARY ==="
    )

    for row in summary:
        print()

        print(
            row[
                "configuration_label"
            ]
        )

        print(
            "  Answer Accuracy      : "
            f"{row['answer_accuracy_%']}%"
        )

        print(
            "  Numerical Accuracy   : "
            f"{row['numerical_accuracy_%']}%"
        )

        print(
            "  Evidence Accuracy    : "
            f"{row['evidence_accuracy_%']}%"
        )

        print(
            "  Retrieval Recall     : "
            f"{row['retrieval_recall_%']}%"
        )

        print(
            "  Hallucination Rate   : "
            f"{row['hallucination_rate_%']}%"
        )

        print(
            "  Hallucination audit  : "
            f"{row['hallucination_resolved']}"
            f"/"
            f"{row['questions_evaluated']} "
            "resolved"
        )

        print(
            "    Supported          : "
            f"{row['hallucination_supported']}"
        )

        print(
            "    Hallucinated       : "
            f"{row['hallucination_flagged']}"
        )

        print(
            "    Uncertain          : "
            f"{row['hallucination_uncertain']}"
        )

        print(
            "    Missing            : "
            f"{row['hallucination_missing']}"
        )

        print(
            "  Reasoning Accuracy   : "
            f"{row['reasoning_accuracy_%']}%"
        )

        print(
            "  Trend Accuracy       : "
            f"{row['trend_accuracy_%']}%"
        )

        print(
            "  Abstention Accuracy  : "
            f"{row['abstention_accuracy_%']}%"
        )

        print(
            "  Avg latency          : "
            f"{row['avg_latency_seconds']}s"
        )

    print()
    print(
        "Hallucination methodology:"
    )

    print(
        "  Automated compact LLM-as-a-judge "
        "over frozen responses and frozen "
        "evaluation references."
    )

    print(
        f"  Judge model: {JUDGE_MODEL}"
    )

    print(
        "  UNCERTAIN judgments are excluded "
        "from the Hallucination Rate "
        "denominator."
    )

    print(
        "  Cached judgments are reused; "
        "A/B/C generation is never rerun."
    )

    print()

    print(
        f"Hallucination cache: "
        f"{cache_path}"
    )

    print(
        f"Results saved in: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()