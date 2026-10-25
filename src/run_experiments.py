import argparse
import json
import time
from pathlib import Path

from llm import ask_llm, ask_rag, ask_rag_evidence


# ============================================================
# Configuration
# ============================================================

QUESTIONS_PATH = Path("data/evaluation/questions.json")
RETRIEVAL_PATH = Path("data/evaluation/retrieval_results.json")

OUTPUT_PATHS = {
    "a": Path("data/evaluation/llm_only.json"),
    "b": Path("data/evaluation/rag.json"),
    "c": Path("data/evaluation/rag_evidence.json"),
}

CONFIGURATION_NAMES = {
    "a": "LLM Only",
    "b": "LLM + RAG",
    "c": "LLM + RAG + Evidence",
}

# Gemini free-tier safety settings
REQUEST_INTERVAL = 15
MAX_RETRIES = 5

# Temporary per-minute quota
RETRY_429_WAIT = 65

# Temporary service overload
RETRY_503_WAIT = 20


# ============================================================
# File helpers
# ============================================================

def load_json(path: Path):
    """
    Load a JSON file.
    """

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions():
    """
    Load evaluation questions.
    """

    data = load_json(QUESTIONS_PATH)

    if not isinstance(data, list):
        raise ValueError(
            f"{QUESTIONS_PATH} must contain a JSON list."
        )

    return data


def load_retrieval_results():
    """
    Load frozen retrieval results used by Configurations B and C.
    """

    data = load_json(RETRIEVAL_PATH)

    if not isinstance(data, list):
        raise ValueError(
            f"{RETRIEVAL_PATH} must contain a JSON list."
        )

    retrieval_map = {}

    for item in data:
        question_id = item.get("id")

        if not question_id:
            raise ValueError(
                "A retrieval result is missing question id."
            )

        if question_id in retrieval_map:
            raise ValueError(
                f"Duplicate retrieval result for {question_id}."
            )

        contexts = item.get("retrieved_chunks")

        if not isinstance(contexts, list):
            raise ValueError(
                f"{question_id} does not contain "
                f"a valid retrieved_chunks list."
            )

        retrieval_map[question_id] = item

    return retrieval_map


def load_existing_results(output_path: Path):
    """
    Load previous experiment results.

    This allows an experiment to resume without calling Gemini
    again for questions that already succeeded.
    """

    if not output_path.exists():
        return []

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except (json.JSONDecodeError, OSError):
        pass

    return []


def save_results(results, output_path: Path):
    """
    Save current experiment results.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Result helpers
# ============================================================

def is_successful(result):
    """
    A result is successful when it does not contain an
    API/infrastructure error.
    """

    return not result.get("error")


def build_ordered_results(questions, result_map):
    """
    Return results in the same order as questions.json.
    """

    return [
        result_map[q["id"]]
        for q in questions
        if q["id"] in result_map
    ]


# ============================================================
# Validation
# ============================================================

def validate_questions(questions):
    """
    Validate the minimum fields required by the experiment.
    """

    seen_ids = set()

    for item in questions:
        question_id = item.get("id")
        question = item.get("question")
        category = item.get("category")

        if not question_id:
            raise ValueError(
                "A question is missing field 'id'."
            )

        if question_id in seen_ids:
            raise ValueError(
                f"Duplicate question id: {question_id}"
            )

        seen_ids.add(question_id)

        if not question:
            raise ValueError(
                f"{question_id} is missing field 'question'."
            )

        if not category:
            raise ValueError(
                f"{question_id} is missing field 'category'."
            )


def validate_retrieval(
    questions,
    retrieval_map,
    expected_top_k=5,
    expected_mode="hybrid",
):
    """
    Verify that frozen retrieval results are suitable for
    Configurations B and C.

    This protects the controlled comparison:
    - same questions
    - same retrieval mode
    - same Top-K
    - same retrieved chunks
    """

    question_ids = {item["id"] for item in questions}
    retrieval_ids = set(retrieval_map.keys())

    missing = question_ids - retrieval_ids
    extra = retrieval_ids - question_ids

    if missing:
        raise ValueError(
            "Missing retrieval results for: "
            + ", ".join(sorted(missing))
        )

    if extra:
        raise ValueError(
            "Unexpected retrieval results for: "
            + ", ".join(sorted(extra))
        )

    for item in questions:
        question_id = item["id"]
        retrieval_item = retrieval_map[question_id]

        retrieval_question = retrieval_item.get("question")

        if retrieval_question != item["question"]:
            raise ValueError(
                f"Question mismatch for {question_id}."
            )

        mode = retrieval_item.get("retrieval_mode")
        top_k = retrieval_item.get("top_k")
        contexts = retrieval_item.get("retrieved_chunks", [])

        if mode != expected_mode:
            raise ValueError(
                f"{question_id}: expected retrieval mode "
                f"'{expected_mode}', got '{mode}'."
            )

        if top_k != expected_top_k:
            raise ValueError(
                f"{question_id}: expected top_k="
                f"{expected_top_k}, got {top_k}."
            )

        if len(contexts) != expected_top_k:
            raise ValueError(
                f"{question_id}: expected "
                f"{expected_top_k} retrieved chunks, "
                f"got {len(contexts)}."
            )

        for context in contexts:
            if not context.get("chunk_id"):
                raise ValueError(
                    f"{question_id}: retrieved chunk "
                    f"is missing chunk_id."
                )

            if "page" not in context:
                raise ValueError(
                    f"{question_id}: retrieved chunk "
                    f"is missing report page."
                )

            if not context.get("text"):
                raise ValueError(
                    f"{question_id}: retrieved chunk "
                    f"is missing text."
                )


# ============================================================
# Generation dispatch
# ============================================================

def generate(configuration, question, contexts=None):
    """
    Dispatch a question to the correct generation configuration.
    """

    if configuration == "a":
        return ask_llm(question)

    if configuration == "b":
        return ask_rag(question, contexts)

    if configuration == "c":
        return ask_rag_evidence(question, contexts)

    raise ValueError(
        f"Unknown configuration: {configuration}"
    )


# ============================================================
# Gemini call with retry
# ============================================================

def call_with_retry(
    configuration,
    question,
    contexts=None,
):
    """
    Call Gemini with retry handling.

    Behavior:
    - 429 per-minute quota:
        wait and retry.

    - 429 per-day quota:
        stop immediately because short retries cannot solve it.

    - 503 service unavailable:
        wait and retry.

    - Other errors:
        propagate immediately.
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            return generate(
                configuration,
                question,
                contexts,
            )

        except Exception as e:
            error_message = str(e)

            # ------------------------------------------------
            # 429 - Resource exhausted
            # ------------------------------------------------

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            ):

                # --------------------------------------------
                # Daily quota exhausted
                # --------------------------------------------

                if (
                    "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                    in error_message
                    or
                    "requests per day limit exceeded"
                    in error_message.lower()
                ):
                    raise RuntimeError(
                        "DAILY_QUOTA_EXHAUSTED: "
                        "Gemini free-tier daily request "
                        "limit reached."
                    ) from e

                # --------------------------------------------
                # Temporary / per-minute quota
                # --------------------------------------------

                wait_time = RETRY_429_WAIT

                print(
                    f"  Rate limit (429). "
                    f"Retry {attempt}/{MAX_RETRIES} "
                    f"after {wait_time}s..."
                )

            # ------------------------------------------------
            # 503 - Temporary service overload
            # ------------------------------------------------

            elif (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                wait_time = RETRY_503_WAIT

                print(
                    f"  Service unavailable (503). "
                    f"Retry {attempt}/{MAX_RETRIES} "
                    f"after {wait_time}s..."
                )

            # ------------------------------------------------
            # Unknown/non-transient error
            # ------------------------------------------------

            else:
                raise

            if attempt == MAX_RETRIES:
                raise

            time.sleep(wait_time)

    raise RuntimeError(
        "Maximum retry attempts exceeded."
    )


# ============================================================
# Experiment
# ============================================================

def run_experiment(configuration):
    """
    Run one of the three experiment configurations.

    A = LLM Only
    B = LLM + RAG
    C = LLM + RAG + Evidence
    """

    configuration = configuration.lower()

    if configuration not in OUTPUT_PATHS:
        raise ValueError(
            "Configuration must be one of: a, b, c"
        )

    output_path = OUTPUT_PATHS[configuration]
    configuration_name = CONFIGURATION_NAMES[configuration]

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    questions = load_questions()
    validate_questions(questions)

    # --------------------------------------------------------
    # Load frozen retrieval for B/C
    # --------------------------------------------------------

    retrieval_map = None

    if configuration in {"b", "c"}:
        retrieval_map = load_retrieval_results()

        validate_retrieval(
            questions,
            retrieval_map,
            expected_top_k=5,
            expected_mode="hybrid",
        )

    # --------------------------------------------------------
    # Load previous results for resume
    # --------------------------------------------------------

    existing_results = load_existing_results(
        output_path
    )

    result_map = {
        item["id"]: item
        for item in existing_results
        if "id" in item
    }

    # --------------------------------------------------------
    # Start information
    # --------------------------------------------------------

    print(
        f"Loaded {len(questions)} questions."
    )
    print(
        f"Running Configuration "
        f"{configuration.upper()}: "
        f"{configuration_name}"
    )

    if configuration in {"b", "c"}:
        print(
            "Frozen retrieval: "
            "hybrid, Top-K=5"
        )
        print(
            f"Retrieval source: {RETRIEVAL_PATH}"
        )

    print(
        f"Request interval: {REQUEST_INTERVAL}s"
    )
    print(
        f"Max retries: {MAX_RETRIES}"
    )
    print(
        f"Output: {output_path}"
    )
    print()

    # --------------------------------------------------------
    # Run questions
    # --------------------------------------------------------

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question_id = item["id"]
        question = item["question"]

        print(
            f"[{index}/{len(questions)}] "
            f"{question_id}: {question}"
        )

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------

        previous = result_map.get(question_id)

        if previous and is_successful(previous):
            print(
                "  SKIP | "
                "already completed successfully"
            )
            continue

        # ----------------------------------------------------
        # Get frozen contexts for B/C
        # ----------------------------------------------------

        contexts = None

        if configuration in {"b", "c"}:
            contexts = retrieval_map[
                question_id
            ]["retrieved_chunks"]

        # ----------------------------------------------------
        # Call Gemini
        # ----------------------------------------------------

        try:
            result = call_with_retry(
                configuration,
                question,
                contexts,
            )

            output = {
                "id": question_id,
                "category": item["category"],
                **result,
            }

            # B/C must retain the exact retrieved chunks used
            # during generation so evaluation can compute
            # retrieval Recall@K and inspect provenance.
            if configuration in {"b", "c"}:
                output["retrieved_chunks"] = contexts

            result_map[question_id] = output

            print(
                f"  OK | "
                f"latency={result['latency']:.2f}s | "
                f"input_tokens="
                f"{result['input_tokens']} | "
                f"output_tokens="
                f"{result['output_tokens']}"
            )

            if configuration == "c":
                print(
                    f"  parse_error="
                    f"{result.get('parse_error', False)}"
                )

        # ----------------------------------------------------
        # Handle final failure
        # ----------------------------------------------------

        except Exception as e:
            error_message = str(e)

            # ------------------------------------------------
            # Daily quota
            # ------------------------------------------------

            if (
                "DAILY_QUOTA_EXHAUSTED"
                in error_message
            ):

                ordered_results = (
                    build_ordered_results(
                        questions,
                        result_map,
                    )
                )

                save_results(
                    ordered_results,
                    output_path,
                )

                print()
                print(
                    "========================================"
                )
                print(
                    "DAILY QUOTA EXHAUSTED"
                )
                print(
                    "========================================"
                )
                print(
                    "Gemini free-tier daily request "
                    "limit has been reached."
                )
                print(
                    "Current progress has been saved."
                )
                print(
                    "Run this configuration again "
                    "after the quota resets."
                )
                print()

                break

            # ------------------------------------------------
            # Other error
            # ------------------------------------------------

            print(
                f"  FAILED after retries: "
                f"{error_message}"
            )

            failed_output = {
                "id": question_id,
                "category": item["category"],
                "configuration": {
                    "a": "llm_only",
                    "b": "rag",
                    "c": "rag_evidence",
                }[configuration],
                "answer": "",
                "calculation": None,
                "evidence": [],
                "model": None,
                "latency": None,
                "input_tokens": None,
                "output_tokens": None,
                "error": error_message,
            }

            if configuration == "c":
                failed_output["parse_error"] = True

            if configuration in {"b", "c"}:
                failed_output[
                    "retrieved_chunks"
                ] = contexts

            result_map[question_id] = (
                failed_output
            )

        # ----------------------------------------------------
        # Save after every actual attempt
        # ----------------------------------------------------

        ordered_results = (
            build_ordered_results(
                questions,
                result_map,
            )
        )

        save_results(
            ordered_results,
            output_path,
        )

        # Only actual API attempts wait.
        time.sleep(REQUEST_INTERVAL)

    # ========================================================
    # Final summary
    # ========================================================

    final_results = build_ordered_results(
        questions,
        result_map,
    )

    successful = sum(
        1
        for item in final_results
        if is_successful(item)
    )

    failed = sum(
        1
        for item in final_results
        if not is_successful(item)
    )

    missing = (
        len(questions) - len(final_results)
    )

    print()
    print(
        "========================================"
    )
    print(
        f"Configuration "
        f"{configuration.upper()} summary"
    )
    print(
        "========================================"
    )
    print(
        f"Questions:  {len(questions)}"
    )
    print(
        f"Stored:     {len(final_results)}"
    )
    print(
        f"Successful: {successful}"
    )
    print(
        f"Failed:     {failed}"
    )
    print(
        f"Missing:    {missing}"
    )
    print(
        f"Saved to:   {output_path}"
    )

    if successful == len(questions):
        print()
        print(
            "READY FOR EVALUATION: "
            "all questions completed successfully."
        )

    else:
        print()
        print(
            "NOT READY FOR FINAL EVALUATION."
        )
        print(
            f"{len(questions) - successful} "
            f"question(s) still need "
            f"successful responses."
        )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run financial QA generation experiments."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        choices=["a", "b", "c"],
        help=(
            "a = LLM Only, "
            "b = LLM + RAG, "
            "c = LLM + RAG + Evidence"
        ),
    )

    return parser.parse_args()


# ============================================================
# Entry point
# ============================================================

def main():
    args = parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()