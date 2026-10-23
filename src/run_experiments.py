import json
import time
from pathlib import Path

from llm import ask_llm


# ============================================================
# Configuration
# ============================================================

QUESTIONS_PATH = Path("data/evaluation/questions.json")
OUTPUT_PATH = Path("data/evaluation/llm_only.json")

# Gemini free-tier safety settings
REQUEST_INTERVAL = 15       # seconds between actual API requests
MAX_RETRIES = 5

# Temporary per-minute quota
RETRY_429_WAIT = 65

# Temporary service overload
RETRY_503_WAIT = 20


# ============================================================
# File helpers
# ============================================================

def load_questions():
    """
    Load evaluation questions.
    """
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results():
    """
    Load previous experiment results.

    This allows the experiment to resume without calling
    Gemini again for questions that already succeeded.
    """

    if not OUTPUT_PATH.exists():
        return []

    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except (json.JSONDecodeError, OSError):
        pass

    return []


def save_results(results):
    """
    Save current experiment results.
    """

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
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
    A result is successful when it does not contain
    an API/infrastructure error.
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
# Gemini call with retry
# ============================================================

def call_with_retry(question):
    """
    Call Gemini with retry handling.

    Behavior:
    - 429 per-minute quota:
        wait and retry.

    - 429 per-day quota:
        stop immediately because retrying after a few
        seconds/minutes cannot solve it.

    - 503 service unavailable:
        wait and retry.

    - Other errors:
        propagate immediately.
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            return ask_llm(question)

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
                        "Gemini free-tier daily request limit reached."
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

            # No more retries left.
            if attempt == MAX_RETRIES:
                raise

            time.sleep(wait_time)

    raise RuntimeError("Maximum retry attempts exceeded.")


# ============================================================
# Main experiment
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    questions = load_questions()

    # --------------------------------------------------------
    # Load previous results for resume
    # --------------------------------------------------------

    existing_results = load_existing_results()

    result_map = {
        item["id"]: item
        for item in existing_results
        if "id" in item
    }

    # --------------------------------------------------------
    # Start information
    # --------------------------------------------------------

    print(f"Loaded {len(questions)} questions.")
    print("Running Configuration A: LLM Only")
    print(f"Request interval: {REQUEST_INTERVAL}s")
    print(f"Max retries: {MAX_RETRIES}")
    print()

    # --------------------------------------------------------
    # Run questions
    # --------------------------------------------------------

    for index, item in enumerate(questions, start=1):

        question_id = item["id"]
        question = item["question"]

        print(
            f"[{index}/{len(questions)}] "
            f"{question_id}: {question}"
        )

        # ----------------------------------------------------
        # Resume:
        # Do not call Gemini again for questions that already
        # completed successfully.
        # ----------------------------------------------------

        previous = result_map.get(question_id)

        if previous and is_successful(previous):

            print(
                "  SKIP | "
                "already completed successfully"
            )

            continue

        # ----------------------------------------------------
        # Call Gemini
        # ----------------------------------------------------

        try:

            result = call_with_retry(question)

            output = {
                "id": question_id,
                "category": item["category"],
                **result,
            }

            # Replace previous failed result if one exists.
            result_map[question_id] = output

            print(
                f"  OK | "
                f"latency={result['latency']:.2f}s | "
                f"input_tokens={result['input_tokens']} | "
                f"output_tokens={result['output_tokens']}"
            )

        # ----------------------------------------------------
        # Handle final failure
        # ----------------------------------------------------

        except Exception as e:

            error_message = str(e)

            # ------------------------------------------------
            # Daily quota:
            # Save everything and stop immediately.
            # ------------------------------------------------

            if "DAILY_QUOTA_EXHAUSTED" in error_message:

                ordered_results = build_ordered_results(
                    questions,
                    result_map,
                )

                save_results(ordered_results)

                print()
                print("========================================")
                print("DAILY QUOTA EXHAUSTED")
                print("========================================")
                print(
                    "Gemini free-tier daily request "
                    "limit has been reached."
                )
                print(
                    "Current progress has been saved."
                )
                print(
                    "Run this script again after "
                    "the quota resets."
                )
                print()

                break

            # ------------------------------------------------
            # Other error after retries
            # ------------------------------------------------

            print(
                f"  FAILED after retries: "
                f"{error_message}"
            )

            result_map[question_id] = {
                "id": question_id,
                "category": item["category"],
                "configuration": "llm_only",
                "answer": "",
                "calculation": None,
                "evidence": [],
                "model": None,
                "latency": None,
                "input_tokens": None,
                "output_tokens": None,
                "error": error_message,
            }

        # ----------------------------------------------------
        # Save after every attempted question
        # ----------------------------------------------------

        ordered_results = build_ordered_results(
            questions,
            result_map,
        )

        save_results(ordered_results)

        # ----------------------------------------------------
        # Rate limiting
        #
        # Only reached after an actual API attempt.
        # SKIP questions use "continue" above and therefore
        # do not wait 15 seconds.
        # ----------------------------------------------------

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

    missing = len(questions) - len(final_results)

    print()
    print("========================================")
    print("Configuration A summary")
    print("========================================")
    print(f"Questions:  {len(questions)}")
    print(f"Stored:     {len(final_results)}")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Missing:    {missing}")
    print(f"Saved to:   {OUTPUT_PATH}")

    # --------------------------------------------------------
    # Evaluation readiness
    # --------------------------------------------------------

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
            f"question(s) still need successful responses."
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()