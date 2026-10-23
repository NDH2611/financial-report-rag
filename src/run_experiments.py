import json
from pathlib import Path

from llm import ask_llm


QUESTIONS_PATH = Path("data/evaluation/questions.json")
OUTPUT_PATH = Path("data/evaluation/llm_only.json")


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    questions = load_questions()

    results = []

    print(f"Loaded {len(questions)} questions.")
    print("Running Configuration A: LLM Only\n")

    for index, item in enumerate(questions, start=1):
        question_id = item["id"]
        question = item["question"]

        print(
            f"[{index}/{len(questions)}] "
            f"{question_id}: {question}"
        )

        try:
            result = ask_llm(question)

            # Keep only prediction-side information.
            # Ground truth must NEVER be sent to the LLM.
            output = {
                "id": question_id,
                "category": item["category"],
                **result,
            }

            results.append(output)

            print(
                f"  OK | "
                f"latency={result['latency']:.2f}s | "
                f"input_tokens={result['input_tokens']} | "
                f"output_tokens={result['output_tokens']}"
            )

        except Exception as e:
            print(f"  ERROR: {e}")

            results.append({
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
                "error": str(e),
            })

        # Save after every question so progress is not lost.
        save_results(results)

    print("\nFinished.")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()