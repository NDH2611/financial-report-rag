import sys
import time

from llm import (
    MODEL_NAME,
    ask_llm,
    ask_rag,
    ask_rag_evidence,
)

from retrieval import retrieve


# ============================================================
# Demo Configuration
# ============================================================

TOP_K = 5
RETRIEVAL_MODE = "hybrid"


# ============================================================
# Terminal formatting helpers
# ============================================================

SEPARATOR = "=" * 72
SUB_SEPARATOR = "-" * 72


def print_header():
    print()
    print(SEPARATOR)
    print("FINANCIAL REPORT QA - LIVE DEMO")
    print(SEPARATOR)
    print(f"LLM       : {MODEL_NAME}")
    print(f"Retrieval : {RETRIEVAL_MODE}")
    print(f"Top-K     : {TOP_K}")
    print(SEPARATOR)


def print_menu():
    print()
    print("Choose configuration:")
    print()
    print("  A  - LLM Only")
    print("  B  - LLM + RAG")
    print("  C  - LLM + RAG + Evidence")
    print()
    print("  Q  - Quit")
    print()


def print_retrieved_chunks(contexts):
    """
    Print a compact summary of retrieved chunks.
    """

    print()
    print(SUB_SEPARATOR)
    print(f"RETRIEVED CONTEXT - HYBRID TOP-{TOP_K}")
    print(SUB_SEPARATOR)

    for rank, chunk in enumerate(contexts, start=1):
        chunk_id = chunk.get("chunk_id", "unknown")
        report_page = chunk.get("page")
        pdf_page = chunk.get("pdf_page")
        score = chunk.get("score")

        if score is None:
            score_text = "N/A"
        else:
            score_text = f"{score:.6f}"

        print(
            f"[{rank}] "
            f"{chunk_id}"
        )

        print(
            f"    Report page : {report_page}"
        )

        print(
            f"    PDF page    : {pdf_page}"
        )

        print(
            f"    Score       : {score_text}"
        )

    print(SUB_SEPARATOR)


def print_generation_result(result):
    """
    Print standardized generation output.
    """

    print()
    print(SUB_SEPARATOR)
    print("GENERATION RESULT")
    print(SUB_SEPARATOR)

    print()
    print("ANSWER")
    print(result.get("answer", ""))

    calculation = result.get("calculation")

    if calculation is not None:
        print()
        print("CALCULATION")

        if isinstance(calculation, (dict, list)):
            import json

            print(
                json.dumps(
                    calculation,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(calculation)

    evidence = result.get("evidence", [])

    if evidence:
        print()
        print("EVIDENCE")

        for index, item in enumerate(
            evidence,
            start=1,
        ):
            print()
            print(f"[{index}]")

            print(
                f"Chunk : "
                f"{item.get('chunk_id', 'unknown')}"
            )

            print(
                f"Page  : "
                f"{item.get('page', 'unknown')}"
            )

            quote = item.get("quote")

            if quote:
                print(
                    f"Quote : {quote}"
                )

    print()
    print(SUB_SEPARATOR)
    print("RUNTIME METADATA")
    print(SUB_SEPARATOR)

    print(
        f"Configuration : "
        f"{result.get('configuration')}"
    )

    print(
        f"Model         : "
        f"{result.get('model')}"
    )

    latency = result.get("latency")

    if latency is not None:
        print(
            f"Latency       : "
            f"{latency:.3f} s"
        )
    else:
        print(
            "Latency       : N/A"
        )

    print(
        f"Input tokens  : "
        f"{result.get('input_tokens')}"
    )

    print(
        f"Output tokens : "
        f"{result.get('output_tokens')}"
    )

    if "parse_error" in result:
        print(
            f"Parse error   : "
            f"{result.get('parse_error')}"
        )

    print(SUB_SEPARATOR)


# ============================================================
# Configuration runners
# ============================================================

def run_configuration_a(question):
    """
    Configuration A:
    Question -> Gemini
    """

    print()
    print("[A] LLM Only")
    print("No retrieval context is provided.")
    print()
    print("Calling Gemini...")

    result = ask_llm(question)

    return result


def run_configuration_b(question):
    """
    Configuration B:
    Question -> Hybrid Top-K -> Gemini
    """

    print()
    print("[B] LLM + RAG")

    print(
        f"Retrieving Hybrid Top-{TOP_K}..."
    )

    retrieval_start = time.perf_counter()

    contexts = retrieve(
        question,
        top_k=TOP_K,
        mode=RETRIEVAL_MODE,
    )

    retrieval_latency = (
        time.perf_counter()
        - retrieval_start
    )

    print(
        f"Retrieved {len(contexts)} chunks "
        f"in {retrieval_latency:.3f} s."
    )

    print_retrieved_chunks(contexts)

    print()
    print("Calling Gemini with retrieved context...")

    result = ask_rag(
        question,
        contexts,
    )

    result["retrieval_latency"] = (
        retrieval_latency
    )

    return result


def run_configuration_c(question):
    """
    Configuration C:
    Question -> Hybrid Top-K -> Gemini
             -> Answer + Calculation + Evidence
    """

    print()
    print("[C] LLM + RAG + Evidence")

    print(
        f"Retrieving Hybrid Top-{TOP_K}..."
    )

    retrieval_start = time.perf_counter()

    contexts = retrieve(
        question,
        top_k=TOP_K,
        mode=RETRIEVAL_MODE,
    )

    retrieval_latency = (
        time.perf_counter()
        - retrieval_start
    )

    print(
        f"Retrieved {len(contexts)} chunks "
        f"in {retrieval_latency:.3f} s."
    )

    print_retrieved_chunks(contexts)

    print()
    print(
        "Calling Gemini with retrieved context "
        "and evidence constraints..."
    )

    result = ask_rag_evidence(
        question,
        contexts,
    )

    result["retrieval_latency"] = (
        retrieval_latency
    )

    return result


# ============================================================
# Demo execution
# ============================================================

def run_question(configuration, question):
    """
    Execute one question using the selected configuration.
    """

    print()
    print(SEPARATOR)

    print(
        f"QUESTION: {question}"
    )

    print(SEPARATOR)

    try:
        if configuration == "A":
            result = run_configuration_a(
                question
            )

        elif configuration == "B":
            result = run_configuration_b(
                question
            )

        elif configuration == "C":
            result = run_configuration_c(
                question
            )

        else:
            raise ValueError(
                "Unknown configuration."
            )

        print_generation_result(result)

        retrieval_latency = result.get(
            "retrieval_latency"
        )

        if retrieval_latency is not None:
            print()
            print(
                f"Retrieval latency : "
                f"{retrieval_latency:.3f} s"
            )

            generation_latency = result.get(
                "latency"
            )

            if generation_latency is not None:
                total_latency = (
                    retrieval_latency
                    + generation_latency
                )

                print(
                    f"Generation latency: "
                    f"{generation_latency:.3f} s"
                )

                print(
                    f"Approx. total     : "
                    f"{total_latency:.3f} s"
                )

        print()

    except KeyboardInterrupt:
        print()
        print("Request cancelled.")

    except Exception as exc:
        print()
        print(SUB_SEPARATOR)
        print("ERROR")
        print(SUB_SEPARATOR)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print(SUB_SEPARATOR)
        print()


# ============================================================
# Main interactive loop
# ============================================================

def main():
    print_header()

    while True:
        print_menu()

        choice = input(
            "Configuration [A/B/C/Q]: "
        ).strip().upper()

        if choice == "Q":
            print()
            print("Demo finished.")
            return

        if choice not in {
            "A",
            "B",
            "C",
        }:
            print()
            print(
                "Invalid choice. "
                "Please choose A, B, C, or Q."
            )
            continue

        print()

        if choice == "A":
            print(
                "Selected: "
                "Configuration A - LLM Only"
            )

        elif choice == "B":
            print(
                "Selected: "
                "Configuration B - LLM + RAG"
            )

        else:
            print(
                "Selected: "
                "Configuration C - "
                "LLM + RAG + Evidence"
            )

        print()
        print(
            "Enter a financial question."
        )

        print(
            "Type /back to choose another "
            "configuration."
        )

        print(
            "Type /quit to exit."
        )

        while True:
            print()

            question = input(
                "Question > "
            ).strip()

            if not question:
                continue

            command = question.lower()

            if command == "/quit":
                print()
                print("Demo finished.")
                return

            if command == "/back":
                break

            run_question(
                choice,
                question,
            )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("Demo finished.")
        sys.exit(0)