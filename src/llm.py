import json
import os
import time

from dotenv import load_dotenv
from google import genai


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

MODEL_NAME = "gemini-3.5-flash"

client = genai.Client(api_key=API_KEY)


# ============================================================
# Core LLM call
# ============================================================

def call_llm(prompt: str) -> dict:
    """
    Send a prompt to Gemini and return the generated text
    together with latency and token usage.

    Args:
        prompt: Prompt sent to the LLM.

    Returns:
        Dictionary containing:
        - text
        - model
        - latency
        - input_tokens
        - output_tokens
    """

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    latency = time.perf_counter() - start_time
    usage = response.usage_metadata

    return {
        "text": response.text,
        "model": MODEL_NAME,
        "latency": latency,
        "input_tokens": (
            usage.prompt_token_count if usage else None
        ),
        "output_tokens": (
            usage.candidates_token_count if usage else None
        ),
    }


# ============================================================
# Configuration A: LLM Only
# ============================================================

def ask_llm(question: str) -> dict:
    """
    Configuration A: LLM Only.

    Answer a financial question without retrieved context.

    Args:
        question: Financial question.

    Returns:
        Standardized generation result.
    """

    prompt = f"""
You are a financial report analysis assistant.

Answer the following financial question clearly and concisely.

QUESTION:
{question}

Rules:
- Do not invent financial values.
- Preserve financial units when they are known.
- If a calculation is required, show the calculation clearly.
- If you do not have enough reliable information, say that
  the information is insufficient.
"""

    result = call_llm(prompt)

    return {
        "configuration": "llm_only",
        "answer": result["text"],
        "calculation": None,
        "evidence": [],
        "model": result["model"],
        "latency": result["latency"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }


# ============================================================
# Context formatting
# ============================================================

def format_context(contexts: list[dict]) -> str:
    """
    Convert retrieved chunks into text context for the LLM.

    Expected context format:
    {
        "chunk_id": str,
        "page": int,
        "text": str,
        "score": float  # optional
    }

    Args:
        contexts: Retrieved chunks.

    Returns:
        Formatted context string.
    """

    formatted_chunks = []

    for context in contexts:
        chunk_id = context.get("chunk_id", "unknown")
        page = context.get("page", "unknown")
        text = context.get("text", "")

        formatted_chunk = (
            f"[Chunk: {chunk_id}]\n"
            f"[Page: {page}]\n"
            f"{text}"
        )

        formatted_chunks.append(formatted_chunk)

    return "\n\n".join(formatted_chunks)


# ============================================================
# Configuration B: LLM + RAG
# ============================================================

def ask_rag(question: str, contexts: list[dict]) -> dict:
    """
    Configuration B: LLM + RAG.

    Answer a financial question using only retrieved context.

    Args:
        question: Financial question.
        contexts: Top-K chunks returned by the retrieval module.

    Returns:
        Standardized generation result.
    """

    context_text = format_context(contexts)

    prompt = f"""
You are a financial report analysis assistant.

Answer the financial question using ONLY the provided context.

CONTEXT:
{context_text}

QUESTION:
{question}

Rules:
- Use only information from the provided context.
- Do not use outside knowledge.
- Do not invent financial values.
- Preserve years and financial units exactly.
- If a calculation is required, show the calculation clearly.
- If the provided context does not contain enough information,
  explicitly say:
  "Insufficient information in the provided context."

Return a clear and concise answer.
"""

    result = call_llm(prompt)

    return {
        "configuration": "rag",
        "answer": result["text"],
        "calculation": None,
        "evidence": [],
        "model": result["model"],
        "latency": result["latency"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }


# ============================================================
# Configuration C: LLM + RAG + Evidence
# ============================================================

def ask_rag_evidence(
    question: str,
    contexts: list[dict],
) -> dict:
    """
    Configuration C: LLM + RAG + Evidence.

    Answer a financial question using retrieved context and
    return structured evidence.

    Args:
        question: Financial question.
        contexts: Top-K chunks returned by the retrieval module.

    Returns:
        Standardized generation result containing:
        - answer
        - calculation
        - evidence
        - model
        - latency
        - token usage
        - parse_error
    """

    context_text = format_context(contexts)

    prompt = f"""
You are a financial report analysis assistant.

Answer the financial question using ONLY the provided context.

CONTEXT:
{context_text}

QUESTION:
{question}

Rules:
- Use only information from the provided context.
- Do not use outside knowledge.
- Do not invent financial values.
- Preserve years and financial units exactly.

- If a calculation is required:
  1. Identify the input values.
  2. Show the formula.
  3. Substitute the values.
  4. Show the final result.

- Every factual answer must be supported by evidence from
  the provided context.

- Evidence must use chunk IDs and page numbers that appear
  in the provided context.

- Never invent a chunk ID or page number.

- If the provided context does not contain enough information,
  use exactly:
  "Insufficient information in the provided context."

- If the information is insufficient:
  calculation must be null
  and evidence must be an empty list.

- Return ONLY valid JSON.
- Do not include Markdown code fences.

Use exactly this JSON structure:

{{
  "answer": "final answer",
  "calculation": null,
  "evidence": [
    {{
      "chunk_id": "source chunk id",
      "page": 0,
      "quote": "supporting evidence from the context"
    }}
  ]
}}
"""

    result = call_llm(prompt)

    try:
        structured_output = json.loads(result["text"])

        answer = structured_output.get("answer", "")
        calculation = structured_output.get("calculation")
        evidence = structured_output.get("evidence", [])
        parse_error = False

    except (json.JSONDecodeError, TypeError):
        answer = result["text"]
        calculation = None
        evidence = []
        parse_error = True

    return {
        "configuration": "rag_evidence",
        "answer": answer,
        "calculation": calculation,
        "evidence": evidence,
        "model": result["model"],
        "latency": result["latency"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "parse_error": parse_error,
    }