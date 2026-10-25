import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np


# ============================================================
# Paths / Configuration
# ============================================================

CHUNKS_PATH = Path("data/processed/chunks.json")
INDEX_PATH = Path("indexes/faiss.index")
QUESTIONS_PATH = Path("data/evaluation/questions.json")

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "BAAI/bge-small-en-v1.5",
)

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
    if "bge" in EMBED_MODEL.lower()
    else ""
)

# Reciprocal Rank Fusion constant
RRF_K = 60


# ============================================================
# Lazy-loaded global resources
# ============================================================

_model = None
_chunks = None
_index = None
_bm25 = None


# ============================================================
# Embedding model
# ============================================================

def _get_model():
    """
    Lazily load the sentence-transformer embedding model.
    """
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)

    return _model


# ============================================================
# Chunk loading / validation
# ============================================================

def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """
    Load chunks produced by extract.py.

    Canonical schema:

    {
        "chunk_id": "...",
        "text": "...",
        "metadata": {
            "company": "Apple Inc.",
            "fiscal_year": 2025,
            "pdf_page": 32,
            "page": 29,
            "section": "...",
            "type": "text"
        }
    }

    Page convention:

        metadata["pdf_page"]
            Physical page in the PDF file.

        metadata["page"]
            Printed page shown inside the Apple Form 10-K.

    questions.json/evidence_pages uses printed report pages.
    Therefore chunk["page"] is the canonical downstream page.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError(
            f"{path} must contain a JSON list of chunks."
        )

    if not chunks:
        raise ValueError(
            f"{path} is empty."
        )

    for i, chunk in enumerate(chunks):

        # ----------------------------------------------------
        # Required top-level fields
        # ----------------------------------------------------

        for key in ("chunk_id", "text"):

            if key not in chunk:
                raise ValueError(
                    f"Chunk #{i} is missing '{key}'."
                )

        if not isinstance(chunk["text"], str):
            raise ValueError(
                f"Chunk #{i} has invalid 'text'."
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = chunk.get("metadata")

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Chunk #{i} is missing valid 'metadata'."
            )

        if "page" not in metadata:
            raise ValueError(
                f"Chunk #{i} is missing 'metadata.page'. "
                "Regenerate chunks.json using the updated extract.py."
            )

        if "pdf_page" not in metadata:
            raise ValueError(
                f"Chunk #{i} is missing 'metadata.pdf_page'. "
                "Regenerate chunks.json using the updated extract.py."
            )

        # ----------------------------------------------------
        # Validate page values
        # ----------------------------------------------------

        report_page = metadata["page"]
        pdf_page = metadata["pdf_page"]

        if report_page is not None:

            try:
                report_page = int(report_page)

            except (TypeError, ValueError) as exc:

                raise ValueError(
                    f"Chunk #{i} has invalid "
                    f"metadata.page: {report_page}"
                ) from exc

        try:
            pdf_page = int(pdf_page)

        except (TypeError, ValueError) as exc:

            raise ValueError(
                f"Chunk #{i} has invalid "
                f"metadata.pdf_page: {pdf_page}"
            ) from exc

        # ----------------------------------------------------
        # Flatten metadata for downstream modules
        # ----------------------------------------------------

        chunk["page"] = report_page
        chunk["pdf_page"] = pdf_page

        chunk["section"] = metadata.get(
            "section"
        )

        chunk["company"] = metadata.get(
            "company"
        )

        chunk["fiscal_year"] = metadata.get(
            "fiscal_year"
        )

        chunk["type"] = metadata.get(
            "type"
        )

    # --------------------------------------------------------
    # Validate chunk IDs
    # --------------------------------------------------------

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate chunk_id found in chunks.json."
        )

    return chunks


# ============================================================
# Embedding
# ============================================================

def _embed(
    texts: list[str],
    is_query: bool = False,
) -> np.ndarray:
    """
    Embed text using the configured sentence-transformer.

    Embeddings are L2-normalized so FAISS IndexFlatIP behaves
    as cosine similarity.
    """

    if is_query and QUERY_PREFIX:
        texts = [
            QUERY_PREFIX + text
            for text in texts
        ]

    vectors = _get_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=not is_query,
    )

    return np.asarray(
        vectors,
        dtype="float32",
    )


# ============================================================
# BM25 tokenizer
# ============================================================

def _tokenize(text: str) -> list[str]:
    """
    Lightweight tokenizer for BM25 retrieval.
    """

    return re.findall(
        r"[a-z0-9$%.,]+",
        text.lower(),
    )


# ============================================================
# Build FAISS index
# ============================================================

def build_index(
    chunks_path: Path = CHUNKS_PATH,
    index_path: Path = INDEX_PATH,
) -> None:
    """
    Build a dense FAISS index from chunks.json.
    """

    chunks = load_chunks(chunks_path)

    print(f"Loaded {len(chunks)} chunks.")
    print("Generating embeddings...")

    vectors = _embed(
        [chunk["text"] for chunk in chunks]
    )

    # Normalized embeddings + inner product
    # = cosine similarity
    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )

    index.add(vectors)

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(index_path),
    )

    # --------------------------------------------------------
    # Save metadata for index consistency checking
    # --------------------------------------------------------

    meta = {
        "embed_model": EMBED_MODEL,
        "num_chunks": len(chunks),
        "chunk_ids": [
            chunk["chunk_id"]
            for chunk in chunks
        ],
    }

    meta_path = index_path.with_suffix(
        ".meta.json"
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            meta,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Indexed {len(chunks)} chunks, "
        f"dim={vectors.shape[1]} "
        f"-> {index_path}"
    )


# ============================================================
# Load FAISS index
# ============================================================

def _load_index(
    chunks_path: Path = CHUNKS_PATH,
    index_path: Path = INDEX_PATH,
):
    """
    Load chunks and FAISS index.

    Also verifies that chunks.json and the index metadata
    still correspond to each other.
    """

    global _chunks
    global _index

    if _index is not None:
        return _chunks, _index

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\n"
            "Run:\n"
            "python src/retrieval.py build"
        )

    _chunks = load_chunks(chunks_path)

    _index = faiss.read_index(
        str(index_path)
    )

    # --------------------------------------------------------
    # Basic consistency check
    # --------------------------------------------------------

    if _index.ntotal != len(_chunks):
        raise RuntimeError(
            "FAISS index size does not match chunks.json.\n"
            "Run:\n"
            "python src/retrieval.py build"
        )

    meta_path = index_path.with_suffix(
        ".meta.json"
    )

    if meta_path.exists():

        with open(
            meta_path,
            "r",
            encoding="utf-8",
        ) as f:
            meta = json.load(f)

        current_ids = [
            chunk["chunk_id"]
            for chunk in _chunks
        ]

        if meta.get("chunk_ids") != current_ids:
            raise RuntimeError(
                "chunks.json changed since the index was built.\n"
                "Run:\n"
                "python src/retrieval.py build"
            )

        if meta.get("embed_model") != EMBED_MODEL:
            raise RuntimeError(
                f"Index built with "
                f"{meta.get('embed_model')} "
                f"but EMBED_MODEL={EMBED_MODEL}.\n"
                "Rebuild the index."
            )

    return _chunks, _index


# ============================================================
# BM25
# ============================================================

def _get_bm25(chunks):
    """
    Lazily build the BM25 index.
    """

    global _bm25

    if _bm25 is None:

        from rank_bm25 import BM25Okapi

        corpus = [
            _tokenize(chunk["text"])
            for chunk in chunks
        ]

        _bm25 = BM25Okapi(corpus)

    return _bm25


# ============================================================
# Dense retrieval
# ============================================================

def _dense_ranking(
    question: str,
    n: int,
):
    """
    Return dense ranking as:

        [(chunk_index, similarity_score), ...]
    """

    chunks, index = _load_index()

    q_vector = _embed(
        [question],
        is_query=True,
    )

    n = min(
        n,
        len(chunks),
    )

    scores, ids = index.search(
        q_vector,
        n,
    )

    return [
        (int(idx), float(score))
        for idx, score
        in zip(ids[0], scores[0])
        if idx >= 0
    ]


# ============================================================
# BM25 retrieval
# ============================================================

def _bm25_ranking(
    question: str,
    n: int,
):
    """
    Return BM25 ranking as:

        [(chunk_index, bm25_score), ...]
    """

    chunks, _ = _load_index()

    bm25 = _get_bm25(chunks)

    scores = bm25.get_scores(
        _tokenize(question)
    )

    n = min(
        n,
        len(chunks),
    )

    order = np.argsort(scores)[::-1][:n]

    return [
        (
            int(idx),
            float(scores[idx]),
        )
        for idx in order
    ]


# ============================================================
# Main retrieval interface
# ============================================================

def retrieve(
    question: str,
    top_k: int = 5,
    mode: str = "dense",
    candidate_pool: int = 30,
) -> list[dict]:
    """
    Retrieve Top-K relevant chunks.

    Supported modes:
        dense
        bm25
        hybrid

    Hybrid retrieval uses Reciprocal Rank Fusion (RRF).

    Returns:
        [
            {
                "chunk_id": "...",
                "page": 29,
                "text": "...",
                "score": 0.91,
                ...
            }
        ]
    """

    if not isinstance(question, str):
        raise TypeError(
            "question must be a string."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "question must not be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    chunks, _ = _load_index()

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    if mode == "dense":

        ranked = _dense_ranking(
            question,
            top_k,
        )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    elif mode == "bm25":

        ranked = _bm25_ranking(
            question,
            top_k,
        )

    # --------------------------------------------------------
    # Hybrid = Dense + BM25 + RRF
    # --------------------------------------------------------

    elif mode == "hybrid":

        candidate_pool = max(
            candidate_pool,
            top_k,
        )

        candidate_pool = min(
            candidate_pool,
            len(chunks),
        )

        fused = defaultdict(float)

        dense_ranking = _dense_ranking(
            question,
            candidate_pool,
        )

        bm25_ranking = _bm25_ranking(
            question,
            candidate_pool,
        )

        for ranking in (
            dense_ranking,
            bm25_ranking,
        ):

            for rank, (idx, _) in enumerate(
                ranking
            ):

                fused[idx] += (
                    1.0
                    / (RRF_K + rank + 1)
                )

        ranked = sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

    else:

        raise ValueError(
            f"Unknown retrieval mode: {mode}"
        )

    # --------------------------------------------------------
    # Build standardized Retrieval -> LLM result
    # --------------------------------------------------------

    results = []

    for idx, score in ranked:

        chunk = chunks[idx]

        result = {
            **chunk,
            "score": float(score),
        }

        results.append(result)

    return results


# ============================================================
# Retrieval evaluation
# ============================================================

def evaluate_retrieval(
    top_k: int,
    mode: str,
) -> None:
    """
    Evaluate page-level hit Recall@K using evidence_pages
    from questions.json.

    A question counts as a hit if at least one expected
    evidence page appears in the retrieved Top-K chunks.

    Unanswerable questions without evidence_pages are skipped.
    """

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Questions file not found: "
            f"{QUESTIONS_PATH}"
        )

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        questions = json.load(f)

    by_category = defaultdict(list)

    misses = []

    evaluated = 0

    for item in questions:

        expected_pages = set(
            item.get("evidence_pages") or []
        )

        # Unanswerable questions normally have no
        # ground-truth evidence page.
        if not expected_pages:
            continue

        contexts = retrieve(
            item["question"],
            top_k=top_k,
            mode=mode,
        )

        retrieved_pages = {
            chunk["page"]
            for chunk in contexts
            if chunk.get("page") is not None
        }

        hit = bool(
            expected_pages
            & retrieved_pages
        )

        category = item.get(
            "category",
            "unknown",
        )

        by_category[category].append(
            hit
        )

        evaluated += 1

        if not hit:

            misses.append(
                (
                    item["id"],
                    sorted(expected_pages),
                    sorted(retrieved_pages),
                )
            )

    all_hits = [
        hit
        for hits in by_category.values()
        for hit in hits
    ]

    hits_count = sum(all_hits)

    recall = (
        100 * hits_count / len(all_hits)
        if all_hits
        else 0.0
    )

    print()
    print(
        f"Recall@{top_k} ({mode}): "
        f"{hits_count}/{len(all_hits)} "
        f"= {recall:.1f}%"
    )

    print()

    for category, hits in sorted(
        by_category.items()
    ):

        category_hits = sum(hits)

        print(
            f"  {category:15s} "
            f"{category_hits}/{len(hits)}"
        )

    if misses:

        print()
        print(
            "Missed questions "
            "(id, expected pages, retrieved pages):"
        )

        for miss in misses:
            print("  ", miss)

    else:

        print()
        print(
            "No retrieval misses "
            "for answerable questions."
        )

    print()
    print(
        f"Evaluated answerable questions: "
        f"{evaluated}"
    )


# ============================================================
# Export retrieval results
# ============================================================

def export_results(
    top_k: int,
    mode: str,
    out_path: str = (
        "data/evaluation/"
        "retrieval_results.json"
    ),
) -> None:
    """
    Run retrieval once for every evaluation question
    and save the Top-K contexts.

    The exported file can later be reused by both
    Configuration B and Configuration C so that they receive
    exactly the same retrieved contexts.
    """

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Questions file not found: "
            f"{QUESTIONS_PATH}"
        )

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        questions = json.load(f)

    results = []

    for index, item in enumerate(
        questions,
        start=1,
    ):

        print(
            f"[{index}/{len(questions)}] "
            f"{item['id']}: "
            f"{item['question']}"
        )

        contexts = retrieve(
            item["question"],
            top_k=top_k,
            mode=mode,
        )

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "retrieval_mode": mode,
                "top_k": top_k,
                "retrieved_chunks": contexts,
            }
        )

    out = Path(out_path)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Saved {len(results)} questions "
        f"(top_k={top_k}, mode={mode}) "
        f"-> {out}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Apple FY2025 10-K "
            "retrieval module"
        )
    )

    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    sub.add_parser(
        "build",
        help="Build FAISS dense index.",
    )

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    query_parser = sub.add_parser(
        "query",
        help="Retrieve chunks for one question.",
    )

    query_parser.add_argument(
        "question",
    )

    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    query_parser.add_argument(
        "--mode",
        default="dense",
        choices=[
            "dense",
            "bm25",
            "hybrid",
        ],
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_parser = sub.add_parser(
        "eval",
        help="Evaluate retrieval Recall@K.",
    )

    eval_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    eval_parser.add_argument(
        "--mode",
        default="dense",
        choices=[
            "dense",
            "bm25",
            "hybrid",
        ],
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    export_parser = sub.add_parser(
        "export",
        help=(
            "Export retrieval results "
            "for all evaluation questions."
        ),
    )

    export_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    export_parser.add_argument(
        "--mode",
        default="dense",
        choices=[
            "dense",
            "bm25",
            "hybrid",
        ],
    )

    export_parser.add_argument(
        "--out",
        default=(
            "data/evaluation/"
            "retrieval_results.json"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Execute command
    # --------------------------------------------------------

    if args.cmd == "build":

        build_index()

    elif args.cmd == "query":

        results = retrieve(
            args.question,
            top_k=args.top_k,
            mode=args.mode,
        )

        print()

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"#{rank} "
                f"[{result['score']:.4f}] "
                f"{result['chunk_id']} "
                f"(report page {result['page']}, "
                f"PDF page {result.get('pdf_page')})"
            )

            print(
                "   ",
                result["text"][:300]
                .replace("\n", " "),
            )

            print()

    elif args.cmd == "eval":

        evaluate_retrieval(
            top_k=args.top_k,
            mode=args.mode,
        )

    elif args.cmd == "export":

        export_results(
            top_k=args.top_k,
            mode=args.mode,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()