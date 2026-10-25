import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np


CHUNKS_PATH = Path("data/processed/chunks.json")
INDEX_PATH = Path("indexes/faiss.index")
QUESTIONS_PATH = Path("data/evaluation/questions.json")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
    if "bge" in EMBED_MODEL.lower()
    else ""
)

RRF_K = 60  

_model = None
_chunks = None
_index = None
_bm25 = None



def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError(f"{path} is empty.")

    for i, c in enumerate(chunks):
        for key in ("chunk_id", "page", "text"):
            if key not in c:
                raise ValueError(f"Chunk #{i} is missing '{key}'.")

    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk_id found in chunks.json.")

    return chunks


def _embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    if is_query and QUERY_PREFIX:
        texts = [QUERY_PREFIX + t for t in texts]

    vectors = _get_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,  
        show_progress_bar=not is_query,
    )
    return np.asarray(vectors, dtype="float32")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$%.,]+", text.lower())




def build_index(
    chunks_path: Path = CHUNKS_PATH,
    index_path: Path = INDEX_PATH,
) -> None:
    chunks = load_chunks(chunks_path)
    vectors = _embed([c["text"] for c in chunks])

    index = faiss.IndexFlatIP(vectors.shape[1])  
    index.add(vectors)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    meta = {
        "embed_model": EMBED_MODEL,
        "num_chunks": len(chunks),
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }
    with open(index_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"Indexed {len(chunks)} chunks, dim={vectors.shape[1]} -> {index_path}")


def _load_index(
    chunks_path: Path = CHUNKS_PATH,
    index_path: Path = INDEX_PATH,
):
    global _chunks, _index

    if _index is not None:
        return _chunks, _index

    _chunks = load_chunks(chunks_path)
    _index = faiss.read_index(str(index_path))

    meta_path = index_path.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta["chunk_ids"] != [c["chunk_id"] for c in _chunks]:
            raise RuntimeError(
                "chunks.json changed since the index was built. "
                "Run: python src/retrieval.py build"
            )
        if meta["embed_model"] != EMBED_MODEL:
            raise RuntimeError(
                f"Index built with {meta['embed_model']} but EMBED_MODEL="
                f"{EMBED_MODEL}. Rebuild the index."
            )

    return _chunks, _index


def _get_bm25(chunks):
    global _bm25
    if _bm25 is None:
        from rank_bm25 import BM25Okapi
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    return _bm25



def _dense_ranking(question: str, n: int):
    chunks, index = _load_index()
    q = _embed([question], is_query=True)
    scores, ids = index.search(q, min(n, len(chunks)))
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def _bm25_ranking(question: str, n: int):
    chunks, _ = _load_index()
    scores = _get_bm25(chunks).get_scores(_tokenize(question))
    order = np.argsort(scores)[::-1][:n]
    return [(int(i), float(scores[i])) for i in order]


def retrieve(
    question: str,
    top_k: int = 5,
    mode: str = "dense",
    candidate_pool: int = 30,
) -> list[dict]:
    chunks, _ = _load_index()

    if mode == "dense":
        ranked = _dense_ranking(question, top_k)

    elif mode == "bm25":
        ranked = _bm25_ranking(question, top_k)

    elif mode == "hybrid":
        fused = defaultdict(float)
        for ranking in (
            _dense_ranking(question, candidate_pool),
            _bm25_ranking(question, candidate_pool),
        ):
            for rank, (idx, _) in enumerate(ranking):
                fused[idx] += 1.0 / (RRF_K + rank + 1)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return [{**chunks[idx], "score": score} for idx, score in ranked]



def evaluate_retrieval(top_k: int, mode: str) -> None:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    by_cat = defaultdict(list)
    misses = []

    for q in questions:
        expected = set(q.get("evidence_pages") or [])
        if not expected:
            continue

        got = {c["page"] for c in retrieve(q["question"], top_k, mode)}
        hit = bool(expected & got)
        by_cat[q["category"]].append(hit)

        if not hit:
            misses.append((q["id"], sorted(expected), sorted(got)))

    total = [h for hits in by_cat.values() for h in hits]
    print(f"\nRecall@{top_k} ({mode}): {sum(total)}/{len(total)} "
          f"= {100 * sum(total) / max(len(total), 1):.1f}%")

    for cat, hits in sorted(by_cat.items()):
        print(f"  {cat:15s} {sum(hits)}/{len(hits)}")

    if misses:
        print("\nMissed questions (id, expected pages, retrieved pages):")
        for m in misses:
            print("  ", m)

def export_results(
    top_k: int,
    mode: str,
    out_path: str = "data/evaluation/retrieval_results.json",
) -> None:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
 
    results = [
        {
            "id": q["id"],
            "question": q["question"],
            "retrieved_chunks": retrieve(q["question"], top_k, mode),
        }
        for q in questions
    ]
 
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
 
    print(f"Saved {len(results)} questions (top_k={top_k}, mode={mode}) -> {out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
 
    sub.add_parser("build")
 
    p_query = sub.add_parser("query")
    p_query.add_argument("question")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--mode", default="dense",
                         choices=["dense", "bm25", "hybrid"])
 
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--mode", default="dense",
                        choices=["dense", "bm25", "hybrid"])
 
    p_export = sub.add_parser("export")
    p_export.add_argument("--top-k", type=int, default=5)
    p_export.add_argument("--mode", default="dense",
                          choices=["dense", "bm25", "hybrid"])
    p_export.add_argument("--out", default="data/evaluation/retrieval_results.json")
 
    args = parser.parse_args()
 
    if args.cmd == "build":
        build_index()
 
    elif args.cmd == "query":
        for r in retrieve(args.question, args.top_k, args.mode):
            print(f"[{r['score']:.4f}] {r['chunk_id']} (page {r['page']})")
            print("   ", r["text"][:200].replace("\n", " "))
 
    elif args.cmd == "eval":
        evaluate_retrieval(args.top_k, args.mode)
 
    elif args.cmd == "export":
        export_results(args.top_k, args.mode, args.out)
 
 
if __name__ == "__main__":
    main()