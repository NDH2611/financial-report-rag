import argparse
import json
import math
import os
import re
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

PAGES_PATH = Path("data/processed/pages.json")
CHUNKS_PATH = Path("data/processed/chunks.json")
INDEX_PATH = Path("indexes/faiss.index")
META_PATH = Path("indexes/meta.json")

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MAX_WORDS = 300    
OVERLAP_LINES = 3
RRF_K = 60

SECTION_RE = re.compile(r"^(Item\s+\d+[A-C]?\.\s*.{0,80})$", re.IGNORECASE)
STATEMENT_RE = re.compile(
    r"^(CONSOLIDATED\s+(STATEMENTS?|BALANCE SHEETS?)[A-Z ’']*|"
    r"Notes to Consolidated Financial Statements)$",
    re.IGNORECASE,
)


# ============================================================
# Chunking
# ============================================================

def _words(line: str) -> int:
    return len(line.split())


FOOTER_PAGE_RE = re.compile(r"Form\s*10-K\s*[|Il!]?\s*(\d{1,3})\s*$", re.IGNORECASE)


def printed_page_numbers(pages: list[dict]) -> dict[int, int]:
    """
    Map PDF page -> printed 10-K page number.

    questions.json evidence_pages use the number printed in the footer
    ("Apple Inc. | 2025 Form 10-K | 29"), which is the PDF page minus 3
    (cover + table of contents). The footer is parsed per page; pages whose
    footer OCR failed fall back to the most common offset. Pages before the
    first numbered page (cover/TOC) map to 0.
    """
    found = {}
    for p in pages:
        for line in reversed(p["text"].splitlines()[-3:]):
            m = FOOTER_PAGE_RE.search(line.strip())
            if m:
                found[p["page"]] = int(m.group(1))
                break

    offsets = Counter(pdf - printed for pdf, printed in found.items())
    offset = offsets.most_common(1)[0][0] if offsets else 0

    return {
        p["page"]: found.get(p["page"], max(p["page"] - offset, 0))
        for p in pages
    }


def strip_footer(text: str) -> str:
    lines = text.splitlines()
    if lines and FOOTER_PAGE_RE.search(lines[-1].strip()):
        lines = lines[:-1]
    return "\n".join(lines)


def chunk_pages(pages: list[dict],
                max_words: int = MAX_WORDS,
                overlap_lines: int = OVERLAP_LINES) -> list[dict]:
    """
    Split OCR'd pages into chunks. Chunks never cross a page boundary, so
    every chunk has a single, exact page number (needed for evidence
    evaluation). `page` is the printed 10-K page; `pdf_page` the PDF page. Lines are kept whole, so table rows are never cut.
    """
    chunks = []
    section = ""
    printed = printed_page_numbers(pages)

    for page in pages:
        lines = [l for l in strip_footer(page["text"]).splitlines() if l.strip()]
        if not lines:
            continue

        start = 0
        part = 0
        while start < len(lines):
            end, words = start, 0
            while end < len(lines) and (words == 0 or words + _words(lines[end]) <= max_words):
                words += _words(lines[end])
                end += 1

            body = lines[start:end]
            for line in body:  # track the most recent heading
                if SECTION_RE.match(line) or STATEMENT_RE.match(line):
                    section = line.strip()

            part += 1
            chunks.append({
                "chunk_id": (
                    f"p{printed[page['page']]:02d}_c{part}"
                    if printed[page["page"]] > 0
                    else f"pdf{page['page']:02d}_c{part}"   # cover / TOC
                ),
                "page": printed[page["page"]],      # printed 10-K page
                "pdf_page": page["page"],
                "section": section,
                "text": "\n".join(body),
            })

            if end >= len(lines):
                break
            start = max(end - overlap_lines, start + 1)

    return chunks


def embedding_text(chunk: dict) -> str:
    """Text that is embedded / BM25-indexed (section gives extra context)."""
    return f"{chunk.get('section', '')}\n{chunk['text']}".strip()


# ============================================================
# Embedders
# ============================================================

class HashingEmbedder:
    """Deterministic bag-of-words embedder. No downloads; for tests only."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts, is_query: bool = False):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, is_query: bool = False):
        if is_query and "bge" in self.model_name.lower():
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        vecs = self.model.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return vecs.astype("float32")


def get_embedder(model_name: str | None = None):
    model_name = model_name or os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    if model_name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)


# ============================================================
# BM25 (small self-contained implementation, no extra dependency)
# ============================================================

def tokenize(text: str) -> list[str]:
    # "416,161" -> "416161" so numbers match however they are written.
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype="float32")
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf), dtype="float32")
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1.0))
            for t in query_tokens:
                f = tf.get(t)
                if f:
                    s[i] += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm)
        return s


# ============================================================
# Index build
# ============================================================

def build_index(pages_path: Path = PAGES_PATH,
                model_name: str | None = None) -> dict:
    import faiss

    model_name = model_name or os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    with open(pages_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")

    embedder = get_embedder(model_name)
    vectors = embedder.encode([embedding_text(c) for c in chunks])

    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine (normalized)
    index.add(vectors)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"embed_model": model_name, "num_chunks": len(chunks),
                   "dim": int(vectors.shape[1])}, f, indent=2)

    print(f"Saved {CHUNKS_PATH}, {INDEX_PATH}, {META_PATH}")
    return {"num_chunks": len(chunks), "dim": int(vectors.shape[1])}


# ============================================================
# Retriever
# ============================================================

class Retriever:
    def __init__(self):
        import faiss

        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.index = faiss.read_index(str(INDEX_PATH))
        # Must use the same model that built the index.
        self.embedder = get_embedder(meta["embed_model"])
        self.bm25 = BM25([tokenize(embedding_text(c)) for c in self.chunks])

    def search(self, question: str, k: int = 8, mode: str = "hybrid") -> list[dict]:
        """
        mode: "dense" | "bm25" | "hybrid" (RRF of both).
        Returns top-k chunks in the format expected by llm.format_context.
        """
        pool = min(len(self.chunks), max(k * 4, 20))

        dense_rank, bm25_rank = [], []

        if mode in ("dense", "hybrid"):
            q = self.embedder.encode([question], is_query=True)
            _, ids = self.index.search(q, pool)
            dense_rank = [int(i) for i in ids[0] if i >= 0]

        if mode in ("bm25", "hybrid"):
            s = self.bm25.scores(tokenize(question))
            bm25_rank = [int(i) for i in np.argsort(-s)[:pool] if s[i] > 0]

        fused = Counter()
        for ranking in (dense_rank, bm25_rank):
            for rank, idx in enumerate(ranking):
                fused[idx] += 1.0 / (RRF_K + rank + 1)

        top = [idx for idx, _ in fused.most_common(k)]

        return [
            {
                "chunk_id": self.chunks[i]["chunk_id"],
                "page": self.chunks[i]["page"],
                "pdf_page": self.chunks[i]["pdf_page"],
                "section": self.chunks[i]["section"],
                "text": self.chunks[i]["text"],
                "score": round(float(fused[i]), 5),
            }
            for i in top
        ]


_retriever = None


def retrieve(question: str, k: int = 8, mode: str = "hybrid") -> list[dict]:
    """Module-level helper: lazy-loads the retriever once."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever.search(question, k=k, mode=mode)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    q = sub.add_parser("query")
    q.add_argument("question")
    q.add_argument("-k", type=int, default=8)
    q.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    args = parser.parse_args()

    if args.cmd == "build":
        build_index()
    else:
        for r in retrieve(args.question, k=args.k, mode=args.mode):
            print(f"[{r['chunk_id']}] page={r['page']} score={r['score']} | {r['section']}")
            print("   " + r["text"][:160].replace("\n", " | "))