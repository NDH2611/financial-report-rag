import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PDF_PATH = Path("data/raw/apple_2025_10k.pdf")
PAGES_PATH = Path("data/processed/pages.json")

OCR_DPI = 200
TESSERACT_CONFIG = "--psm 6"


HEADER_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}:\d{2}\s*[AP]M\s+aapl-\d+\s*$")
FOOTER_RE = re.compile(r"^(https?://\S+\s*)?\d{1,3}\s*/\s*\d{1,3}\s*$")


def clean_page_text(text: str) -> str:
    """Remove browser header/footer lines and normalise whitespace."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or HEADER_RE.match(line) or FOOTER_RE.match(line):
            continue
        lines.append(re.sub(r"[ \t]+", " ", line))
    return "\n".join(lines)


def _ocr_page(args):
    """Worker: rasterize and OCR one page. Imports are local for pickling."""
    pdf_path, page_index = args

    import pymupdf
    import pytesseract
    from PIL import Image

    with pymupdf.open(pdf_path) as doc:
        pix = doc[page_index].get_pixmap(dpi=OCR_DPI)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    raw = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)

    return page_index + 1, clean_page_text(raw)


def load_pages(path: Path = PAGES_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(pages_by_no: dict, out_path: Path) -> list[dict]:
    pages = [{"page": p, "text": pages_by_no[p]} for p in sorted(pages_by_no)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    return pages


def extract_pages(pdf_path: Path = PDF_PATH,
                  out_path: Path = PAGES_PATH,
                  force: bool = False,
                  workers: int = 4) -> list[dict]:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        n_pages = len(doc)

    done = {}
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        done = {p["page"]: p["text"] for p in load_pages(out_path)}

    todo = [(str(pdf_path), i) for i in range(n_pages) if (i + 1) not in done]
    print(f"{n_pages} pages | cached: {len(done)} | to OCR: {len(todo)}")

    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for page, text in pool.map(_ocr_page, todo):
                done[page] = text
                _save(done, out_path)  # checkpoint: safe to interrupt/resume
                print(f"  OCR page {page}/{n_pages}: {len(text)} chars", flush=True)

    pages = _save(done, out_path)

    print(f"Saved {len(pages)} pages to {out_path}")
    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    extract_pages(force=args.force, workers=args.workers)