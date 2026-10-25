from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

EXTRACT_PATH = ROOT / "src" / "extract.py"
RETRIEVAL_PATH = ROOT / "src" / "retrieval.py"


# ============================================================
# Helpers
# ============================================================

def replace_once(text: str, old: str, new: str, description: str) -> str:
    if old not in text:
        raise RuntimeError(
            f"Không tìm thấy đoạn cần sửa: {description}\n"
            "Có thể file hiện tại khác với phiên bản đã gửi cho ChatGPT."
        )

    return text.replace(old, new, 1)


# ============================================================
# Patch extract.py
# ============================================================

def patch_extract():
    text = EXTRACT_PATH.read_text(encoding="utf-8")

    # --------------------------------------------------------
    # 1. Add report-page convention
    # --------------------------------------------------------

    old = """OCR_DPI = 250
"""

    new = """OCR_DPI = 250

# Apple 2025 Form 10-K page convention:
# physical PDF page 4 == printed report page 1.
#
# We preserve both page systems:
#   pdf_page = physical page in the PDF file
#   page     = printed page shown inside the Form 10-K
#
# questions.json / evidence_pages use the printed report page.
REPORT_PAGE_OFFSET = 3


def pdf_page_to_report_page(pdf_page: int) -> int | None:
    \"\"\"Convert physical PDF page to printed Form 10-K page.\"\"\"

    if pdf_page <= REPORT_PAGE_OFFSET:
        return None

    return pdf_page - REPORT_PAGE_OFFSET
"""

    text = replace_once(
        text,
        old,
        new,
        "thêm REPORT_PAGE_OFFSET",
    )

    # --------------------------------------------------------
    # 2. Compute both page systems for every PDF page
    # --------------------------------------------------------

    old = """        for page_index, page in enumerate(pdf):
            page_text, page_tables, method = extract_page_segments(page)
            page_chunks = split_into_chunks(page_text)
"""

    new = """        for page_index, page in enumerate(pdf):
            pdf_page = page_index + 1
            report_page = pdf_page_to_report_page(pdf_page)

            page_text, page_tables, method = extract_page_segments(page)
            page_chunks = split_into_chunks(page_text)
"""

    text = replace_once(
        text,
        old,
        new,
        "tạo pdf_page và report_page",
    )

    # --------------------------------------------------------
    # 3. Text chunk metadata
    # --------------------------------------------------------

    old = '''                        "page": page_index + 1,
                        "section": section,
'''

    new = '''                        "pdf_page": pdf_page,
                        "page": report_page,
                        "section": section,
'''

    text = replace_once(
        text,
        old,
        new,
        "metadata của text chunk",
    )

    # --------------------------------------------------------
    # 4. Table chunk metadata
    # --------------------------------------------------------

    old = '''                            "page": page_index + 1,
                            "section": section or "financial_statements",
'''

    new = '''                            "pdf_page": pdf_page,
                            "page": report_page,
                            "section": section or "financial_statements",
'''

    text = replace_once(
        text,
        old,
        new,
        "metadata của table chunk",
    )

    EXTRACT_PATH.write_text(
        text,
        encoding="utf-8",
    )

    print("[OK] Patched src/extract.py")


# ============================================================
# Patch retrieval.py
# ============================================================

def patch_retrieval():
    text = RETRIEVAL_PATH.read_text(encoding="utf-8")

    # --------------------------------------------------------
    # Replace complete load_chunks()
    # --------------------------------------------------------

    start_marker = "def load_chunks("
    end_marker = (
        "\n\n# ============================================================\n"
        "# Embedding"
    )

    start = text.find(start_marker)

    if start == -1:
        raise RuntimeError(
            "Không tìm thấy load_chunks() trong retrieval.py"
        )

    end = text.find(
        end_marker,
        start,
    )

    if end == -1:
        raise RuntimeError(
            "Không tìm thấy điểm kết thúc load_chunks()."
        )

    new_load_chunks = '''def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
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
'''

    text = (
        text[:start]
        + new_load_chunks
        + text[end:]
    )

    # --------------------------------------------------------
    # Make Recall@K ignore front-matter page=None
    # --------------------------------------------------------

    old = '''        retrieved_pages = {
            chunk["page"]
            for chunk in contexts
        }'''

    new = '''        retrieved_pages = {
            chunk["page"]
            for chunk in contexts
            if chunk.get("page") is not None
        }'''

    if old in text:
        text = text.replace(
            old,
            new,
            1,
        )

    # --------------------------------------------------------
    # Improve query CLI output
    # --------------------------------------------------------

    old = '''                f"{result['chunk_id']} "
                f"(page {result['page']})"
            )'''

    new = '''                f"{result['chunk_id']} "
                f"(report page {result['page']}, "
                f"PDF page {result.get('pdf_page')})"
            )'''

    if old in text:
        text = text.replace(
            old,
            new,
            1,
        )

    RETRIEVAL_PATH.write_text(
        text,
        encoding="utf-8",
    )

    print("[OK] Patched src/retrieval.py")


# ============================================================
# Main
# ============================================================

def main():

    if not EXTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {EXTRACT_PATH}"
        )

    if not RETRIEVAL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {RETRIEVAL_PATH}"
        )

    print("Patching page convention...")
    print()

    patch_extract()
    patch_retrieval()

    print()
    print("Hoàn tất.")
    print()
    print("Page convention:")
    print("  PDF page 32 -> report page 29")
    print("  PDF page 34 -> report page 31")
    print("  PDF page 36 -> report page 33")
    print()
    print(
        "Tiếp theo chạy: python src/extract.py"
    )


if __name__ == "__main__":
    main()