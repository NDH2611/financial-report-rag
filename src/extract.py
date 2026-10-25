
import argparse
import json
import re
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image


# Sửa đường dẫn nếu Tesseract được cài ở vị trí khác.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "apple_2025_10k.pdf"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "chunks.json"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
OCR_DPI = 250



def clean_text(text: str) -> str:
    """Loại đầu trang, URL và chân trang; giữ nội dung báo cáo."""
    text = text.replace("\x00", " ")

    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        # Đầu trang PDF: 10/22/26, 2:24 AM aapl-20250927
        if re.match(
            r"^\d{1,2}/\d{1,2}/\d{2,4},\s*"
            r"\d{1,2}:\d{2}\s*[AP]M\s+aapl-20250927\b",
            line,
            re.IGNORECASE,
        ):
            continue

        # Bỏ URL kể cả khi OCR nhận sai dấu / thành ký tự khác.
        if re.search(r"https?\s*:", line, re.IGNORECASE):
            # Chỉ bỏ khi dòng có vẻ là URL, tránh xóa câu văn bình thường.
            if re.search(
                r"www|sec\s*[./\\]|edgar|\.gov|\.com",
                line,
                re.IGNORECASE,
            ):
                continue

        # Chân trang của báo cáo.
        if re.search(
            r"Apple\s+Inc\.\s*\|\s*2025\s+Form\s+10-K\s*\|\s*\d+",
            line,
            re.IGNORECASE,
        ):
            continue

        line = re.sub(r"[ \t]+", " ", line)
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



def extract_page_text(page) -> tuple[str, str]:
    """Trích xuất text trước; dùng OCR nếu trang không có text."""
    text = clean_text(page.get_text("text"))

    if len(text) >= 30:
        return text, "text"

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(OCR_DPI / 72, OCR_DPI / 72),
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples,
    )

    text = clean_text(
        pytesseract.image_to_string(
            image,
            lang="eng",
            config="--psm 6",
        )
    )

    return text, "ocr"


def find_split_boundary(text: str, limit: int) -> int:
    """
    Tìm vị trí cắt gần limit.
    Ưu tiên xuống dòng, sau đó dấu kết thúc câu, rồi khoảng trắng.
    """
    if limit >= len(text):
        return len(text)

    # Chỉ tìm ranh giới trong vùng gần cuối chunk để tránh chunk quá ngắn.
    min_pos = max(1, int(limit * 0.7))
    candidates = []

    for match in re.finditer(r"\n", text[min_pos:limit]):
        candidates.append(min_pos + match.start() + 1)

    if candidates:
        return candidates[-1]

    # Ưu tiên cắt sau dấu kết thúc câu.
    for match in re.finditer(r'[.!?]["\')\]]?\s+', text[min_pos:limit]):
        candidates.append(min_pos + match.end())

    if candidates:
        return candidates[-1]

    # Cuối cùng cắt tại khoảng trắng để tránh cắt đôi từ.
    boundary = text.rfind(" ", min_pos, limit)

    if boundary > min_pos:
        return boundary

    return limit


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Chia văn bản theo giới hạn ký tự, ưu tiên ranh giới dòng/câu.
    Các chunk có overlap để giữ ngữ cảnh giữa các đoạn.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Cần chunk_size > overlap >= 0.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        limit = min(start + chunk_size, text_length)

        if limit == text_length:
            end = text_length
        else:
            end = find_split_boundary(text, limit)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Tạo overlap, ưu tiên bắt đầu tại ranh giới dòng/khoảng trắng.
        next_start = max(start + 1, end - overlap)

        newline_pos = text.find("\n", next_start, end)
        space_pos = text.find(" ", next_start, end)

        if newline_pos != -1:
            next_start = newline_pos + 1
        elif space_pos != -1:
            next_start = space_pos + 1

        # Bảo đảm vòng lặp luôn tiến về phía trước.
        start = max(start + 1, next_start)

    return chunks


def extract_pdf(input_path: Path) -> list[dict]:
    records = []

    with pymupdf.open(input_path) as pdf:
        total_pages = len(pdf)
        print(f"PDF: {input_path.name} — {total_pages} trang")

        for page_index, page in enumerate(pdf):
            text, method = extract_page_text(page)

            if not text:
                print(
                    f"[WARN] Trang {page_index + 1}: "
                    "không trích xuất được text"
                )
                continue

            page_chunks = split_into_chunks(text)

            for chunk_index, chunk in enumerate(page_chunks):
                records.append({
                    "chunk_id": (
                        f"{input_path.stem}_p{page_index + 1}"
                        f"_c{chunk_index + 1}"
                    ),
                    "text": chunk,
                    "metadata": {
                        "source": input_path.name,
                        "company": "Apple Inc.",
                        "fiscal_year": 2025,
                        "page": page_index + 1,
                        "section": None,
                        "type": "text",
                        "extraction_method": method,
                    },
                })

            print(
                f"[{page_index + 1}/{total_pages}] "
                f"{method.upper()}: {len(text)} ký tự, "
                f"{len(page_chunks)} chunks"
            )

    if not records:
        raise RuntimeError("Không trích xuất được văn bản nào từ PDF.")

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Extract text/OCR from financial report PDFs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Không tìm thấy PDF: {args.input}")

    records = extract_pdf(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    print("\nHoàn tất!")
    print(f"Tổng số chunks: {len(records)}")
    print(f"File output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
