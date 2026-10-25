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
TABLE_CHUNK_SIZE = 3600
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

        # Chỉ loại dòng header PDF của báo cáo Apple
        if re.match(
            r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+"
            r"\d{1,2}:\d{2}\s*[AP]\.?M\.?\s*"
            r"(?:\|\s*)?aapl-20250927\s*$",
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


def classify_chunk(text: str, current_section: str | None = None) -> tuple[str | None, str]:
    """Phân loại section và type của một chunk trong báo cáo tài chính."""
    normalized = re.sub(r"\s+", " ", text).strip()
    upper = normalized.upper()

    # Nhận diện tiêu đề section của báo cáo.
    section_patterns = [
        (r"\bITEM\s+1A\.?\s+RISK FACTORS\b", "risk_factors"),
        (r"\bITEM\s+7\.?\s+MANAGEMENT['’]?S DISCUSSION", "management_discussion"),
        (r"\bITEM\s+8\.?\s+FINANCIAL STATEMENTS", "financial_statements"),
        (r"\bNOTES TO CONSOLIDATED FINANCIAL STATEMENTS\b", "notes"),
        (r"\bBUSINESS\b", "business"),
    ]

    detected_section = None
    for pattern, section_name in section_patterns:
        if re.search(pattern, upper):
            detected_section = section_name
            break

    section = detected_section or current_section

    # Nhận diện tiêu đề của các bảng tài chính.
    table_headings = [
        "CONSOLIDATED STATEMENTS OF OPERATIONS",
        "CONSOLIDATED BALANCE SHEETS",
        "CONSOLIDATED STATEMENTS OF CASH FLOWS",
        "CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME",
        "CONSOLIDATED STATEMENTS OF SHAREHOLDERS’ EQUITY",
    ]

    if any(heading in upper for heading in table_headings):
        return "financial_statements", "financial_table"

    # Nhận diện tiêu đề độc lập hoặc gần như độc lập.
    if re.match(r"^(ITEM\s+\d+[A-Z]?\.?|NOTE\s+\d+\s*[—-]?)\s+", upper):
        return section, "heading"

    # Nếu chưa đủ căn cứ thì giữ section hiện tại và loại text.
    return section, "text"


def _boxes_overlap(a, b, margin: float = 2.0) -> bool:
    """Kiểm tra hai bounding box có giao nhau hay không."""
    return not (
        a[2] <= b[0] + margin
        or a[0] >= b[2] - margin
        or a[3] <= b[1] + margin
        or a[1] >= b[3] - margin
    )


def _normalize_table_rows(rows) -> tuple[str, int, int]:
    """Chuẩn hóa kết quả extract() và trả về text, số hàng, số hàng có >=2 ô."""
    rendered_rows = []
    multi_cell_rows = 0

    for row in rows or []:
        cells = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in (row or [])]
        while cells and not cells[-1]:
            cells.pop()
        if not any(cells):
            continue
        if sum(bool(cell) for cell in cells) >= 2:
            multi_cell_rows += 1
        rendered_rows.append(" | ".join(cells))

    return "\n".join(rendered_rows).strip(), len(rendered_rows), multi_cell_rows


def _table_quality(table_text: str, row_count: int, multi_cell_rows: int) -> bool:
    """Lọc các kết quả find_tables() yếu, chẳng hạn một đoạn văn bị hiểu nhầm là bảng."""
    if not table_text or row_count < 3 or multi_cell_rows < 2:
        return False
    if len(table_text) < 20:
        return False
    return True


def _find_tables_with_strategies(page) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Thử nhiều chiến lược vì bảng 10-K thường chỉ có một vài đường kẻ."""
    strategies = [
        # Bảng có đường kẻ ngang/dọc.
        {"strategy": "lines"},
        {"strategy": "lines_strict"},
        # Bảng ít hoặc không có đường kẻ: suy luận cột/hàng từ tọa độ chữ.
        {
            "strategy": "text",
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
            "text_x_tolerance": 5,
            "text_y_tolerance": 4,
        },
        # Thử riêng các kiểu hỗn hợp: có cột căn bằng chữ nhưng hàng có đường kẻ,
        # hoặc ngược lại.
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "lines",
            "min_words_vertical": 2,
            "text_x_tolerance": 5,
            "text_y_tolerance": 4,
        },
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "text",
            "min_words_horizontal": 1,
            "text_x_tolerance": 5,
            "text_y_tolerance": 4,
        },
    ]

    found = []
    positioned_lines = _positioned_lines(page)
    table_line_ys = [line["y"] for line in positioned_lines if line.get("table_line")]
    for options in strategies:
        try:
            finder = page.find_tables(**options)
        except (AttributeError, TypeError, RuntimeError, ValueError):
            continue

        for table in getattr(finder, "tables", []):
            try:
                table_text, row_count, multi_cell_rows = _normalize_table_rows(table.extract())
                if not _table_quality(table_text, row_count, multi_cell_rows):
                    continue
                bbox = tuple(float(value) for value in table.bbox)

                # Bỏ các ứng viên quá rộng do strategy="text" đôi khi gom cả tiêu đề/
                # đoạn văn ở phía trên thành một bảng. Với báo cáo tài chính, vùng bảng
                # thường gần các dòng có số liệu căn cột; nếu chênh lệch quá lớn thì coi
                # đó là phát hiện giả và để fallback theo tọa độ xử lý.
                nearby_numeric_rows = [
                    y for y in table_line_ys
                    if bbox[1] - 4 <= y <= bbox[3] + 4
                ]
                if not nearby_numeric_rows:
                    continue
                if min(nearby_numeric_rows) - bbox[1] > 180:
                    continue

                # Tránh thêm cùng một bảng nhiều lần khi nhiều strategy cùng phát hiện.
                duplicate = False
                for old_text, old_bbox in found:
                    if _boxes_overlap(bbox, old_bbox, margin=4.0):
                        duplicate = True
                        # Nếu bảng mới có cấu trúc tốt hơn/nhiều nội dung hơn, thay bản cũ.
                        if len(table_text) > len(old_text) * 1.15:
                            found[found.index((old_text, old_bbox))] = (table_text, bbox)
                        break
                if not duplicate:
                    found.append((table_text, bbox))
            except (AttributeError, RuntimeError, ValueError, TypeError):
                continue

    # Trả theo thứ tự từ trên xuống dưới.
    found.sort(key=lambda item: (item[1][1], item[1][0]))
    return found


def _positioned_lines(page) -> list[dict]:
    """Gom từ theo tọa độ Y để giữ hàng ngay cả khi PDF chia cột thành nhiều block."""
    words = page.get_text("words", sort=True)
    if not words:
        return []

    # Mỗi từ: x0, y0, x1, y1, text, block_no, line_no, word_no.
    word_items = sorted(
        [w for w in words if len(w) >= 5 and str(w[4]).strip()],
        key=lambda w: ((float(w[1]) + float(w[3])) / 2, float(w[0])),
    )
    lines = []
    for word in word_items:
        x0, y0, x1, y1 = map(float, word[:4])
        word_text = str(word[4]).strip()
        y_center = (y0 + y1) / 2

        # Ghép vào baseline gần nhất trong khoảng 3.5 points.
        target = None
        for line in reversed(lines[-5:]):
            if abs(line["y"] - y_center) <= 3.5:
                target = line
                break
            if line["y"] < y_center - 3.5:
                break

        if target is None:
            target = {"y": y_center, "words": []}
            lines.append(target)
        target["words"].append((x0, y0, x1, y1, word_text))
        target["y"] = sum((w[1] + w[3]) / 2 for w in target["words"]) / len(target["words"])

    for line in lines:
        line["words"].sort(key=lambda w: w[0])
        line["x0"] = min(w[0] for w in line["words"])
        line["x1"] = max(w[2] for w in line["words"])
        line["text"] = " ".join(w[4] for w in line["words"])
        # Tách cell tại khoảng trống lớn; ngưỡng có chủ đích để giữ cụm từ trong cùng ô.
        cells = []
        current = []
        previous_x1 = None
        for word in line["words"]:
            if current and previous_x1 is not None and word[0] - previous_x1 >= 26:
                cells.append(" ".join(w[4] for w in current))
                current = []
            current.append(word)
            previous_x1 = word[2]
        if current:
            cells.append(" ".join(w[4] for w in current))
        line["cells"] = cells
        line["table_line"] = (
            len(cells) >= 2
            and any(re.search(r"\d|\$|%|\u2014|\u2013", w[4]) for w in line["words"])
            and any(
                line["words"][i + 1][0] - line["words"][i][2] >= 38
                for i in range(len(line["words"]) - 1)
            )
        )

    lines.sort(key=lambda line: line["y"])
    return lines


def _fallback_tables_from_positions(page) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Phát hiện bảng tài chính không có viền bằng khoảng trắng và vị trí số liệu.

    Đây là phương án dự phòng cho các bảng như share repurchase activity: mô tả ở
    bên trái, các số liệu căn thành nhiều cột ở bên phải, nhưng không có lưới đầy đủ.
    """
    lines = _positioned_lines(page)
    hits = [i for i, line in enumerate(lines) if line["table_line"]]
    if len(hits) < 2:
        return []

    # Gom các dòng có nhiều cột số liệu gần nhau thành một vùng bảng.
    groups = []
    current = [hits[0]]
    for idx in hits[1:]:
        if lines[idx]["y"] - lines[current[-1]]["y"] <= 90:
            current.append(idx)
        else:
            if len(current) >= 2:
                groups.append(current)
            current = [idx]
    if len(current) >= 2:
        groups.append(current)

    results = []
    page_rect = page.rect
    for group in groups:
        first_y = lines[group[0]]["y"]
        last_y = lines[group[-1]]["y"]
        group_lines = [lines[i] for i in group]

        # Mở rộng lên trên để giữ phần tiêu đề nhiều dòng của bảng và xuống dưới
        # để giữ dòng Total; không mở rộng quá mức sang đoạn văn kế tiếp.
        top = max(float(page_rect.y0), first_y - 165)
        bottom = min(float(page_rect.y1), last_y + 24)
        x0 = max(float(page_rect.x0), min(line["x0"] for line in group_lines) - 14)
        x1 = min(float(page_rect.x1), max(line["x1"] for line in group_lines) + 14)

        candidate_lines = [
            line for line in lines
            if top <= line["y"] <= bottom
            and line["x1"] >= x0
            and line["x0"] <= x1
        ]
        rendered = []
        for line in candidate_lines:
            cells = [re.sub(r"\s+", " ", cell).strip() for cell in line["cells"]]
            cells = [cell for cell in cells if cell]
            # Giữ các khoảng cách cột thành dấu |; các dòng văn bản liền mạch vẫn là 1 ô.
            rendered.append(" | ".join(cells) if len(cells) > 1 else line["text"])

        table_text = "\n".join(row for row in rendered if row.strip()).strip()
        if table_text and len(group_lines) >= 2:
            bbox = (x0, top, x1, bottom)
            if not any(_boxes_overlap(bbox, old_bbox, margin=8.0) for _, old_bbox in results):
                results.append((table_text, bbox))

    results.sort(key=lambda item: (item[1][1], item[1][0]))
    return results


def _extract_ocr_with_positions(page) -> tuple[str, list[tuple[str, tuple[float, float, float, float]]]]:
    """OCR có tọa độ để còn phân biệt vùng bảng trên PDF scan."""
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(OCR_DPI / 72, OCR_DPI / 72),
        alpha=False,
    )
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    scale = 72 / OCR_DPI
    words = []
    for i, raw in enumerate(data.get("text", [])):
        word = str(raw).strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError, KeyError):
            conf = -1
        if conf < 0:
            continue
        x0 = float(data["left"][i]) * scale
        y0 = float(data["top"][i]) * scale
        x1 = (float(data["left"][i]) + float(data["width"][i])) * scale
        y1 = (float(data["top"][i]) + float(data["height"][i])) * scale
        words.append((x0, y0, x1, y1, word, 0, int(data["line_num"][i]), i))

    # Reuse the positional line grouper's logic by creating a lightweight proxy.
    # Keep OCR output as normal text when there is not enough evidence of a table.
    lines = []
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    for w in words:
        yc = (w[1] + w[3]) / 2
        line = next((ln for ln in reversed(lines[-5:]) if abs(ln["y"] - yc) <= 4.0), None)
        if line is None:
            line = {"y": yc, "words": []}
            lines.append(line)
        line["words"].append(w[:5])
        line["y"] = sum((item[1] + item[3]) / 2 for item in line["words"]) / len(line["words"])

    for line in lines:
        line["words"].sort(key=lambda w: w[0])
        line["x0"] = min(w[0] for w in line["words"])
        line["x1"] = max(w[2] for w in line["words"])
        line["text"] = " ".join(w[4] for w in line["words"])
        cells, current, prev_x1 = [], [], None
        for word in line["words"]:
            if current and prev_x1 is not None and word[0] - prev_x1 >= 26:
                cells.append(" ".join(w[4] for w in current))
                current = []
            current.append(word)
            prev_x1 = word[2]
        if current:
            cells.append(" ".join(w[4] for w in current))
        line["cells"] = cells
        line["table_line"] = (
            len(cells) >= 2
            and any(re.search(r"\d|\$|%", w[4]) for w in line["words"])
            and any(line["words"][i + 1][0] - line["words"][i][2] >= 38
                    for i in range(len(line["words"]) - 1))
        )

    # Ghép các dòng OCR thành plain text trước.
    page_text = clean_text("\n".join(line["text"] for line in lines))
    hit_indexes = [i for i, line in enumerate(lines) if line["table_line"]]
    groups, current = [], []
    for idx in hit_indexes:
        if current and lines[idx]["y"] - lines[current[-1]]["y"] > 90:
            if len(current) >= 2:
                groups.append(current)
            current = []
        current.append(idx)
    if len(current) >= 2:
        groups.append(current)

    table_results = []
    for group in groups:
        first_y, last_y = lines[group[0]]["y"], lines[group[-1]]["y"]
        top = max(0.0, first_y - 165)
        bottom = min(float(page.rect.height), last_y + 24)
        x0 = max(0.0, min(lines[i]["x0"] for i in group) - 14)
        x1 = min(float(page.rect.width), max(lines[i]["x1"] for i in group) + 14)
        selected = [ln for ln in lines if top <= ln["y"] <= bottom and ln["x1"] >= x0 and ln["x0"] <= x1]
        table_text = "\n".join(
            " | ".join(ln["cells"]) if len(ln["cells"]) > 1 else ln["text"]
            for ln in selected
        ).strip()
        if table_text:
            table_results.append((table_text, (x0, top, x1, bottom)))

    # Nếu đã nhận diện được bảng OCR, loại các dòng thuộc vùng bảng khỏi text chunk
    # để dữ liệu không xuất hiện lặp ở cả type=text và type=table.
    if table_results:
        outside_lines = []
        for line in lines:
            center_x = (line["x0"] + line["x1"]) / 2
            center_y = line["y"]
            in_table = any(
                bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]
                for _, bbox in table_results
            )
            if not in_table:
                outside_lines.append(line["text"])
        page_text = clean_text("\n".join(outside_lines))

    return page_text, table_results


def extract_page_segments(page) -> tuple[str, list[str], str]:
    """Trích riêng text và table, thử nhiều phương pháp nhận diện trước khi fallback."""
    raw_text = clean_text(page.get_text("text"))
    method = "text"
    table_candidates = []

    if len(raw_text) >= 30:
        # Thử PyMuPDF theo cả đường kẻ và bố cục chữ.
        table_candidates = _find_tables_with_strategies(page)
        if not table_candidates:
            table_candidates = _fallback_tables_from_positions(page)
    else:
        # OCR không chỉ lấy chuỗi chữ mà còn giữ tọa độ từng từ để tìm các cột số liệu.
        raw_text, table_candidates = _extract_ocr_with_positions(page)
        method = "ocr"

    # Chuẩn hóa và bỏ bảng trùng do các chiến lược khác nhau phát hiện.
    deduped = []
    for table_text, bbox in table_candidates:
        table_text = table_text.strip()
        if not table_text:
            continue
        if any(_boxes_overlap(bbox, old_bbox, margin=6.0) for _, old_bbox in deduped):
            continue
        deduped.append((table_text, bbox))

    # Khi không có bảng nào được tìm thấy, giữ nguyên trang là text để không mất dữ liệu.
    if not deduped:
        return raw_text, [], method

    # Lọc theo từng dòng/span và tọa độ, thay vì bỏ nguyên block lớn vốn có thể chứa
    # cả bảng lẫn văn bản bên ngoài bảng.
    text_lines = []
    page_dict = page.get_text("dict") if method == "text" else None
    if page_dict:
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                parts = []
                for span in line.get("spans", []):
                    value = str(span.get("text", "")).strip()
                    if not value:
                        continue
                    bbox = tuple(float(v) for v in span.get("bbox", (0, 0, 0, 0)))
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    inside = any(
                        b[0] <= center_x <= b[2] and b[1] <= center_y <= b[3]
                        for _, b in deduped
                    )
                    if not inside:
                        parts.append(value)
                if parts:
                    text_lines.append(" ".join(parts))
        page_text = clean_text("\n".join(text_lines))
    else:
        # Với OCR, _extract_ocr_with_positions() đã loại các dòng trong vùng bảng.
        page_text = raw_text

    tables = [table_text for table_text, _ in deduped]
    return page_text, tables, method

def split_table_into_chunks(
    table_text: str,
    chunk_size: int = TABLE_CHUNK_SIZE,
) -> list[str]:
    """Chia bảng theo hàng, không cắt một hàng thành nhiều chunk nếu có thể."""
    rows = [row.strip() for row in table_text.splitlines() if row.strip()]
    if not rows:
        return []

    chunks = []
    current_rows = []
    current_length = 0

    for row in rows:
        # Nếu một hàng dài hơn chunk_size, tách riêng hàng đó theo hàm text.
        if len(row) > chunk_size:
            if current_rows:
                chunks.append("\n".join(current_rows))
                current_rows, current_length = [], 0
            chunks.extend(split_into_chunks(row, chunk_size, CHUNK_OVERLAP))
            continue

        extra_length = len(row) + (1 if current_rows else 0)
        if current_rows and current_length + extra_length > chunk_size:
            chunks.append("\n".join(current_rows))
            # Lặp lại hàng đầu tiên (thường là header) để giữ ngữ cảnh bảng.
            header = rows[0]
            current_rows = [header] if row != header and len(header) < chunk_size else []
            current_length = len(header) if current_rows else 0

        if current_rows and current_rows[-1] == row:
            continue
        current_rows.append(row)
        current_length += len(row) + (1 if len(current_rows) > 1 else 0)

    if current_rows:
        chunks.append("\n".join(current_rows))

    return chunks


def extract_pdf(input_path: Path) -> list[dict]:
    records = []

    with pymupdf.open(input_path) as pdf:
        total_pages = len(pdf)
        print(f"PDF: {input_path.name} — {total_pages} trang")

        for page_index, page in enumerate(pdf):
            page_text, page_tables, method = extract_page_segments(page)
            page_chunks = split_into_chunks(page_text)

            chunk_index = 0

            for chunk in page_chunks:
                section, _ = classify_chunk(chunk)
                records.append({
                    "chunk_id": (
                        f"{input_path.stem}_p{page_index + 1}"
                        f"_c{chunk_index + 1}"
                    ),
                    "text": chunk,
                    "metadata": {
                        "company": "Apple Inc.",
                        "fiscal_year": 2025,
                        "page": page_index + 1,
                        "section": section,
                        "type": "text",
                        "extraction_method": method,
                    },
                })
                chunk_index += 1

            for table_index, table_text in enumerate(page_tables, start=1):
                table_chunks = split_table_into_chunks(table_text)
                for table_chunk_index, chunk in enumerate(table_chunks, start=1):
                    # Làm sạch lần cuối trước khi lưu chunk bảng
                    chunk = clean_text(chunk)
                    if not chunk:
                        continue
                    section, _ = classify_chunk(chunk)
                    records.append({
                        "chunk_id": (
                            f"{input_path.stem}_p{page_index + 1}"
                            f"_t{table_index}_c{table_chunk_index}"
                        ),
                        "text": chunk,
                        "metadata": {
                            "company": "Apple Inc.",
                            "fiscal_year": 2025,
                            "page": page_index + 1,
                            "section": section or "financial_statements",
                            "type": "table",
                            "table_id": (
                                f"{input_path.stem}_p{page_index + 1}"
                                f"_t{table_index}"
                            ),
                            "extraction_method": method,
                        },
                    })

            if not page_text and not page_tables:
                print(
                    f"[WARN] Trang {page_index + 1}: "
                    "không trích xuất được text hoặc bảng"
                )

            print(
                f"[{page_index + 1}/{total_pages}] "
                f"{method.upper()}: {len(page_text)} ký tự text, "
                f"{len(page_chunks)} text chunks, "
                f"{len(page_tables)} bảng"
            )

    if not records:
        raise RuntimeError("Không trích xuất được nội dung nào từ PDF.")

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
