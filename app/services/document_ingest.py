import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pymupdf

from app.core.config import settings
from app.schemas.document import PageDocument


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def slugify_filename(file_path: Path) -> str:
    stem = file_path.stem.strip().lower()
    stem = re.sub(r"[^\w\s-]", "", stem)
    stem = re.sub(r"[-\s]+", "-", stem)
    return stem or "document"


def infer_language(text: str) -> str:
    if not text.strip():
        return "unknown"

    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))

    if zh_chars > 0 and en_chars > 0:
        return "mixed"
    if zh_chars > 0:
        return "zh"
    if en_chars > 0:
        return "en"
    return "unknown"


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = text.replace("\t", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_inline_page_artifacts(text: str) -> str:
    """
    移除常見頁碼型噪音，例如：
    - Page 3 of 10
    - 第 3 頁
    - 3 / 10
    """
    lines = [line.strip() for line in text.splitlines()]
    cleaned = []

    patterns = [
        r"^page\s+\d+(\s+of\s+\d+)?$",
        r"^第\s*\d+\s*頁$",
        r"^\d+\s*/\s*\d+$",
    ]

    for line in lines:
        lower_line = line.lower()
        if any(re.match(p, lower_line) for p in patterns):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def merge_broken_lines(text: str) -> str:
    """
    把 PDF 常見的硬斷行整理得稍微自然一點：
    - 空白行視為段落分隔
    - 同段內的換行合併成空白
    """
    paragraphs = []
    current = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs).strip()


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = remove_inline_page_artifacts(text)
    text = merge_broken_lines(text)
    text = normalize_whitespace(text)
    return text


def extract_candidate_header_footer(lines: list[str]) -> tuple[str | None, str | None]:
    non_empty = [line.strip() for line in lines if line.strip()]
    if not non_empty:
        return None, None

    header = non_empty[0]
    footer = non_empty[-1]
    return header, footer


def detect_repeated_headers_footers(raw_page_texts: list[str]) -> tuple[set[str], set[str]]:
    """
    很簡化但很實用的 heuristic：
    統計每頁第一行 / 最後一行，若同一內容在多頁重複出現，就視為 header/footer。
    """
    headers = []
    footers = []

    for text in raw_page_texts:
        lines = text.splitlines()
        header, footer = extract_candidate_header_footer(lines)
        if header:
            headers.append(header)
        if footer:
            footers.append(footer)

    header_counts = Counter(headers)
    footer_counts = Counter(footers)

    repeated_headers = {line for line, count in header_counts.items() if count >= 2}
    repeated_footers = {line for line, count in footer_counts.items() if count >= 2}

    return repeated_headers, repeated_footers


def remove_detected_headers_footers(
    text: str,
    repeated_headers: set[str],
    repeated_footers: set[str],
) -> str:
    lines = text.splitlines()
    stripped_lines = [line.strip() for line in lines if line.strip()]

    if not stripped_lines:
        return text.strip()

    start_idx = 0
    end_idx = len(lines)

    first_non_empty = None
    last_non_empty = None

    for i, line in enumerate(lines):
        if line.strip():
            first_non_empty = (i, line.strip())
            break

    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_non_empty = (i, lines[i].strip())
            break

    if first_non_empty and first_non_empty[1] in repeated_headers:
        start_idx = first_non_empty[0] + 1

    if last_non_empty and last_non_empty[1] in repeated_footers:
        end_idx = last_non_empty[0]

    return "\n".join(lines[start_idx:end_idx]).strip()


def read_pdf_pages(file_path: Path) -> list[PageDocument]:
    with pymupdf.open(file_path) as doc:
        raw_page_texts = [page.get_text("text", sort=True) for page in doc]
        repeated_headers, repeated_footers = detect_repeated_headers_footers(raw_page_texts)

        total_pages = len(raw_page_texts)
        doc_id = slugify_filename(file_path)
        document_title = file_path.stem

        results: list[PageDocument] = []

        for page_index, raw_text in enumerate(raw_page_texts, start=1):
            text = remove_detected_headers_footers(
                raw_text,
                repeated_headers=repeated_headers,
                repeated_footers=repeated_footers,
            )
            text = clean_text(text)

            if not text:
                continue

            language = infer_language(text)
            page_id = f"{doc_id}-p{page_index:04d}"

            metadata = {
                "doc_id": doc_id,
                "source_path": str(file_path),
                "file_name": file_path.name,
                "document_title": document_title,
                "file_type": "pdf",
                "page_number": page_index,
                "total_pages": total_pages,
                "language": language,
            }

            results.append(
                PageDocument(
                    id=page_id,
                    doc_id=doc_id,
                    source_path=str(file_path),
                    file_name=file_path.name,
                    document_title=document_title,
                    file_type="pdf",
                    page_number=page_index,
                    total_pages=total_pages,
                    language=language,
                    text=text,
                    metadata=metadata,
                )
            )

        return results


def read_text_like_file(file_path: Path) -> list[PageDocument]:
    raw_text = file_path.read_text(encoding="utf-8")
    text = clean_text(raw_text)

    if not text:
        return []

    doc_id = slugify_filename(file_path)
    document_title = file_path.stem
    language = infer_language(text)

    metadata = {
        "doc_id": doc_id,
        "source_path": str(file_path),
        "file_name": file_path.name,
        "document_title": document_title,
        "file_type": file_path.suffix.lstrip(".").lower(),
        "page_number": 1,
        "total_pages": 1,
        "language": language,
    }

    return [
        PageDocument(
            id=f"{doc_id}-p0001",
            doc_id=doc_id,
            source_path=str(file_path),
            file_name=file_path.name,
            document_title=document_title,
            file_type=file_path.suffix.lstrip(".").lower(),
            page_number=1,
            total_pages=1,
            language=language,
            text=text,
            metadata=metadata,
        )
    ]


def read_file(file_path: Path) -> list[PageDocument]:
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        return read_pdf_pages(file_path)

    return read_text_like_file(file_path)


def iter_raw_files(raw_dir: Path) -> Iterable[Path]:
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def save_pages_to_jsonl(pages: list[PageDocument], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for page in pages:
            f.write(page.model_dump_json(ensure_ascii=False) + "\n")


def ingest_raw_documents() -> dict:
    raw_dir = Path(settings.data_dir) / "raw"
    processed_dir = Path(settings.data_dir) / "processed"
    output_path = processed_dir / "pages.jsonl"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    all_pages: list[PageDocument] = []
    ingested_files = []

    for file_path in iter_raw_files(raw_dir):
        pages = read_file(file_path)
        all_pages.extend(pages)
        ingested_files.append(str(file_path))

    save_pages_to_jsonl(all_pages, output_path)

    return {
        "ingested_file_count": len(ingested_files),
        "page_record_count": len(all_pages),
        "output_path": str(output_path),
        "files": ingested_files,
    }