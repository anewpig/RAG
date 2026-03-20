import json
from pathlib import Path

import tiktoken

from app.core.config import settings
from app.schemas.chunk import ChunkDocument
from app.schemas.document import PageDocument


ENCODING_NAME = "cl100k_base"
TOKENIZER = tiktoken.get_encoding(ENCODING_NAME)


def load_pages_from_jsonl(input_path: Path) -> list[PageDocument]:
    if not input_path.exists():
        raise FileNotFoundError(f"pages.jsonl not found: {input_path}")

    pages: list[PageDocument] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pages.append(PageDocument.model_validate_json(line))
    return pages


def save_chunks_to_jsonl(chunks: list[ChunkDocument], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")


def chunk_page(
    page: PageDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkDocument]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    token_ids = TOKENIZER.encode(page.text)
    if not token_ids:
        return []

    chunks: list[ChunkDocument] = []
    start = 0
    chunk_index = 1

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_token_ids = token_ids[start:end]
        chunk_text = TOKENIZER.decode(chunk_token_ids).strip()

        if chunk_text:
            chunk_id = f"{page.id}-c{chunk_index:04d}"

            metadata = {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_token_count": len(chunk_token_ids),
                "start_token": start,
                "end_token": end,
                "doc_id": page.doc_id,
                "page_id": page.id,
                "source_path": page.source_path,
                "file_name": page.file_name,
                "document_title": page.document_title,
                "file_type": page.file_type,
                "page_number": page.page_number,
                "total_pages": page.total_pages,
                "language": page.language,
            }

            chunks.append(
                ChunkDocument(
                    id=chunk_id,
                    chunk_index=chunk_index,
                    chunk_token_count=len(chunk_token_ids),
                    start_token=start,
                    end_token=end,
                    doc_id=page.doc_id,
                    page_id=page.id,
                    source_path=page.source_path,
                    file_name=page.file_name,
                    document_title=page.document_title,
                    file_type=page.file_type,
                    page_number=page.page_number,
                    total_pages=page.total_pages,
                    language=page.language,
                    text=chunk_text,
                    metadata=metadata,
                )
            )
            chunk_index += 1

        if end == len(token_ids):
            break

        start = end - chunk_overlap

    return chunks


def build_chunks_from_pages_jsonl() -> dict:
    processed_dir = Path(settings.data_dir) / "processed"
    input_path = processed_dir / "pages.jsonl"
    output_path = processed_dir / "chunks.jsonl"

    pages = load_pages_from_jsonl(input_path)

    all_chunks: list[ChunkDocument] = []
    for page in pages:
        page_chunks = chunk_page(
            page=page,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        all_chunks.extend(page_chunks)

    save_chunks_to_jsonl(all_chunks, output_path)

    return {
        "page_record_count": len(pages),
        "chunk_record_count": len(all_chunks),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "output_path": str(output_path),
    }