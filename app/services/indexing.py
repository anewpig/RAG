from pathlib import Path

import chromadb
from openai import OpenAI

from app.core.config import settings
from app.schemas.chunk import ChunkDocument
from app.services.embeddings import embed_texts_local


MAX_TOTAL_TOKENS_PER_EMBED_REQUEST = 250000


def load_chunks_from_jsonl(input_path: Path) -> list[ChunkDocument]:
    if not input_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {input_path}")

    chunks: list[ChunkDocument] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(ChunkDocument.model_validate_json(line))
    return chunks


def build_embedding_batches(chunks: list[ChunkDocument]) -> list[list[ChunkDocument]]:
    batches: list[list[ChunkDocument]] = []
    current_batch: list[ChunkDocument] = []
    current_tokens = 0

    for chunk in chunks:
        next_count = chunk.chunk_token_count

        if current_batch and (
            len(current_batch) >= settings.embedding_batch_size
            or current_tokens + next_count > MAX_TOTAL_TOKENS_PER_EMBED_REQUEST
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(chunk)
        current_tokens += next_count

    if current_batch:
        batches.append(current_batch)

    return batches


def embed_texts_openai(texts: list[str]) -> list[list[float]]:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing.")

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    provider = settings.embedding_provider.lower()

    if provider == "local":
        return embed_texts_local(texts)

    if provider == "openai":
        return embed_texts_openai(texts)

    raise ValueError(
        f"Unsupported embedding provider: {settings.embedding_provider}. "
        "Use 'local' or 'openai'."
    )


def get_chroma_collection():
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={
            "project": "policylens-rag",
            "embedding_provider": settings.embedding_provider,
            "openai_embedding_model": settings.openai_embedding_model,
            "local_embedding_model_name": settings.local_embedding_model_name,
        },
    )
    return collection


def index_chunks_to_chroma() -> dict:
    processed_dir = Path(settings.data_dir) / "processed"
    input_path = processed_dir / "chunks.jsonl"

    chunks = load_chunks_from_jsonl(input_path)
    if not chunks:
        raise ValueError("No chunks found in chunks.jsonl")

    collection = get_chroma_collection()
    batches = build_embedding_batches(chunks)

    total_indexed = 0
    for batch in batches:
        texts = [chunk.text for chunk in batch]
        embeddings = embed_texts(texts)

        ids = [chunk.id for chunk in batch]
        documents = [chunk.text for chunk in batch]
        metadatas = [chunk.metadata for chunk in batch]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_indexed += len(batch)

    return {
        "chunk_record_count": len(chunks),
        "indexed_count": total_indexed,
        "batch_count": len(batches),
        "embedding_provider": settings.embedding_provider,
        "openai_embedding_model": settings.openai_embedding_model,
        "local_embedding_model_name": settings.local_embedding_model_name,
        "collection_name": settings.chroma_collection_name,
        "persist_dir": settings.chroma_persist_dir,
    }