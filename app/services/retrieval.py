from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from app.core.config import settings
from app.schemas.chat import ChatResponse, Citation, RetrievedResult
from app.services.embeddings import embed_texts_local
from app.services.indexing import get_chroma_collection


@dataclass
class RetrievedChunk:
    source_id: str
    chunk_id: str
    document_title: str
    file_name: str
    page_number: int
    text: str
    distance: float | None
    metadata: dict[str, Any]


def embed_query(question: str) -> list[float]:
    provider = settings.embedding_provider.lower()

    if provider == "local":
        return embed_texts_local([question])[0]

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing.")

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=[question],
        )
        return response.data[0].embedding

    raise ValueError(
        f"Unsupported embedding provider: {settings.embedding_provider}. "
        "Use 'local' or 'openai'."
    )


def build_where_filter(file_name: str | None = None, doc_id: str | None = None):
    clauses = []

    if file_name:
        clauses.append({"file_name": file_name})
    if doc_id:
        clauses.append({"doc_id": doc_id})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}


def retrieve_chunks(
    question: str,
    top_k: int | None = None,
    file_name: str | None = None,
    doc_id: str | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or settings.retrieval_top_k
    collection = get_chroma_collection()

    query_embedding = embed_query(question)
    where_filter = build_where_filter(file_name=file_name, doc_id=doc_id)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if where_filter:
        query_kwargs["where"] = where_filter

    result = collection.query(**query_kwargs)

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[RetrievedChunk] = []

    for idx, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        if not document or not metadata:
            continue

        hits.append(
            RetrievedChunk(
                source_id=f"S{idx}",
                chunk_id=metadata.get("chunk_id", f"chunk-{idx}"),
                document_title=metadata.get("document_title", "unknown"),
                file_name=metadata.get("file_name", "unknown"),
                page_number=int(metadata.get("page_number", 0)),
                text=document,
                distance=float(distance) if distance is not None else None,
                metadata=metadata,
            )
        )

    return hits


def is_retrieval_confident(hits: list[RetrievedChunk]) -> tuple[bool, str | None]:
    if len(hits) < settings.retrieval_min_results:
        return False, "沒有檢索到足夠結果"

    best = hits[0]
    if best.distance is None:
        return False, "缺少距離資訊"

    if best.distance > settings.retrieval_max_distance:
        return False, f"最佳結果距離過高（distance={best.distance:.4f}）"

    return True, None


def build_citations(hits: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []

    for hit in hits[: settings.generation_max_chunks]:
        quote = hit.text.strip().replace("\n", " ")
        if len(quote) > settings.citation_quote_max_chars:
            quote = quote[: settings.citation_quote_max_chars].rstrip() + "..."

        citations.append(
            Citation(
                source_id=hit.source_id,
                chunk_id=hit.chunk_id,
                document_title=hit.document_title,
                file_name=hit.file_name,
                page_number=hit.page_number,
                quote=quote,
            )
        )

    return citations


def build_retrieval_debug(hits: list[RetrievedChunk]) -> list[RetrievedResult]:
    debug_rows: list[RetrievedResult] = []

    for hit in hits:
        debug_rows.append(
            RetrievedResult(
                source_id=hit.source_id,
                chunk_id=hit.chunk_id,
                document_title=hit.document_title,
                file_name=hit.file_name,
                page_number=hit.page_number,
                distance=hit.distance,
                text=hit.text,
            )
        )

    return debug_rows


def generate_grounded_fallback_answer(question: str, hits: list[RetrievedChunk]) -> str:
    if not hits:
        return "目前知識庫中沒有找到可支持回答的內容。"

    lines = []
    for hit in hits[:2]:
        lines.append(
            f"根據 {hit.document_title} 第 {hit.page_number} 頁的內容，[{hit.source_id}]：{hit.text}"
        )

    return "\n\n".join(lines)


def answer_question(
    question: str,
    top_k: int | None = None,
    file_name: str | None = None,
    doc_id: str | None = None,
) -> ChatResponse:
    hits = retrieve_chunks(
        question=question,
        top_k=top_k,
        file_name=file_name,
        doc_id=doc_id,
    )

    confident, reason = is_retrieval_confident(hits)
    citations = build_citations(hits)
    retrieval_debug = build_retrieval_debug(hits)

    if not confident:
        return ChatResponse(
            answer="目前知識庫沒有足夠證據支持明確回答。",
            citations=citations,
            retrieved_count=len(hits),
            abstained=True,
            abstain_reason=reason,
            retrieval_debug=retrieval_debug,
        )

    answer = generate_grounded_fallback_answer(question, hits)

    return ChatResponse(
        answer=answer,
        citations=citations,
        retrieved_count=len(hits),
        abstained=False,
        abstain_reason=None,
        retrieval_debug=retrieval_debug,
    )