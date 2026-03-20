from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from app.core.config import settings
from app.schemas.chat import ChatResponse, Citation
from app.services.indexing import embed_texts, get_chroma_collection


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


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def retrieve_chunks(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.retrieval_top_k

    openai_client = get_openai_client()
    collection = get_chroma_collection()

    query_embedding = embed_texts(openai_client, [question])[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

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


def format_context_for_llm(hits: list[RetrievedChunk]) -> str:
    selected_hits = hits[: settings.generation_max_chunks]
    blocks: list[str] = []

    for hit in selected_hits:
        block = (
            f"[{hit.source_id}]\n"
            f"Document: {hit.document_title}\n"
            f"File: {hit.file_name}\n"
            f"Page: {hit.page_number}\n"
            f"Chunk ID: {hit.chunk_id}\n"
            f"Content:\n{hit.text}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def build_citations(hits: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []

    for hit in hits[: settings.generation_max_chunks]:
        quote = hit.text.strip().replace("\n", " ")
        if len(quote) > 160:
            quote = quote[:160].rstrip() + "..."

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


def generate_grounded_answer(question: str, hits: list[RetrievedChunk]) -> str:
    if not hits:
        return "目前知識庫中沒有找到可支持回答的內容。"

    context = format_context_for_llm(hits)
    client = get_openai_client()

    instructions = (
        "你是一個只能根據已提供文件內容回答的企業知識助理。\n"
        "規則：\n"
        "1. 只能根據提供的 sources 回答。\n"
        "2. 若證據不足，必須明確說不知道、找不到，不能自行補完。\n"
        "3. 每個關鍵結論後面都要加上來源標記，例如 [S1]、[S2]。\n"
        "4. 只能引用已存在的 source labels。\n"
        "5. 請用繁體中文回答。\n"
        "6. 不要捏造公司政策、日期、流程或門檻。"
    )

    user_input = (
        f"Question:\n{question}\n\n"
        f"Sources:\n{context}\n\n"
        "請根據以上 sources 作答，並在句末附上對應來源標記。"
    )

    response = client.responses.create(
        model=settings.chat_model,
        instructions=instructions,
        input=user_input,
        max_output_tokens=settings.generation_max_output_tokens,
    )

    output_text = getattr(response, "output_text", None)
    if output_text and output_text.strip():
        return output_text.strip()

    return "目前無法生成回答，但已成功完成檢索。"


def answer_question(question: str, top_k: int | None = None) -> ChatResponse:
    hits = retrieve_chunks(question=question, top_k=top_k)
    answer = generate_grounded_answer(question=question, hits=hits)
    citations = build_citations(hits)

    return ChatResponse(
        answer=answer,
        citations=citations,
        retrieved_count=len(hits),
    )