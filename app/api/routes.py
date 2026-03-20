from fastapi import APIRouter, HTTPException
import traceback

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chunking import build_chunks_from_pages_jsonl
from app.services.document_ingest import ingest_raw_documents
from app.services.indexing import index_chunks_to_chroma, get_chroma_collection
from app.services.retrieval import answer_question, retrieve_chunks

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "env": settings.app_env,
    }


@router.get("/config-check")
def config_check():
    return {
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "chroma_persist_dir": settings.chroma_persist_dir,
        "chroma_collection_name": settings.chroma_collection_name,
        "data_dir": settings.data_dir,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_batch_size": settings.embedding_batch_size,
        "retrieval_top_k": settings.retrieval_top_k,
        "generation_max_chunks": settings.generation_max_chunks,
        "generation_max_output_tokens": settings.generation_max_output_tokens,
        "has_openai_api_key": bool(settings.openai_api_key),
    }


@router.get("/debug/index-info")
def debug_index_info():
    try:
        collection = get_chroma_collection()
        return {
            "collection_name": settings.chroma_collection_name,
            "count": collection.count(),
            "peek": collection.peek(limit=3),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"index-info failed: {repr(e)}")


@router.post("/ingest")
def ingest_documents():
    try:
        result = ingest_raw_documents()
        return {"status": "success", **result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ingest failed: {repr(e)}")


@router.post("/chunk")
def chunk_documents():
    try:
        result = build_chunks_from_pages_jsonl()
        return {"status": "success", **result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"chunk failed: {repr(e)}")


@router.post("/index")
def index_documents():
    try:
        result = index_chunks_to_chroma()
        return {"status": "success", **result}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"index failed: {repr(e)}")


@router.post("/retrieve")
def retrieve_only(request: ChatRequest):
    try:
        hits = retrieve_chunks(question=request.question, top_k=request.top_k)
        return {
            "question": request.question,
            "count": len(hits),
            "results": [
                {
                    "source_id": hit.source_id,
                    "chunk_id": hit.chunk_id,
                    "document_title": hit.document_title,
                    "file_name": hit.file_name,
                    "page_number": hit.page_number,
                    "distance": hit.distance,
                    "text": hit.text,
                }
                for hit in hits
            ],
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"retrieve failed: {repr(e)}")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return answer_question(question=request.question, top_k=request.top_k)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"chat failed: {repr(e)}")