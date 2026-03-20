from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    document_title: str
    file_name: str
    page_number: int
    quote: str


class RetrievedResult(BaseModel):
    source_id: str
    chunk_id: str
    document_title: str
    file_name: str
    page_number: int
    distance: float | None = None
    text: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="使用者問題")
    top_k: int = Field(default=5, ge=1, le=10)
    file_name: str | None = None
    doc_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_count: int = 0
    abstained: bool = False
    abstain_reason: str | None = None
    retrieval_debug: list[RetrievedResult] = Field(default_factory=list)