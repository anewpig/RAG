from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    document_title: str
    file_name: str
    page_number: int
    quote: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="使用者問題")
    top_k: int = Field(default=5, ge=1, le=10, description="要取回的候選 chunks 數量")


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_count: int = 0