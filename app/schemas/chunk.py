from pydantic import BaseModel, Field


class ChunkDocument(BaseModel):
    id: str = Field(..., description="唯一 chunk ID")
    chunk_index: int
    chunk_token_count: int
    start_token: int
    end_token: int

    doc_id: str
    page_id: str
    source_path: str
    file_name: str
    document_title: str
    file_type: str
    page_number: int
    total_pages: int
    language: str

    text: str
    metadata: dict