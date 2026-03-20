from pydantic import BaseModel, Field


class PageDocument(BaseModel):
    id: str = Field(..., description="唯一頁面 ID")
    doc_id: str = Field(..., description="文件 ID")
    source_path: str
    file_name: str
    document_title: str
    file_type: str
    page_number: int
    total_pages: int
    language: str
    text: str
    metadata: dict