from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="PolicyLens RAG API",
    version="0.1.0",
    description="A citation-grounded RAG system for internal knowledge base QA.",
)

app.include_router(router)