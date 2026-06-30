"""FastAPI application for the ALINA GPSS Consultant API."""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from alina_rag.config import settings
from alina_rag.domain.models import BotPlatform, UserId

if TYPE_CHECKING:
    from alina_rag.application.chat_service import ChatService
    from alina_rag.application.document_service import DocumentService

logger = logging.getLogger(__name__)


# ── Request / Response models ─────────────────────────


class ChatRequest(BaseModel):
    user_id: str = Field(default="api:anonymous")
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str


class UploadResponse(BaseModel):
    chunks: int
    filename: str


class DocumentsResponse(BaseModel):
    documents: list[dict[str, str]]


class HealthResponse(BaseModel):
    status: str
    chunks: int


# ── Application factory ───────────────────────────────


def create_app(
    chat_service: "ChatService",
    document_service: "DocumentService",
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="ALINA GPSS Consultant API")

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        """Send a message and get an AI response."""
        user_id = UserId(BotPlatform.API, request.user_id)
        try:
            response = await chat_service.handle_message(
                user_id, request.message
            )
        except Exception as exc:
            logger.exception("Chat endpoint error")
            raise HTTPException(
                status_code=500, detail="Internal processing error"
            ) from exc
        return ChatResponse(response=response)

    @app.post("/documents/upload", response_model=UploadResponse)
    async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
        """Upload a document for ingestion into the knowledge base."""
        if not file.filename:
            raise HTTPException(
                status_code=400, detail="Filename is required"
            )

        docs_path = settings.docs_path
        docs_path.mkdir(parents=True, exist_ok=True)

        file_path = docs_path / file.filename
        try:
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as exc:
            logger.exception("File save error")
            raise HTTPException(
                status_code=500, detail="Failed to save file"
            ) from exc

        try:
            chunks = await document_service.ingest_file(file_path)
        except Exception as exc:
            logger.exception("Document ingestion error")
            raise HTTPException(
                status_code=500, detail="Failed to ingest document"
            ) from exc

        return UploadResponse(chunks=chunks, filename=file.filename)

    @app.get("/documents", response_model=DocumentsResponse)
    async def list_documents() -> DocumentsResponse:
        """List all ingested documents."""
        try:
            docs = await document_service.list_documents()
        except Exception as exc:
            logger.exception("List documents error")
            raise HTTPException(
                status_code=500, detail="Failed to list documents"
            ) from exc

        return DocumentsResponse(
            documents=[
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "source": doc.source.value,
                }
                for doc in docs
            ]
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint returning total chunk count."""
        try:
            docs = await document_service.list_documents()
        except Exception:
            docs = []
        total_chunks = sum(doc.chunk_count for doc in docs)
        return HealthResponse(status="ok", chunks=total_chunks)

    return app


# Module-level factory for uvicorn (use with --factory flag):
#   uvicorn alina_rag.presentation.api:app --factory
app = create_app
