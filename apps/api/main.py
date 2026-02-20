"""FastAPI app — ICD-11 Multi-Agent RAG."""

from fastapi import FastAPI
from apps.api.routers import sessions, batch, health

app = FastAPI(
    title="ICD-11 Multi-Agent RAG",
    description="Educational multi-agent system with RAG over ICD-11",
    version="0.1.0",
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(batch.router, prefix="/batch", tags=["batch"])
