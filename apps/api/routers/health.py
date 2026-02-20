"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from apps.api.state import app_state

router = APIRouter()


@router.get("/", summary="System health and loaded-model status")
def check_health() -> dict:
    """Returns service status and which components are loaded.

    Used by load balancers, monitoring, and the Streamlit sidebar to
    indicate whether inference is available.
    """
    return {
        "status": "ok",
        "models_loaded": app_state.models_loaded,
        "rag_available": app_state.rag_available,
        "active_sessions": len(app_state.sessions),
    }
