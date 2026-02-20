"""Shared in-process application state for the FastAPI server.

Holds the session store and flags indicating whether models and the RAG
pipeline have been successfully initialised.  In a multi-worker deployment
this should be replaced with a Redis-backed store, but for single-process
use (development, demos) the in-memory dict is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppState:
    """Container for server-wide mutable state."""

    sessions: dict[str, dict] = field(default_factory=dict)
    models_loaded: bool = False
    rag_available: bool = False


# Singleton — imported by routers
app_state = AppState()
