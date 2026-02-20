"""Entry point CLI for the ICD-11 Multi-Agent RAG system.

Usage examples::

    python main.py ingest --pdf files/cie11.pdf
    python main.py run --profile evals/profiles/anxiety_basic.json
    python main.py serve
    python main.py eval --suite evals/suites/standard.yaml
    python main.py download_models
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
import yaml


def _load_config(config_path: str) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """ICD-11 Multi-Agent RAG — Educational multi-agent clinical interview simulator."""


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--pdf", required=True, type=click.Path(exists=True), help="Path to the ICD-11 PDF.")
@click.option("--config", default="configs/app.yaml", show_default=True, help="Config file.")
def ingest(pdf: str, config: str) -> None:
    """Parses the ICD-11 PDF, chunks it, and builds the ChromaDB vector index."""
    cfg = _load_config(config)

    pdf_path = Path(pdf)
    persist_dir = Path(cfg["retrieval"]["persist_dir"])
    collection_name: str = cfg["retrieval"]["collection_name"]
    chunk_size: int = cfg["chunking"]["chunk_size"]
    chunk_overlap: int = cfg["chunking"]["chunk_overlap"]
    embedding_model: str = cfg["embeddings"]["model_name"]

    click.echo(f"Ingesting PDF: {pdf_path}")
    click.echo(f"  Collection  : {collection_name}")
    click.echo(f"  Index path  : {persist_dir}")

    from knowledge.ingest.pdf_parser import extract_pages
    from knowledge.indexing.chunker import ChunkConfig, chunk_documents
    from knowledge.indexing.chroma_builder import build_chroma_index

    pages = list(extract_pages(pdf_path))
    click.echo(f"  Extracted {len(pages)} pages.")

    chunk_cfg = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunk_documents(pages, chunk_cfg)
    click.echo(f"  Created {len(chunks)} chunks.")

    build_chroma_index(
        chunks=chunks,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model,
    )
    click.echo("  ✓ ChromaDB index built successfully.")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--profile", required=True, type=click.Path(exists=True), help="Client profile JSON.")
@click.option("--config", default="configs/app.yaml", show_default=True, help="Config file.")
@click.option("--language", default=None, help="Override language ('Español' or 'English').")
@click.option("--seed", default=42, show_default=True, type=int, help="Random seed.")
def run(profile: str, config: str, language: str | None, seed: int) -> None:
    """Executes a full automated session and writes the result to runs/."""
    import random
    random.seed(seed)

    cfg = _load_config(config)

    with open(profile) as fh:
        client_profile: dict = json.load(fh)

    session_language = language or client_profile.get("language", "Español")

    click.echo(f"Starting session")
    click.echo(f"  Profile  : {profile}")
    click.echo(f"  Language : {session_language}")

    from core.orchestration.graph import build_graph
    from core.orchestration.state import SessionState
    import uuid
    from datetime import datetime, timezone

    session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    initial_state: SessionState = {
        "session_id": session_id,
        "client_profile": client_profile,
        "transcript": [],
        "messages": [],
        "domains_covered": [],
        "domains_pending": [],
        "coverage_complete": False,
        "retrieved_chunks": [],
        "query_history": [],
        "hypotheses": [],
        "audit_report": None,
        "risk_detected": False,
        "risk_type": None,
        "current_step": "",
        "turn_count": 0,
        "max_turns": cfg["session"]["max_turns"],
        "finalized": False,
        "interactive_mode": False,
        "language": session_language,
    }

    graph = build_graph(interactive=False)
    final_state = graph.invoke(initial_state)

    # Persist output
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    output_path = runs_dir / f"{session_id}.json"
    with open(output_path, "w") as fh:
        json.dump(
            {
                "session_id": session_id,
                "hypotheses": final_state.get("hypotheses", []),
                "audit_report": final_state.get("audit_report"),
                "transcript": final_state.get("transcript", []),
                "coverage": {
                    "domains_covered": final_state.get("domains_covered", []),
                    "domains_pending": final_state.get("domains_pending", []),
                },
                "risk_detected": final_state.get("risk_detected", False),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    click.echo(f"  ✓ Session complete. Output: {output_path}")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--config", default="configs/app.yaml", show_default=True, help="Config file.")
@click.option("--host", default=None, help="API host (overrides config).")
@click.option("--port", default=None, type=int, help="API port (overrides config).")
def serve(config: str, host: str | None, port: int | None) -> None:
    """Launches the FastAPI REST API and the Streamlit UI."""
    import threading
    import uvicorn

    cfg = _load_config(config)
    api_host = host or cfg.get("api", {}).get("host", "0.0.0.0")
    api_port = port or cfg.get("api", {}).get("port", 8000)
    reload = cfg.get("api", {}).get("reload", False)

    click.echo(f"Starting API on http://{api_host}:{api_port}")
    click.echo("Starting Streamlit UI on http://localhost:8501")
    click.echo("Press Ctrl+C to stop both services.")

    # Launch Streamlit in a background thread
    def _run_streamlit() -> None:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "apps/ui/app.py",
             "--server.headless", "true"],
            check=False,
        )

    ui_thread = threading.Thread(target=_run_streamlit, daemon=True)
    ui_thread.start()

    # Run FastAPI in the main thread (blocking)
    uvicorn.run("apps.api.main:app", host=api_host, port=api_port, reload=reload)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--suite", required=True, type=click.Path(exists=True), help="Evaluation suite YAML.")
@click.option("--config", default="configs/app.yaml", show_default=True, help="Config file.")
@click.option("--output", default="runs/eval_report.json", show_default=True, help="Report output path.")
def eval(suite: str, config: str, output: str) -> None:
    """Runs an evaluation suite and writes a JSON report."""
    cfg = _load_config(config)

    with open(suite) as fh:
        suite_def: dict = yaml.safe_load(fh)

    profiles: list[str] = suite_def.get("profiles", [])
    click.echo(f"Evaluation suite: {suite}")
    click.echo(f"  Profiles: {len(profiles)}")

    results = []
    for profile_path in profiles:
        click.echo(f"  Running profile: {profile_path} …")
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(run, ["--profile", profile_path, "--config", config])
        results.append({"profile": profile_path, "exit_code": result.exit_code})

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump({"suite": suite, "results": results}, fh, indent=2)

    click.echo(f"  ✓ Evaluation complete. Report: {output_path}")


# ---------------------------------------------------------------------------
# download_models
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--config", default="configs/app.yaml", show_default=True, help="Config file.")
def download_models(config: str) -> None:
    """Downloads the GGUF LLM and PubMedBERT embeddings declared in the config."""
    from scripts.download_models import download_llm, download_embeddings, _load_config as _dl_cfg

    cfg = _dl_cfg(config)
    download_llm(cfg)
    download_embeddings(cfg)
    click.echo("✓ All models downloaded.")


if __name__ == "__main__":
    cli()
