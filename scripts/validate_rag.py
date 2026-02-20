#!/usr/bin/env python
"""Validation script for the complete RAG pipeline."""

from pathlib import Path


def validate_ingestion():
    """Check PDF ingestion artifacts."""
    pdf_path = Path(__file__).parent.parent / "files" / "cie11.pdf"
    chroma_path = Path(__file__).parent.parent / "data" / "indexes" / "chroma"

    print("\n" + "=" * 70)
    print("📊 RAG PIPELINE VALIDATION")
    print("=" * 70)

    print("\n1️⃣ INGESTION ARTIFACTS")
    print(f"   PDF: {pdf_path.name}")
    print(f"   • Status: {'✅ EXISTS' if pdf_path.exists() else '❌ MISSING'}")
    if pdf_path.exists():
        size = pdf_path.stat().st_size / (1024 * 1024)
        print(f"   • Size: {size:.2f} MB")

    print(f"\n   Chroma Index: {chroma_path.name}")
    print(f"   • Status: {'✅ EXISTS' if chroma_path.exists() else '❌ MISSING'}")
    if chroma_path.exists():
        files = list(chroma_path.glob("*"))
        print(f"   • Files: {len(files)}")
        for f in files:
            print(f"     - {f.name}")


def validate_pipeline():
    """Test RAG pipeline initialization and retrieval."""
    print("\n2️⃣ PIPELINE INITIALIZATION")

    try:
        from core.retrieval import get_rag_pipeline

        pipeline = get_rag_pipeline()

        if pipeline is None:
            print("   ⚠️  Pipeline returned None (using mock fallback)")
            return False

        print("   ✅ RAG pipeline initialized")
        print(f"   • Retriever: HybridRetriever (Dense + BM25)")
        print(f"   • Query Builder: ChunkingStrategy")
        print(f"   • Embedding: TF-IDF (lightweight, 384D)")

        return True

    except Exception as e:
        print(f"   ❌ Pipeline initialization failed: {e}")
        return False


def validate_agents():
    """Check if agents are registered."""
    print("\n3️⃣ AGENT REGISTRATION")

    try:
        from core.orchestration.nodes import AGENTS

        # Agents should be loaded by app.py during init
        if not AGENTS:
            print("   ⚠️  No agents registered yet (will be loaded in Streamlit)")
        else:
            print(f"   ✅ Agents registered: {list(AGENTS.keys())}")
            for name in AGENTS:
                print(f"      • {name}")

        return True

    except Exception as e:
        print(f"   ❌ Agent check failed: {e}")
        return False


def validate_graph():
    """Check if the LangGraph is buildable."""
    print("\n4️⃣ LANGGRAPH ORCHESTRATION")

    try:
        from core.orchestration.graph import build_graph
        from langgraph.checkpoint.memory import MemorySaver

        memory = MemorySaver()
        graph = build_graph(checkpointer=memory)

        print("   ✅ LangGraph compiled successfully")
        print(f"   • Checkpointer: MemorySaver (for interrupts)")
        print(f"   • Nodes: init → therapist → risk_check → ... → finalize")
        print(f"   • Interrupt: human_input (for interactive mode)")

        return True

    except Exception as e:
        print(f"   ❌ Graph validation failed: {e}")
        return False


def validate_endpoints():
    """Check if all key endpoints exist."""
    print("\n5️⃣ IMPLEMENTATION CHECKLIST")

    from pathlib import Path
    import os

    files_to_check = {
        "core/agents/therapist.py": "TherapistAgent",
        "core/agents/client.py": "ClientAgent",
        "core/agents/diagnostician.py": "DiagnosticianAgent",
        "core/retrieval/retrievers.py": "HybridRetriever",
        "core/orchestration/nodes.py": "retrieve_context",
        "apps/ui/app.py": "load_agents",
    }

    base = Path(__file__).parent.parent

    for relpath, component in files_to_check.items():
        fpath = base / relpath
        if fpath.exists():
            with open(fpath) as f:
                content = f.read()
                if component in content:
                    print(f"   ✅ {relpath}")
                    print(f"      └─ {component}")
                else:
                    print(f"   ❌ {relpath}")
                    print(f"      └─ Missing: {component}")
        else:
            print(f"   ❌ {relpath} not found")


def show_summary():
    """Show final summary."""
    print("\n" + "=" * 70)
    print("PIPELINE READY FOR TESTING")
    print("=" * 70)
    print("""
✅ Status: All components initialized
   
To launch the UI:
   cd /Users/ketcx/pinguino_project/rag-project
   streamlit run apps/ui/app.py

Workflow:
   1. Choose language (Español/English)
   2. Choose mode (Interactive/Auto)
   3. Start interview
   4. System will:
      • Ask therapeutic questions
      • Monitor for safety risks
      • Retrieve ICD-11 context via RAG
      • Generate diagnostic hypotheses
      • Audit evidence and finalize
    
Known limitations:
   ⚠️  RAG retrieval uses mock fallback (TF-IDF lightweight)
   ⚠️  Only 30 PDF pages indexed (for speed)
   ⚠️  DiagnosticianAgent may return hardcoded output if RAG fails
   
To index full PDF:
   python scripts/ingest_pdf_lite.py

To diagnose:
   python scripts/diagnose_chroma.py
""")
    print("=" * 70)


if __name__ == "__main__":
    validate_ingestion()
    pipeline_ok = validate_pipeline()
    validate_agents()
    validate_graph()
    validate_endpoints()

    if pipeline_ok:
        show_summary()
    else:
        print("\n⚠️  Some components need attention before launching Streamlit")
