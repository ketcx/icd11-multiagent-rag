#!/usr/bin/env python
"""Diagnostic tool for Chroma Index status."""

import sys
from pathlib import Path
import json

CHROMA_PATH = Path(__file__).parent.parent / "data" / "indexes" / "chroma"
PDF_PATH = Path(__file__).parent.parent / "files" / "cie11.pdf"


def check_chroma_status():
    """Check if Chroma index exists and its contents."""
    print("\n" + "=" * 70)
    print("🔍 CHROMA INDEX DIAGNOSTIC")
    print("=" * 70)

    # 1. Check if directory exists
    print(f"\n1️⃣ Directory Check:")
    print(f"   Path: {CHROMA_PATH}")

    if CHROMA_PATH.exists():
        print(f"   Status: ✅ EXISTS")

        # Count files
        files = list(CHROMA_PATH.rglob("*"))
        print(f"   Files: {len([f for f in files if f.is_file()])}")

        # Check for critical Chroma files
        critical_files = ["data", "metadata", "index"]
        for fname in critical_files:
            exists = any(CHROMA_PATH.rglob(f"*{fname}*"))
            status = "✅" if exists else "❌"
            print(f"   {status} {fname}")
    else:
        print(f"   Status: ❌ MISSING")

    # 2. Check PDF
    print(f"\n2️⃣ PDF Source Check:")
    print(f"   Path: {PDF_PATH}")

    if PDF_PATH.exists():
        size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
        print(f"   Status: ✅ EXISTS")
        print(f"   Size: {size_mb:.2f} MB")
    else:
        print(f"   Status: ❌ MISSING")

    # 3. Try to load Chroma
    print(f"\n3️⃣ Chroma Load Test:")
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        if CHROMA_PATH.exists() and any(CHROMA_PATH.glob("*")):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/PubMedBERT-base-uncased-abstract"
            )
            vectorstore = Chroma(
                persist_directory=str(CHROMA_PATH),
                embedding_function=embeddings,
                collection_name="icd11",
            )

            # Try to get collection count
            try:
                count = vectorstore._collection.count()
                print(f"   Status: ✅ LOADED")
                print(f"   Index size: {count} chunks")
            except Exception as e:
                print(f"   Status: ⚠️  COLLECTION EMPTY - {e}")
        else:
            print(f"   Status: ⚠️  Index directory empty or missing")

    except ImportError as e:
        print(f"   Status: ❌ Import error - {e}")
        print(f"   Run: pip install langchain langchain-chroma langchain-huggingface")
    except Exception as e:
        print(f"   Status: ⚠️  {e}")

    # 4. Recommendation
    print(f"\n4️⃣ Next Steps:")

    if not CHROMA_PATH.exists() or not any(CHROMA_PATH.glob("*")):
        print(f"   Run: python scripts/ingest_pdf.py")
        print(f"   This will:")
        print(f"     • Extract text from {PDF_PATH.name}")
        print(f"     • Create embeddings (PubMedBERT)")
        print(f"     • Build Chroma index at {CHROMA_PATH}")
    else:
        print(f"   ✅ Index ready for RAG retrieval")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    check_chroma_status()
