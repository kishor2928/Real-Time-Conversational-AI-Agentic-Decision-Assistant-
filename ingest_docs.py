#!/usr/bin/env python3
"""
One-shot script to load everything under data/sample_docs into FAISS.
Run from project root: python -m scripts.ingest_docs
"""

import sys
from pathlib import Path

# make sure app is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.vectorstore import vector_store


def main():
    docs_dir = Path("data/sample_docs")
    if not docs_dir.exists():
        print("No sample_docs folder found")
        return

    texts = []
    metas = []
    for f in docs_dir.glob("*.txt"):
        content = f.read_text(encoding="utf-8")
        texts.append(content)
        metas.append({"source": f.name})
        print(f"  queued {f.name} ({len(content)} chars)")

    if not texts:
        print("Nothing to ingest")
        return

    n = vector_store.add_texts(texts, metadatas=metas)
    print(f"Done – added {n} chunks")


if __name__ == "__main__":
    main()
