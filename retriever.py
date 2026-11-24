from typing import List
from langchain_core.documents import Document
from app.rag.vectorstore import vector_store


def retrieve_context(query: str, k: int = 4) -> List[Document]:
    """Thin wrapper so the agent code stays clean."""
    return vector_store.similarity_search(query, k=k)


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)
