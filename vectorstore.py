"""
FAISS-backed vector store using local sentence-transformers embeddings.
We keep the embedding model local so the whole stack can run offline
once Ollama + the model are pulled.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import get_settings


class VectorStoreManager:
    def __init__(self):
        self.settings = get_settings()
        self._embeddings = None
        self._store: Optional[FAISS] = None
        self._path = Path(self.settings.vector_store_path)

    @property
    def embeddings(self):
        if self._embeddings is None:
            # using a small, fast model that works well locally
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def _ensure_dir(self):
        self._path.mkdir(parents=True, exist_ok=True)

    def load_or_create(self) -> FAISS:
        self._ensure_dir()
        index_file = self._path / "index.faiss"
        if index_file.exists():
            self._store = FAISS.load_local(
                str(self._path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            # empty store with a dummy doc so FAISS doesn't complain
            dummy = [Document(page_content="placeholder", metadata={"source": "init"})]
            self._store = FAISS.from_documents(dummy, self.embeddings)
            self.save()
        return self._store

    def save(self):
        if self._store is not None:
            self._ensure_dir()
            self._store.save_local(str(self._path))

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        if self._store is None:
            self.load_or_create()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        docs = splitter.create_documents(texts, metadatas=metadatas)
        self._store.add_documents(docs)
        self.save()
        return len(docs)

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        if self._store is None:
            self.load_or_create()
        k = k or self.settings.top_k
        return self._store.similarity_search(query, k=k)


# process-wide instance
vector_store = VectorStoreManager()
