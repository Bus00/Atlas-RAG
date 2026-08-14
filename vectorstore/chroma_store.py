"""
vectorstore/chroma_store.py
------------------------------
ChromaDB persistent client wrapper.

Neden ChromaDB (FAISS yerine):
  - Metadata filtreleme (gemi, tarih, event_type, ekipman) built-in;
    FAISS'te bunu elle yönetmek gerekirdi.
  - Tek satırla disk üzerinde persist ediyor (PersistentClient) — process
    yeniden başlasa da embedding'ler tekrar hesaplanmıyor.
  - Kurulumu tek `pip install chromadb`, ekstra servis/sunucu gerekmiyor.
  - Ücretsiz, tamamen lokal/offline çalışıyor.
  - Python entegrasyonu çok basit; V2'de veri büyürse (çoklu gemi, binlerce
    doküman) hâlâ yeterli performans sağlıyor.
"""
from __future__ import annotations

import chromadb

from config.settings import settings


class ChromaStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 6,
        where: dict | None = None,
    ) -> dict:
        """
        where örneği: {"event_type": "maintenance"} veya
        {"$and": [{"vessel_id": 1}, {"year": 2021}]}
        """
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

    def count(self) -> int:
        return self._collection.count()
