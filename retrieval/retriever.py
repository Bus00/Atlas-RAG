"""
retrieval/retriever.py
-------------------------
Kullanıcı sorusunu alır, embed eder, ChromaDB'de arar ve sonuçları
sade bir yapıya (RetrievedChunk listesi) çevirir.

Ayrıca sorudan basit, kural tabanlı metadata filtresi çıkarır (örn. soru
"2021" veya "bakım" içeriyorsa ilgili filtreyi otomatik uygular). Bu V1
için yeterli; gelişmiş NLU/intent-parsing V2 konusu.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config.settings import settings
from embeddings.embedder import Embedder, get_default_embedder
from vectorstore.chroma_store import ChromaStore

EVENT_TYPE_HINTS = {
    "maintenance": ["bakım", "bakim", "yağ değişim", "yag degisim", "maintenance"],
    "failure": ["arıza", "ariza", "kandas", "karaya", "failure", "hasar"],
    "bunker": ["ikmal", "bunker", "supply", "yakıt", "yakit"],
    "voyage": ["sefer", "voyage", "liman", "port"],
}


@dataclass
class RetrievedChunk:
    chunk_text: str
    vessel: str
    report_date: str
    chunk_type: str
    event_type: str
    equipment_hint: str
    distance: float


class Retriever:
    def __init__(self, embedder: Embedder | None = None, store: ChromaStore | None = None):
        self._embedder = embedder or get_default_embedder()
        self._store = store or ChromaStore()

    @staticmethod
    def _infer_filters(question: str) -> dict | None:
        clauses = []

        year_match = re.search(r"\b(19|20)\d{2}\b", question)
        if year_match:
            clauses.append({"year": int(year_match.group(0))})

        lowered = question.lower()
        for event_type, hints in EVENT_TYPE_HINTS.items():
            if any(h in lowered for h in hints):
                clauses.append({"event_type": event_type})
                break  # birden fazla türle eşleşirse ilkini kullan, aşırı daraltmayı önle

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        extra_where: dict | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval_top_k
        query_vector = self._embedder.embed_query(question)

        where = extra_where or self._infer_filters(question)
        result = self._store.query(query_vector, top_k=top_k, where=where)

        # Filtre çok daraltıp sonuç boşsa, filtresiz tekrar dene (kullanıcıyı
        # boş elle bırakmamak için) — ama bunu açıkça loglayalım.
        if where and not result.get("documents", [[]])[0]:
            result = self._store.query(query_vector, top_k=top_k, where=None)

        chunks: list[RetrievedChunk] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0] if result.get("distances") else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append(
                RetrievedChunk(
                    chunk_text=doc,
                    vessel=meta.get("vessel", ""),
                    report_date=meta.get("report_date", ""),
                    chunk_type=meta.get("chunk_type", ""),
                    event_type=meta.get("event_type", ""),
                    equipment_hint=meta.get("equipment_hint", ""),
                    distance=dist,
                )
            )
        return chunks
