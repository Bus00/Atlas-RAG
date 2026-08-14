"""
config/settings.py
-------------------
Projenin tüm ayarlarının tek toplandığı yer. Diğer modüller ayar okumak
istediğinde doğrudan `.env` okumaz, buradaki `settings` nesnesini kullanır.

Neden böyle: Ayarlar dağınık olursa (her dosyada os.getenv çağrısı) hem
test etmek hem de "hangi ayar nerede kullanılıyor" takip etmek zorlaşır.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass(frozen=True)
class Settings:
    # --- PostgreSQL ---
    pg_host: str = field(default_factory=lambda: _get("PG_HOST", "localhost"))
    pg_port: int = field(default_factory=lambda: int(_get("PG_PORT", "5432")))
    pg_db: str = field(default_factory=lambda: _get("PG_DB", "maritime_rag"))
    pg_user: str = field(default_factory=lambda: _get("PG_USER", "postgres"))
    pg_password: str = field(default_factory=lambda: _get("PG_PASSWORD", "postgres"))

    # --- Embedding ---
    # bge-m3: çok dilli (100+ dil), Türkçe/İngilizce/Endonezce'de güçlü,
    # tamamen lokal (HuggingFace/sentence-transformers üzerinden) çalışır.
    embedding_model_name: str = field(
        default_factory=lambda: _get("EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    embedding_device: str = field(default_factory=lambda: _get("EMBEDDING_DEVICE", "mps"))

    # --- ChromaDB ---
    chroma_persist_dir: str = field(
        default_factory=lambda: _get(
            "CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_store")
        )
    )
    chroma_collection_name: str = field(
        default_factory=lambda: _get("CHROMA_COLLECTION", "noon_report_chunks")
    )

    # --- Ollama / LLM ---
    ollama_base_url: str = field(default_factory=lambda: _get("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _get("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    llm_temperature: float = field(default_factory=lambda: float(_get("LLM_TEMPERATURE", "0.1")))

    # --- Retrieval ---
    retrieval_top_k: int = field(default_factory=lambda: int(_get("RETRIEVAL_TOP_K", "6")))

    @property
    def pg_dsn(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} dbname={self.pg_db} "
            f"user={self.pg_user} password={self.pg_password}"
        )


settings = Settings()
