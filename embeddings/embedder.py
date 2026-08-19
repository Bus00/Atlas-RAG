"""
embeddings/embedder.py
------------------------
Lokal, çok dilli embedding modeli (varsayılan: BAAI/bge-m3) için ince bir
wrapper. sentence-transformers kütüphanesi üzerinden çalışır — model ilk
çalıştırmada HuggingFace'ten indirilir (internet gerekir), sonrasında
`~/.cache/huggingface` içinde saklanır ve tamamen offline kullanılabilir.

Neden bge-m3:
  - Çok dilli (100+ dil) ve Türkçe + İngilizce + Endonezce
    karışık metinlerde (bu projenin verisi tam olarak böyle) iyi performans
    gösteriyor.
  - Apache 2.0 lisanslı, ücretsiz, ticari kullanıma uygun.
  - sentence-transformers ile tek satırda yüklenebiliyor, ekstra bir
    servis/sunucu gerektirmiyor (tamamen in-process, offline).
  - Apple Silicon'da MPS (Metal) hızlandırma ile makul hızda çalışıyor;
    16 GB'lik bir Mac'i zorlamıyor (LLM'e göre çok daha hafif).
"""
from __future__ import annotations

from functools import lru_cache

from config.settings import settings


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        # Ağır import'u sınıf içine alıyoruz: bu modül import edildiğinde
        # (örn. testlerde) sentence-transformers hemen yüklenmesin.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.embedding_device
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Embedding modeli yüklenemedi ({self.model_name}, device={self.device}). "
                f"İlk çalıştırmada internet gerekir (model indirilir). "
                f"MPS ile ilgili bir hata alıyorsanız EMBEDDING_DEVICE=cpu deneyin.\n"
                f"Orijinal hata: {exc}"
            ) from exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


@lru_cache(maxsize=1)
def get_default_embedder() -> Embedder:
    """Modeli process başına yalnızca bir kez yükler (Streamlit'te tekrar tekrar yüklememek için)."""
    return Embedder()
