"""
llm/ollama_client.py
-----------------------
Ollama'nın lokal HTTP API'sine (varsayılan: http://localhost:11434) istek
atan basit istemci. Ollama'nın kendi Python SDK'sı yerine bilinçli olarak
`requests` kullanıyoruz — tek bir HTTP endpoint'e istek atmak için ayrı bir
bağımlılık eklemeye gerek yok, bu da projeyi hafif tutuyor.
"""
from __future__ import annotations

import requests

from config.settings import settings


class OllamaConnectionError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(self, prompt: str, system: str | None = None, temperature: float | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature if temperature is not None else settings.llm_temperature},
        }
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(
                f"Ollama'ya bağlanılamadı ({self.base_url}). Ollama uygulamasının açık olduğundan "
                f"ve '{self.model}' modelinin indirildiğinden ('ollama pull {self.model}') emin olun.\n"
                f"Orijinal hata: {exc}"
            ) from exc
        data = resp.json()
        return data.get("response", "").strip()
