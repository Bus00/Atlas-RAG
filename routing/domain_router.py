"""
routing/domain_router.py
---------------------------
V2 Faz 2: Domain routing katmani.

Kullanicinin sorusunu MARITIME / MEDICAL / BOTH / GENERAL olarak siniflandirir.
Bu siniflandirma, RagPipeline'in soruyu hangi alt sisteme (denizcilik RAG'i,
gelecekteki tibbi karar destek modulu, vb.) yonlendirecegini belirler.

Tasarim (proje talimati geregi -- "Routing should be deterministic where
practical and LLM-assisted only when necessary"):
  - Bu katman TAMAMEN deterministik, anahtar-kelime tabanli bir siniflandiricidir.
    retrieval/structured_lookup.py'deki TR/EN anahtar kelime deseni (MONTHS_TR/
    MONTHS_EN) ve retrieval/retriever.py'deki EVENT_TYPE_HINTS ile ayni
    felsefeyi izler: basit, seffaf, test edilebilir kurallar.
  - LLM-destekli belirsizlik cozumleme (brief'in bahsettigi "LLM-assisted only
    when necessary" kismi) V2 Faz 2'de HENUZ IMPLEMENTE EDILMEDI. Bunu var gibi
    gostermek "no fake implementation" ilkesini ihlal eder -- bu yuzden burada
    acikca "gelecek is" olarak belirtiliyor, sessizce atlanmiyor.

Bilinen sinirlama (bilerek dokumante ediliyor, gizlenmiyor):
  Anahtar kelime eslesmesi baglamdan bagimsizdir. Ornegin "agri"/"ağrı" hem
  "pain" (tibbi) hem de bir Turkiye ili adi (Ağrı) olabilir. V1 veri setinde
  (MT IVANI, Endonezya sularinda) bu cakisma pratikte sorun yaratmiyor, ama
  genel bir NLU sistemi degildir -- gelismis intent-parsing V2'nin sonraki
  fazlarina birakildi.
"""
from __future__ import annotations

import re
from enum import Enum


class Domain(str, Enum):
    MARITIME = "MARITIME"
    MEDICAL = "MEDICAL"
    BOTH = "BOTH"
    GENERAL = "GENERAL"


MARITIME_KEYWORDS: list[str] = [
    # TR
    "gemi", "gemide", "mt ivani", "sefer", "liman", "bunker",
    "yakıt", "yakit", "rob", "tüketim", "tuketim", "noon report",
    "ana makine", "yardımcı makine", "yardimci makine", "ae", "me",
    "taze su", "dümen", "duman", "manevra", "kandas", "karaya oturma",
    "gövde", "govde", "yük", "yuk", "kargo", "mürettebat", "murettebat",
    "vardiya", "seyir", "tahliye",
    # EN
    "vessel", "ship", "voyage", "port", "fuel", "main engine",
    "auxiliary engine", "consumption", "cargo", "crew", "watch",
    "grounding", "hull", "steering", "bunkering",
]

MEDICAL_KEYWORDS: list[str] = [
    # TR
    "hasta", "hastanın", "hastanin", "yaralı", "yarali", "ateş", "ates",
    "nabız", "nabiz", "tansiyon", "kan basıncı", "kan basinci",
    "solunum", "nefes darlığı", "nefes darligi", "bilinç", "bilinc",
    "kanama", "bulantı", "bulanti", "kusma", "ishal", "baş dönmesi",
    "bas donmesi", "güçsüzlük", "gucsuzluk", "kafa karışıklığı",
    "konfüzyon", "konfuzyon", "döküntü", "dokuntu", "semptom", "belirti",
    "vital bulgu", "yaşam bulgusu", "yasam bulgusu", "spo2",
    "oksijen satürasyonu", "oksijen saturasyonu", "ilaç", "ilac",
    "alerji", "klinik", "tanı", "tani", "yoğun bakım", "yogun bakim",
    # EN
    "patient", "symptom", "fever", "pulse", "blood pressure",
    "consciousness", "bleeding", "vomiting", "diarrhea", "dizziness",
    "weakness", "confusion", "rash", "vital sign", "medication",
    "allergy", "clinical", "diagnosis",
]


def _contains_any(text: str, keywords: list[str]) -> bool:
    """
    Kelime-sınırı (\\b) eşleşmesi kullanılıyor -- düz substring değil.
    Neden: kısa ve yüksek değerli tokenler (örn. "ae", "me" -- Auxiliary/Main
    Engine, bu veri setinde equipment_hint olarak sürekli geçiyor, bkz.
    rag/chunker.py test verisi "AE 2") düz substring ile eşleşirse başka
    kelimelerin içinde yanlış-pozitif üretir (örn. "kamera" içinde "me" geçer).
    \\b sınırı bunu önler; Türkçe harfler (ı, ş, ğ, ü, ö, ç) Python'un
    Unicode-farkında \\w/\\b davranışıyla doğru kelime karakteri sayılır.
    """
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", lowered) for kw in keywords)


def classify_domain(question: str) -> Domain:
    """
    Soruyu deterministik anahtar-kelime eşleşmesiyle sınıflandırır.

    Bu fonksiyon LLM çağırmaz, dış servise (Postgres/Chroma/Ollama) bağımlı
    değildir -- öngörülebilirlik ve test edilebilirlik için bilinçli tercih
    (proje talimatı madde 4: "deterministic where practical").
    """
    has_maritime = _contains_any(question, MARITIME_KEYWORDS)
    has_medical = _contains_any(question, MEDICAL_KEYWORDS)

    if has_maritime and has_medical:
        return Domain.BOTH
    if has_medical:
        return Domain.MEDICAL
    if has_maritime:
        return Domain.MARITIME
    return Domain.GENERAL
