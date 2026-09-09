"""
medical/models/patient.py
---------------------------
Patient (hasta) veri modeli.

BU SADECE BİR VERİ MODELİDİR -- klinik yorum/çıkarım İÇERMEZ.

Tasarım: repodaki mevcut ingestion/csv_loader.py ve rag/chunker.py'deki
@dataclass + __post_init__ validasyon deseniyle tutarlı (pydantic gibi yeni
bir bağımlılık eklenmedi -- requirements.txt'te yok).

ÖNEMLİ (proje talimatı Adım 3): "Do not store clinical diagnoses as facts
unless they are explicitly documented as historical/known information."
`known_history` alanı bu yüzden serbest metin bir liste olarak tutuluyor --
ATLAS'ın kendi çıkarımı değil, açıkça belgelenmiş/bildirilmiş geçmiş bilgi
içindir (örn. "bilinen astım öyküsü, hasta tarafından bildirildi").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from medical.models.errors import MedicalDataValidationError


@dataclass
class Patient:
    patient_id: str
    display_name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None  # serbest metin (örn. "male"/"female"/"unknown") -- kapalı bir enum'a zorlanmadı
    allergies: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    known_history: list[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.patient_id or not self.patient_id.strip():
            raise MedicalDataValidationError("patient_id boş olamaz.")

        if self.age is not None:
            if not isinstance(self.age, int) or isinstance(self.age, bool):
                raise MedicalDataValidationError(f"age tam sayı olmalı: {self.age!r}")
            if self.age < 0:
                # Yalnızca yapısal olarak imkansız bir değeri (negatif yaş)
                # reddediyoruz -- üst sınır koymuyoruz, bu klinik bir
                # değerlendirme değil, sadece imkansızlık kontrolü.
                raise MedicalDataValidationError(f"age negatif olamaz: {self.age}")

        for label, values in (
            ("allergies", self.allergies),
            ("current_medications", self.current_medications),
            ("known_history", self.known_history),
        ):
            if not isinstance(values, list):
                raise MedicalDataValidationError(f"{label} bir liste olmalı: {values!r}")
            for item in values:
                if not isinstance(item, str) or not item.strip():
                    raise MedicalDataValidationError(f"{label} içindeki her öğe boş olmayan bir metin olmalı: {item!r}")

        for label, ts in (("created_at", self.created_at), ("updated_at", self.updated_at)):
            if ts is not None and not isinstance(ts, datetime):
                raise MedicalDataValidationError(f"{label} bir datetime nesnesi olmalı: {ts!r}")
