"""
medical/models/observation.py
--------------------------------
Observation (gözlem) veri modeli -- doğrudan gözlemlenen veya hasta/mürettebat
tarafından bildirilen bulgu.

KRİTİK AYRIM (proje talimatı Adım 4/8 -- OBSERVED FACT vs ATLAS INTERPRETATION):
Observation, GÖZLEMLENEN/BİLDİRİLEN BİR OLGUYU temsil eder. Bu model hiçbir
şekilde otomatik olarak tanıya, hastalık olasılığına, klinik sonuca veya
şiddet skoruna DÖNÜŞTÜRÜLMEZ -- burada hiçbir yorumlama/çıkarım mantığı
yoktur. Örnek: "hasta göğüs ağrısı bildiriyor" geçerli bir Observation'dır;
bu model bunu asla "olası miyokard enfarktüsü" gibi bir sonuca çevirmez.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from medical.models.errors import MedicalDataValidationError


class ObservationCategory(str, Enum):
    """
    Proje talimatındaki örnek listeye dayanır. Kapalı bir küme değil --
    listede olmayan bir gözlem için OTHER kullanılır (description alanına
    serbest metin yazılır).
    """
    SYMPTOM = "symptom"
    PAIN = "pain"
    BLEEDING = "bleeding"
    CONSCIOUSNESS = "consciousness"
    BREATHING_DIFFICULTY = "breathing_difficulty"
    VOMITING = "vomiting"
    DIARRHEA = "diarrhea"
    DIZZINESS = "dizziness"
    WEAKNESS = "weakness"
    CONFUSION = "confusion"
    SKIN_FINDING = "skin_finding"
    NEUROLOGICAL_FINDING = "neurological_finding"
    OTHER = "other"


@dataclass
class Observation:
    category: ObservationCategory
    description: str
    timestamp: datetime
    source: str  # örn. "patient_report", "crew_observation", "onboard_medic" -- serbest metin
    reliability: Optional[str] = None  # örn. "reported", "directly_observed" -- opsiyonel, serbest metin

    def __post_init__(self) -> None:
        if not isinstance(self.category, ObservationCategory):
            raise MedicalDataValidationError(
                f"category geçerli bir ObservationCategory olmalı: {self.category!r}"
            )
        if not self.description or not self.description.strip():
            raise MedicalDataValidationError("description boş olamaz.")
        if not isinstance(self.timestamp, datetime):
            raise MedicalDataValidationError(f"timestamp bir datetime nesnesi olmalı: {self.timestamp!r}")
        if not self.source or not self.source.strip():
            raise MedicalDataValidationError("source boş olamaz.")
        if self.reliability is not None and not self.reliability.strip():
            raise MedicalDataValidationError("reliability verilmişse boş olamaz.")
