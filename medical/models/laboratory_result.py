"""
medical/models/laboratory_result.py
--------------------------------------
LaboratoryResult (laboratuvar sonucu) veri modeli.

GENİŞLETİLEBİLİRLİK ÖNEMLİ (proje talimatı Adım 6): "Design must be
extensible so additional laboratory tests can be added without modifying
the core architecture." Bu yüzden `test_name` KAPALI BİR ENUM DEĞİL,
serbest bir string'dir -- yeni bir laboratuvar testi eklemek için bu
dosyanın veya çekirdek mimarinin değiştirilmesi GEREKMEZ.

COMMON_LAB_TEST_NAMES yalnızca bilgi/referans amaçlıdır (proje talimatında
"Initial supported examples" olarak listelenen testler) -- zorunlu bir
kısıtlama DEĞİLDİR, doğrulamada kullanılmaz.

Hiçbir referans aralığı değeri veya klinik eşik burada sabit-kodlanmamıştır.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from medical.models.errors import MedicalDataValidationError
from medical.models.reference_range import ReferenceRange

# Yalnızca referans/bilgi amaçlı -- test_name doğrulamasında KULLANILMAZ.
# Bkz. modül docstring'i: sistem bunların dışında herhangi bir test adını
# da kabul eder.
COMMON_LAB_TEST_NAMES = (
    "Hb", "WBC", "RBC", "platelets", "glucose", "CRP", "ALT", "AST", "creatinine",
)


@dataclass
class LaboratoryResult:
    test_name: str
    value: float
    unit: str
    timestamp: datetime
    source: str
    reference_range: Optional[ReferenceRange] = None
    status: Optional[str] = None  # örn. "final", "preliminary", "pending_confirmation" -- serbest metin

    def __post_init__(self) -> None:
        if not self.test_name or not self.test_name.strip():
            raise MedicalDataValidationError("test_name boş olamaz.")
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise MedicalDataValidationError(f"value sayısal olmalı: {self.value!r}")
        if not self.unit or not self.unit.strip():
            raise MedicalDataValidationError("unit boş olamaz.")
        if not isinstance(self.timestamp, datetime):
            raise MedicalDataValidationError(f"timestamp bir datetime nesnesi olmalı: {self.timestamp!r}")
        if not self.source or not self.source.strip():
            raise MedicalDataValidationError("source boş olamaz.")
        if self.status is not None and not self.status.strip():
            raise MedicalDataValidationError("status verilmişse boş olamaz.")
