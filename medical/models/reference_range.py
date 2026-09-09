"""
medical/models/reference_range.py
------------------------------------
VitalSign ve LaboratoryResult'ın ortak kullandığı, TAMAMEN OPSİYONEL
referans aralığı gösterimi.

ÖNEMLİ (proje talimatı Adım 5/6): "Do not fabricate reference ranges."
Bu modül HİÇBİR referans aralığı DEĞERİ içermez -- sadece bir taşıyıcı
(carrier) veri yapısıdır. Gerçek low/high değerleri, ileride gerçek,
otoriter bir kaynaktan (örn. laboratuvar cihazı, vetted klinik referans)
çağıran kod tarafından sağlanmalıdır. Burada hiçbir sayısal eşik
sabit-kodlanmamıştır.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from medical.models.errors import MedicalDataValidationError


@dataclass
class ReferenceRange:
    low: Optional[float] = None
    high: Optional[float] = None
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name, val in (("low", self.low), ("high", self.high)):
            if val is not None and (not isinstance(val, (int, float)) or isinstance(val, bool)):
                raise MedicalDataValidationError(f"{field_name} sayısal olmalı: {val!r}")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise MedicalDataValidationError(
                f"low ({self.low}) high'tan ({self.high}) büyük olamaz."
            )
