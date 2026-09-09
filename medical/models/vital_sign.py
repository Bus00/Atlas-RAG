"""
medical/models/vital_sign.py
-------------------------------
VitalSign (yaşam bulgusu) veri modeli.

BU MODEL KLİNİK YORUM YAPMAZ. Örnek: temperature=38.4, unit="°C" saklanabilir,
ama bu model bunu ASLA "olası enfeksiyon" gibi bir sonuca çevirmez -- bu,
ileri faz(lar)daki klinik değerlendirme katmanının işidir.

Blood pressure özel durumu (proje talimatı Adım 5): tek bir skaler değer
yerine systolic/diastolic ayrı alanlar olarak temsil edilir. Bu yüzden aynı
dataclass hem skaler vital'ları (temperature/heart_rate/respiratory_rate/
spo2) hem de blood_pressure'ı, vital_type alanına göre farklı zorunlu
alanlarla temsil ediyor -- ayrı bir "BloodPressure" sınıfı yerine tek,
tutarlı bir VitalSign arayüzü tercih edildi (çağıran kodun tip kontrolü
yapması gerekmesin diye).

Hiçbir referans aralığı değeri burada sabit-kodlanmamıştır (bkz.
reference_range.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from medical.models.errors import MedicalDataValidationError
from medical.models.reference_range import ReferenceRange


class VitalSignType(str, Enum):
    TEMPERATURE = "temperature"
    HEART_RATE = "heart_rate"
    BLOOD_PRESSURE = "blood_pressure"
    RESPIRATORY_RATE = "respiratory_rate"
    SPO2 = "spo2"


def _is_number(val: object) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


@dataclass
class VitalSign:
    vital_type: VitalSignType
    timestamp: datetime
    source: str
    unit: str
    value: Optional[float] = None      # temperature / heart_rate / respiratory_rate / spo2 için
    systolic: Optional[float] = None   # SADECE blood_pressure için
    diastolic: Optional[float] = None  # SADECE blood_pressure için
    reference_range: Optional[ReferenceRange] = None
    quality: Optional[str] = None  # örn. "reliable", "motion_artifact" -- serbest metin, opsiyonel

    def __post_init__(self) -> None:
        if not isinstance(self.vital_type, VitalSignType):
            raise MedicalDataValidationError(f"vital_type geçerli bir VitalSignType olmalı: {self.vital_type!r}")
        if not isinstance(self.timestamp, datetime):
            raise MedicalDataValidationError(f"timestamp bir datetime nesnesi olmalı: {self.timestamp!r}")
        if not self.source or not self.source.strip():
            raise MedicalDataValidationError("source boş olamaz.")
        if not self.unit or not self.unit.strip():
            raise MedicalDataValidationError("unit boş olamaz.")

        if self.vital_type == VitalSignType.BLOOD_PRESSURE:
            if self.value is not None:
                raise MedicalDataValidationError(
                    "blood_pressure için 'value' kullanılmaz; systolic/diastolic kullanın."
                )
            if self.systolic is None or self.diastolic is None:
                raise MedicalDataValidationError(
                    "blood_pressure için hem systolic hem diastolic zorunludur."
                )
            if not _is_number(self.systolic):
                raise MedicalDataValidationError(f"systolic sayısal olmalı: {self.systolic!r}")
            if not _is_number(self.diastolic):
                raise MedicalDataValidationError(f"diastolic sayısal olmalı: {self.diastolic!r}")
        else:
            if self.systolic is not None or self.diastolic is not None:
                raise MedicalDataValidationError(
                    f"{self.vital_type.value} için systolic/diastolic geçerli değildir."
                )
            if self.value is None:
                raise MedicalDataValidationError(f"{self.vital_type.value} için 'value' zorunludur.")
            if not _is_number(self.value):
                raise MedicalDataValidationError(f"value sayısal olmalı: {self.value!r}")
