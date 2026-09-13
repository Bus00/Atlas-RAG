"""
medical/patient_timeline.py
----------------------------------
Hasta Zaman Çizelgesi (Patient Timeline).

BU MODÜL HİÇBİR YENİ KLİNİK SONUÇ ÇIKARMAZ. Sadece PatientRecord'daki
(Faz 4) mevcut Observation / VitalSign / LaboratoryResult kayıtlarını,
DEĞİŞTİRMEDEN, kronolojik ve deterministik bir sırada bir araya getirip
sunan, saf bir ORGANİZASYON katmanıdır. Tanı, olasılık, eşik, trend veya
başka bir yorum İÇERMEZ -- bunlar zaten medical/clinical_assessment.py ve
sonraki katmanların işidir; bu modül onların üzerine değil, PatientRecord'un
üzerine kuruludur (bağımsız, basit bir görünüm katmanı).

TASARIM KARARI:
Faz 3/4 dosyaları (Observation, VitalSign, LaboratoryResult, PatientRecord)
DEĞİŞTİRİLMEDİ. Bu modül SADECE PatientRecord.get_observations() /
.get_vital_signs() / .get_laboratory_results() metodlarını okur (zaten
kronolojik, kararlı sıralanmış kopyalar döndürürler -- bkz.
medical/patient_record.py) ve orijinal nesnelere REFERANS veren
TimelineEntry kayıtları üretir. Hiçbir alan kopyalanıp yeniden
yorumlanmaz.

DETERMİNİSTİK SIRALAMA (aynı zaman damgası için):
Birleşik zaman çizelgesi şu sabit sırayla oluşturulur: önce TÜM
observation'lar (kendi kronolojik sıralarında), sonra TÜM vital
sign'lar (kendi kronolojik sıralarında), sonra TÜM laboratuvar
sonuçları (kendi kronolojik sıralarında) -- ardından bu birleşik liste
SADECE timestamp'e göre KARARLI (stable) bir sort ile sıralanır. Python'un
sort()'u kararlı olduğu için, aynı timestamp'teki kayıtlar arasında bu
başlangıç sırası (Observation -> VitalSign -> LaboratoryResult, ve her
türün kendi içinde zaten kronolojik sırası) KORUNUR. Bu sabit tür-sırası
KLİNİK BİR ÖNCELİK DEĞİLDİR -- sadece öngörülebilir bir gösterim
kuralıdır.

MEDICAL HANDOFF İÇİN TEMEL (ileride kullanılacak, bu fazda TAM BİR HANDOFF
SİSTEMİ KURULMAMIŞTIR):
TimelineEntry, patient_id/timestamp/entry_type/description/source alanlarını
VE orijinal kayda (`original_record`) tam bir referansı taşır -- bu, gelecekte
bir Medical Handoff katmanının (henüz yazılmadı) ihtiyaç duyacağı temel
provenance bilgisini zaten sağlar. Bu fazda handoff'a özgü hiçbir ek alan,
format veya "özet" mantığı EKLENMEDİ (proje talimatı: "gereksiz dosya veya
mimari değişiklik yapma") -- sadece PatientTimeline'ın kendisi, böyle bir
katmanın doğrudan üzerine inşa edilebileceği kadar temiz ve tam kalıyor.

DETERMİNİZM:
  - Sistem saati (wall-clock time) veya rastgele sıralama KULLANILMAZ.
  - Sıralama SADECE PatientRecord'un zaten deterministik olan
    get_*() çıktılarına ve yukarıda açıklanan sabit tür-sırasına dayanır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Union

from medical.models.errors import MedicalDataValidationError
from medical.models.laboratory_result import LaboratoryResult
from medical.models.observation import Observation
from medical.models.vital_sign import VitalSign, VitalSignType
from medical.patient_record import PatientRecord


class TimelineEntryType(str, Enum):
    OBSERVATION = "observation"
    VITAL_SIGN = "vital_sign"
    LABORATORY_RESULT = "laboratory_result"


TimelineSourceRecord = Union[Observation, VitalSign, LaboratoryResult]


@dataclass
class TimelineEntry:
    """
    Zaman çizelgesindeki TEK bir kaydın yapısal temsili.

    `original_record`, ait olduğu Observation/VitalSign/LaboratoryResult
    nesnesine doğrudan bir referanstır (kopya/yeniden yorum DEĞİLDİR) --
    tam izlenebilirlik için korunur.
    """
    patient_id: str
    entry_type: TimelineEntryType
    timestamp: datetime
    description: str  # SADECE mevcut alanların düz-metin yeniden ifadesi; yeni bir yorum İÇERMEZ
    source: str
    original_record: TimelineSourceRecord


@dataclass
class PatientTimeline:
    """
    Bir hastaya ait, kronolojik ve deterministik sırada tutulan zaman
    çizelgesi. Hiçbir klinik yorum/tanı alanı YOKTUR -- sadece mevcut
    kayıtların düzenlenmiş bir görünümüdür.
    """
    patient_id: str
    entries: list[TimelineEntry] = field(default_factory=list)

    def entries_by_type(self, entry_type: TimelineEntryType) -> list[TimelineEntry]:
        return [e for e in self.entries if e.entry_type == entry_type]


# ---------------------------------------------------------------------
# Açıklama (description) üretimi -- SADECE mevcut alanların düz-metin
# yeniden ifadesi, hiçbir yeni değer/yorum İCAT EDİLMEZ.
# ---------------------------------------------------------------------

def _observation_description(obs: Observation) -> str:
    return f"{obs.category.value}: {obs.description}"


def _vital_sign_description(vital: VitalSign) -> str:
    if vital.vital_type == VitalSignType.BLOOD_PRESSURE:
        return f"{vital.vital_type.value}: {vital.systolic}/{vital.diastolic} {vital.unit}"
    return f"{vital.vital_type.value}: {vital.value} {vital.unit}"


def _laboratory_result_description(lab: LaboratoryResult) -> str:
    return f"{lab.test_name}: {lab.value} {lab.unit}"


# ---------------------------------------------------------------------
# Ana giriş noktası
# ---------------------------------------------------------------------

def build_patient_timeline(record: PatientRecord) -> PatientTimeline:
    """
    Bir PatientRecord'dan deterministik bir PatientTimeline üretir.

    PatientRecord'daki mevcut veriyi DEĞİŞTİRMEZ -- sadece
    get_observations()/get_vital_signs()/get_laboratory_results()
    üzerinden okur (bunlar zaten kopya listeler döndürür, bkz.
    medical/patient_record.py) ve orijinal nesnelere referans veren
    TimelineEntry kayıtları üretir.

    Boş bir PatientRecord için güvenle boş bir PatientTimeline
    (entries=[]) döner -- hata FIRLATILMAZ.
    """
    if not isinstance(record, PatientRecord):
        raise MedicalDataValidationError(f"record bir PatientRecord nesnesi olmalı: {record!r}")

    patient_id = record.patient_id
    entries: list[TimelineEntry] = []

    # Sabit tür-sırası (klinik öncelik DEĞİL, sadece deterministik
    # başlangıç sırası -- bkz. modül docstring'i): observation -> vital
    # sign -> laboratory result. Her tür kendi içinde zaten kronolojik.
    for obs in record.get_observations():
        entries.append(TimelineEntry(
            patient_id=patient_id,
            entry_type=TimelineEntryType.OBSERVATION,
            timestamp=obs.timestamp,
            description=_observation_description(obs),
            source=obs.source,
            original_record=obs,
        ))
    for vital in record.get_vital_signs():
        entries.append(TimelineEntry(
            patient_id=patient_id,
            entry_type=TimelineEntryType.VITAL_SIGN,
            timestamp=vital.timestamp,
            description=_vital_sign_description(vital),
            source=vital.source,
            original_record=vital,
        ))
    for lab in record.get_laboratory_results():
        entries.append(TimelineEntry(
            patient_id=patient_id,
            entry_type=TimelineEntryType.LABORATORY_RESULT,
            timestamp=lab.timestamp,
            description=_laboratory_result_description(lab),
            source=lab.source,
            original_record=lab,
        ))

    # KARARLI (stable) sort -- yukarıdaki sabit tür-sırası, aynı
    # timestamp'teki kayıtlar için KORUNUR (bkz. modül docstring'i).
    entries.sort(key=lambda e: e.timestamp)

    # Savunmacı tutarlılık kontrolü: tüm kayıtlar TEK bir PatientRecord'dan
    # geldiği için bu her zaman doğrudur, ama proje talimatı gereği hasta
    # ID tutarlılığı AÇIKÇA doğrulanır (bkz. test_patient_id_consistency_*).
    for entry in entries:
        if entry.patient_id != patient_id:
            raise MedicalDataValidationError(
                f"Zaman çizelgesi kaydı farklı bir hastaya ait: {entry.patient_id!r} != {patient_id!r}"
            )

    return PatientTimeline(patient_id=patient_id, entries=entries)
