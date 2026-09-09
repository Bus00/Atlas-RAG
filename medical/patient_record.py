"""
medical/patient_record.py
----------------------------
V2 Faz 4: Medikal yapısal-veri entegrasyon katmanı (Patient Record container).

BU MODÜL KLİNİK ÇIKARIM/YORUM İÇERMEZ. Faz 3'teki tekil veri modellerini
(Patient, Observation, VitalSign, LaboratoryResult) tek bir hasta için bir
araya getiren, saf bir KOMPOZİSYON/CONTAINER katmanıdır. Burada:
  - tanı YOKTUR
  - ayırıcı tanı YOKTUR
  - klinik/acil eşik YOKTUR
  - olasılık/skor YOKTUR
  - ilaç önerisi/güvenliği YOKTUR
Bunların tümü ileri fazların işidir (bkz. proje talimatı).

TASARIM KARARI (önceden tartışıldı ve onaylandı):
Faz 3 modelleri (Observation/VitalSign/LaboratoryResult) bilinçli olarak
patient_id İÇERMEZ -- yeniden kullanılabilir, hastadan bağımsız value object
olarak tasarlandılar. Bu yüzden Faz 3 dosyaları DEĞİŞTİRİLMEDİ; bunun yerine
PatientRecord, bir Patient'ı ve ona ait gözlem/vital/lab koleksiyonlarını
saran ayrı bir kompozisyon katmanı olarak eklendi. Böylece:
  - Faz 3 modelleri hâlâ hastadan bağımsız, yeniden kullanılabilir kalır,
  - "hangi kayıt hangi hastaya ait" bilgisi container seviyesinde tutulur
    (PatientRecord örneği = tek bir hasta bağlamı),
  - Faz 3 dosyalarında "sadece patient_id eklemek için" bir değişiklik
    yapılmadı (proje talimatı: yalnızca güçlü mimari bir gerekçe varsa
    değiştir).

Repodaki mevcut desenle tutarlı: @dataclass + __post_init__ yapısal
doğrulama (bkz. medical/models/*.py, ingestion/csv_loader.py). Pydantic gibi
yeni bir bağımlılık eklenmedi.

DETERMİNİSTİK SIRALAMA:
get_observations() / get_vital_signs() / get_laboratory_results(),
kayıtları timestamp'e göre ARTAN sırada döndürür. Python'un sort()/sorted()
algoritması kararlıdır (stable) -- yani aynı timestamp'e sahip kayıtlar
arasında EKLENME SIRASI korunur. Bu, testlerde ve çağıran kodda
öngörülebilir/tekrarlanabilir bir sıra garanti eder.

Döndürülen listeler her zaman birer KOPYADIR (iç listenin kendisi değil) --
çağıran kod get_observations() sonucunu mutasyona uğratsa bile
PatientRecord'un iç durumu bozulmaz.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from medical.models.errors import MedicalDataValidationError
from medical.models.laboratory_result import LaboratoryResult
from medical.models.observation import Observation
from medical.models.patient import Patient
from medical.models.vital_sign import VitalSign


def _validate_list(label: str, items: object, expected_type: type) -> list:
    if not isinstance(items, list):
        raise MedicalDataValidationError(f"{label} bir liste olmalı: {items!r}")
    for item in items:
        if not isinstance(item, expected_type):
            raise MedicalDataValidationError(
                f"{label} içindeki her öğe {expected_type.__name__} olmalı: {item!r}"
            )
    return items


@dataclass
class PatientRecord:
    """
    Tek bir hastaya ait yapısal medikal durumu temsil eden container.

    Faz 3 modellerini (Observation, VitalSign, LaboratoryResult) hastadan
    bağımsız value object olarak korurken, bunları tek bir hasta bağlamında
    bir araya getirmek için kullanılır. Klinik yorum/çıkarım YAPMAZ.
    """

    patient: Patient
    observations: list[Observation] = field(default_factory=list)
    vital_signs: list[VitalSign] = field(default_factory=list)
    laboratory_results: list[LaboratoryResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.patient, Patient):
            raise MedicalDataValidationError(
                f"patient bir Patient nesnesi olmalı: {self.patient!r}"
            )
        # Constructor'a doğrudan liste verilirse (örn. testlerde) yine de
        # yapısal doğrulama uygulanır -- add_* metodlarıyla eklenenlerle
        # aynı garanti.
        self.observations = _validate_list("observations", self.observations, Observation)
        self.vital_signs = _validate_list("vital_signs", self.vital_signs, VitalSign)
        self.laboratory_results = _validate_list(
            "laboratory_results", self.laboratory_results, LaboratoryResult
        )

    @property
    def patient_id(self) -> str:
        """Kolaylık için: self.patient.patient_id'ye kısayol."""
        return self.patient.patient_id

    # ------------------------------------------------------------------
    # Ekleme (add_*)
    # ------------------------------------------------------------------

    def add_observation(self, observation: Observation) -> None:
        if not isinstance(observation, Observation):
            raise MedicalDataValidationError(
                f"observation bir Observation nesnesi olmalı: {observation!r}"
            )
        self.observations.append(observation)

    def add_vital_sign(self, vital_sign: VitalSign) -> None:
        if not isinstance(vital_sign, VitalSign):
            raise MedicalDataValidationError(
                f"vital_sign bir VitalSign nesnesi olmalı: {vital_sign!r}"
            )
        self.vital_signs.append(vital_sign)

    def add_laboratory_result(self, laboratory_result: LaboratoryResult) -> None:
        if not isinstance(laboratory_result, LaboratoryResult):
            raise MedicalDataValidationError(
                f"laboratory_result bir LaboratoryResult nesnesi olmalı: {laboratory_result!r}"
            )
        self.laboratory_results.append(laboratory_result)

    # ------------------------------------------------------------------
    # Getirme (get_*) -- her zaman timestamp'e göre deterministik ARTAN
    # sırada, kararlı (stable) sort sayesinde eşit timestamp'lerde ekleme
    # sırası korunarak. Her çağrı bağımsız bir kopya döndürür.
    # ------------------------------------------------------------------

    def get_observations(self) -> list[Observation]:
        return sorted(self.observations, key=lambda o: o.timestamp)

    def get_vital_signs(self) -> list[VitalSign]:
        return sorted(self.vital_signs, key=lambda v: v.timestamp)

    def get_laboratory_results(self) -> list[LaboratoryResult]:
        return sorted(self.laboratory_results, key=lambda lr: lr.timestamp)
