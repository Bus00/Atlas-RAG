"""
medical/next_best_measurement.py
----------------------------------------
V2 Faz (Next Best Measurement & Resource-Aware Decision Support), Bölüm A:
Sonraki En İyi Ölçüm (Next Best Measurement).

BU MODÜL TANI KOYMAZ, TEDAVİ ÖNERMEZ, İLAÇ ÖNERMEZ. Hiçbir şekilde:
  - hastalık tanısı / olasılığı / risk skoru
  - istatistiksel çıkarım veya "information gain" hesaplaması (böyle bir
    hesaplama bu depoda GERÇEKTEN uygulanmadığı için ASLA iddia edilmez)
  - klinik eşik değeri / fabrikasyon klinik kılavuz
  - ilaç/dozaj/tedavi süresi
  - acil durum/tahliye kararı
ÜRETMEZ.

AMAÇ:
Faz 5-7'nin ürettiği yapısal veriyi (ClinicalAssessment, ClinicalReasoning,
opsiyonel olarak DifferentialAssessment) kullanarak, "hangi ölçüm sonraki
adayı olabilir" sorusuna SADECE VERİ-TAMAMLIĞI (data completeness)
açısından cevap veren, tamamen yapısal ve deterministik bir aday listesi
üretir. Bu bir ÖNCELİKLENDİRME/SIRALAMA ALGORİTMASI DEĞİLDİR -- proje
talimatı gereği ("Do not pretend to calculate information gain unless an
actual validated information-theoretic implementation exists"), burada
HİÇBİR istatistiksel/olasılıksal sıralama yapılmaz. Adaylar sadece SABİT,
açık bir yapısal sırada (aşağıya bkz.) sunulur ve bu sıranın klinik bir
öncelik OLMADIĞI her NextBestMeasurementAssessment.scope_limitations
içinde açıkça belirtilir.

ADAY TÜRLERİ (SADECE ÜÇ TANE, hepsi veri-tamlığı temellidir):
  1. MISSING_MEASUREMENT   -- bir vital-sign tipi hiç kaydedilmemiş
                               (SADECE VitalSignType için; laboratuvar
                               testleri açık-uçlu/genişletilebilir olduğu
                               için -- bkz. medical/models/laboratory_result.py
                               -- "eksik laboratuvar paneli" kavramı
                               İCAT EDİLMEZ, bkz. aşağıdaki tasarım kararı).
  2. INSUFFICIENT_FOR_TREND -- bir seri için sadece 1 ölçüm var (Faz 5'in
                               Trend.INSUFFICIENT_DATA'sıyla birebir eşleşir).
  3. CONFLICT_RESOLUTION    -- bir seride aynı zaman damgasında çelişkili
                               değerler var (Faz 5/6/7'nin conflict
                               tespitiyle birebir eşleşir); YİNE DE hangi
                               ölçümün "doğru" olduğuna dair HİÇBİR seçim
                               yapılmaz -- sadece tekrar ölçüm bir aday
                               olarak temsil edilir.

TASARIM KARARI -- NEDEN LABORATUVAR İÇİN "MISSING_MEASUREMENT" YOK:
Faz 3'te LaboratoryResult.test_name serbest metin/genişletilebilir olarak
tasarlandı (sabit bir "beklenen panel" yok). Bu yüzden "hangi laboratuvar
testi eksik" sorusunu cevaplamak, olmayan bir laboratuvar panelini İCAT
ETMEYİ gerektirirdi -- bu, "Do NOT invent medical knowledge" kuralını ihlal
eder. Bunun yerine, laboratuvar tarafında sadece INSUFFICIENT_FOR_TREND ve
CONFLICT_RESOLUTION adayları üretilir (zaten kayıtlı bir test için).

MİMARİ:
    ClinicalAssessment + ClinicalReasoning (+ opsiyonel DifferentialAssessment)
    -> MeasurementCandidate listesi (NextBestMeasurementAssessment)
    -> (Bölüm B) ResourceRegistry ile eşleştirilerek -> MeasurementFeasibility

Faz 3/4/5/6/7 dosyaları DEĞİŞTİRİLMEDİ; sadece OKUNUR (kompozisyon).
medical/resource_availability.py'ye bağımlıdır (tek yönlü: bu modül onu
kullanır, o modül bunu KULLANMAZ).

DETERMİNİZM:
  - Sistem saati (wall-clock time), rastgele sayı üretimi veya otomatik
    üretilen benzersiz kimlikler KULLANILMAZ.
  - candidate_id, sabit alanlardan (aday türü + etiket) deterministik
    olarak üretilir.
  - Aday sırası: (1) MISSING_MEASUREMENT (VitalSignType kanonik tanım
    sırası), (2) INSUFFICIENT_FOR_TREND (vital kanonik sıra, sonra lab
    alfabetik sıra), (3) CONFLICT_RESOLUTION (aynı sıra deseni). Bu, Faz
    5'in zaten deterministik ürettiği vital_sign_series/
    laboratory_result_series sözlük sırasına DAYANIR, yeniden sıralama
    YAPILMAZ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from medical.clinical_assessment import ClinicalAssessment, MeasurementSeries, Trend, build_clinical_assessment
from medical.clinical_reasoning import (
    ClinicalReasoning,
    ReasoningCategory,
    ReasoningItem,
    ReasoningItemKind,
    build_clinical_reasoning,
)
from medical.differential_assessment import DifferentialAssessment
from medical.models.errors import MedicalDataValidationError
from medical.models.vital_sign import VitalSignType
from medical.patient_record import PatientRecord
from medical.resource_availability import ResourceRegistry, ResourceState

_BLOOD_PRESSURE_LABEL_PREFIX = "blood_pressure_"  # bkz. clinical_assessment.py etiketleme deseni


class MeasurementConsiderationReason(str, Enum):
    """
    Bir ölçümün neden aday olarak temsil edildiği -- SADECE veri-tamlığı
    nedenleridir, hiçbir klinik gerekçe İÇERMEZ.
    """
    MISSING_MEASUREMENT = "missing_measurement"
    INSUFFICIENT_FOR_TREND = "insufficient_for_trend"
    CONFLICT_RESOLUTION = "conflict_resolution"


class FeasibilityStatus(str, Enum):
    """Bir adayın, mevcut kaynak durumuna göre yapılabilir olup olmadığı."""
    FEASIBLE = "feasible"
    NOT_FEASIBLE = "not_feasible"
    UNKNOWN = "unknown"


@dataclass
class MeasurementCandidate:
    """
    TEK bir ölçüm adayının tamamen yapısal temsili. Hiçbir klinik
    öncelik/skor alanı YOKTUR -- sadece hangi ölçümün hangi VERİ-TAMLIĞI
    nedeniyle aday olduğu, hangi kanıta dayandığı ve hangi kaynağı
    gerektirdiği temsil edilir.
    """
    candidate_id: str
    measurement_label: str            # örn. "temperature", "glucose"
    measurement_kind: str             # "vital_sign" | "laboratory"
    reason: MeasurementConsiderationReason
    rationale: str                    # yapısal, klinik iddia İÇERMEYEN açıklama
    patient_id: str
    required_resource_id: str
    supporting_evidence: list[ReasoningItem] = field(default_factory=list)
    related_missing_information: list[ReasoningItem] = field(default_factory=list)
    related_concern_ids: list[str] = field(default_factory=list)  # sadece DifferentialAssessment'a çapraz-referans


@dataclass
class MeasurementFeasibility:
    """
    Bir MeasurementCandidate'ın, verilen bir ResourceRegistry'ye göre
    yapılabilirlik durumu. UNKNOWN, ASLA NOT_FEASIBLE olarak
    YORUMLANMAZ -- kaynağın durumu bilinmiyorsa, yapılabilirlik de
    bilinmez (bkz. resource_availability.py).
    """
    candidate: MeasurementCandidate
    patient_id: str
    required_resource_id: str
    resource_state: ResourceState
    status: FeasibilityStatus
    limitation: Optional[str] = None


@dataclass
class NextBestMeasurementAssessment:
    """
    ClinicalAssessment/ClinicalReasoning üzerinden üretilen, tamamen
    yapısal/deterministik "sonraki ölçüm adayları" çıktısı. Hiçbir
    tanı/olasılık/öncelik-skoru alanı YOKTUR (bkz. modül docstring'i).
    """
    patient_id: str
    candidates: list[MeasurementCandidate] = field(default_factory=list)
    scope_limitations: list[str] = field(default_factory=list)


_SCOPE_LIMITATION_STATEMENTS = [
    "Bu katman hiçbir istatistiksel 'information gain' veya olasılıksal sıralama hesaplamaz; "
    "aday sırası sabit ve yapısaldır (önce hiç kaydedilmemiş ölçümler, sonra eğilim için "
    "yetersiz ölçümler, sonra çelişkili ölçümler) -- bu, klinik bir öncelik SIRALAMASI DEĞİLDİR.",
    "Bu depoda, hangi ölçümün klinik olarak en değerli olduğunu belirleyecek otoriter bir "
    "klinik kural tabanı mevcut değil -- bu yüzden sadece veri-tamlığı (data completeness) "
    "boşlukları temsil edilir.",
    "Eksik bir ölçüm, ilgili değerin normal, negatif veya gerçekte yok olduğu anlamına GELMEZ "
    "-- sadece bu hastanın yapısal verisinde henüz kaydedilmediğini gösterir.",
    "Bu katman hiçbir tanı koymaz, risk/olasılık hesaplamaz, tedavi veya ilaç önermez.",
    "Bir adayın yapılabilirliği, ayrı bir kaynak-farkındalık katmanına (resource availability) "
    "bağlıdır; gerekli kaynak mevcut değilse veya durumu bilinmiyorsa aday SESSİZCE KALDIRILMAZ "
    "-- kısıtlama açıkça temsil edilir.",
]


# ---------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------

def _vital_type_value_for_label(label: str) -> str:
    """
    clinical_assessment.py'nin blood_pressure_systolic/diastolic etiket
    ayrımını, tek bir VitalSignType değerine ("blood_pressure") geri
    eşler -- ölçüm CİHAZI ikisi için de tektir (aynı cihaz sistolik VE
    diyastolik ölçer).
    """
    if label.startswith(_BLOOD_PRESSURE_LABEL_PREFIX):
        return VitalSignType.BLOOD_PRESSURE.value
    return label


def _required_resource_id_for_vital(vital_type_value: str) -> str:
    return f"device:{vital_type_value}"


_LABORATORY_RESOURCE_ID = "laboratory_capability"


def _find_items(
    reasoning: ClinicalReasoning, category: ReasoningCategory, kind: ReasoningItemKind, label: str,
) -> list[ReasoningItem]:
    """reasoning.items ZATEN deterministik sırada -- burada yeniden sıralama yapılmaz."""
    return [
        item for item in reasoning.items
        if item.category == category and item.kind == kind
        and item.evidence and item.evidence[0].label == label
    ]


def _related_concern_ids(differential: Optional[DifferentialAssessment], label: str) -> list[str]:
    """
    Faz 7'nin trend::<label> concern_id deseniyle SADECE kimlik bazlı
    çapraz-referans yapar -- hiçbir yeni klinik çıkarım YAPILMAZ, sadece
    zaten var olan bir concern_id'ye işaret edilir.
    """
    if differential is None:
        return []
    expected_id = f"trend::{label}"
    return [c.concern_id for c in differential.possible_clinical_concerns if c.concern_id == expected_id]


# ---------------------------------------------------------------------
# Aday üretimi
# ---------------------------------------------------------------------

def _build_missing_measurement_candidates(
    patient_id: str, assessment: ClinicalAssessment, differential: Optional[DifferentialAssessment],
) -> list[MeasurementCandidate]:
    candidates: list[MeasurementCandidate] = []
    for vital_type in VitalSignType:  # enum tanım sırası -- deterministik
        if vital_type.value not in assessment.missing_vital_sign_types:
            continue
        candidates.append(MeasurementCandidate(
            candidate_id=f"missing::{vital_type.value}",
            measurement_label=vital_type.value,
            measurement_kind="vital_sign",
            reason=MeasurementConsiderationReason.MISSING_MEASUREMENT,
            rationale=(
                f"'{vital_type.value}' bu hasta için hiç kaydedilmemiş; bu yapısal bir veri "
                "boşluğudur, klinik bir bulgu DEĞİLDİR."
            ),
            patient_id=patient_id,
            required_resource_id=_required_resource_id_for_vital(vital_type.value),
            related_concern_ids=_related_concern_ids(differential, vital_type.value),
        ))
    return candidates


def _build_insufficient_trend_candidates(
    patient_id: str,
    series_by_label: dict[str, MeasurementSeries],
    reasoning: ClinicalReasoning,
    differential: Optional[DifferentialAssessment],
    *,
    measurement_kind: str,
    required_resource_id_fn,
) -> list[MeasurementCandidate]:
    candidates: list[MeasurementCandidate] = []
    for label, series in series_by_label.items():  # ZATEN deterministik sırada (bkz. clinical_assessment.py)
        if series.trend != Trend.INSUFFICIENT_DATA:
            continue
        supporting = _find_items(reasoning, ReasoningCategory.CHRONOLOGY, ReasoningItemKind.OBSERVED_FACT, label)
        missing = _find_items(
            reasoning, ReasoningCategory.DATA_LEVEL_TREND, ReasoningItemKind.MISSING_INFORMATION, label,
        )
        candidates.append(MeasurementCandidate(
            candidate_id=f"insufficient_trend::{label}",
            measurement_label=label,
            measurement_kind=measurement_kind,
            reason=MeasurementConsiderationReason.INSUFFICIENT_FOR_TREND,
            rationale=(
                f"'{label}' için sadece bir ölçüm mevcut; ek bir ölçüm, veri-düzeyinde bir "
                "eğilim hesaplanmasına imkan tanır. Bu bir veri-tamlığı gerekçesidir, klinik "
                "bir yargı DEĞİLDİR."
            ),
            patient_id=patient_id,
            required_resource_id=required_resource_id_fn(label),
            supporting_evidence=supporting,
            related_missing_information=missing,
            related_concern_ids=_related_concern_ids(differential, label),
        ))
    return candidates


def _build_conflict_resolution_candidates(
    patient_id: str,
    series_by_label: dict[str, MeasurementSeries],
    reasoning: ClinicalReasoning,
    differential: Optional[DifferentialAssessment],
    *,
    measurement_kind: str,
    required_resource_id_fn,
) -> list[MeasurementCandidate]:
    candidates: list[MeasurementCandidate] = []
    for label, series in series_by_label.items():  # ZATEN deterministik sırada
        if not series.has_conflicting_measurements:
            continue
        supporting = _find_items(
            reasoning, ReasoningCategory.MEASUREMENT_CONFLICT, ReasoningItemKind.DATA_INTERPRETATION, label,
        )
        missing_note = ReasoningItem(
            category=ReasoningCategory.SCOPE_LIMITATION,
            kind=ReasoningItemKind.MISSING_INFORMATION,
            statement=(
                f"'{label}' için hangi çelişkili ölçümün doğru olduğunu belirleyecek otoriter "
                "bir kural mevcut değil -- hiçbir değer otomatik olarak seçilmedi."
            ),
            patient_id=patient_id,
        )
        candidates.append(MeasurementCandidate(
            candidate_id=f"conflict_resolution::{label}",
            measurement_label=label,
            measurement_kind=measurement_kind,
            reason=MeasurementConsiderationReason.CONFLICT_RESOLUTION,
            rationale=(
                f"'{label}' için aynı zaman damgasında çelişkili değerler kaydedilmiş; ek bir "
                "ölçüm veriyi netleştirmeye yardımcı olabilir. Bu, önceki ölçümlerden hangisinin "
                "doğru olduğuna dair bir KARAR VERMEZ."
            ),
            patient_id=patient_id,
            required_resource_id=required_resource_id_fn(label),
            supporting_evidence=supporting,
            related_missing_information=[missing_note],
            related_concern_ids=_related_concern_ids(differential, label),
        ))
    return candidates


# ---------------------------------------------------------------------
# Ana giriş noktaları
# ---------------------------------------------------------------------

def build_next_best_measurement_assessment(
    assessment: ClinicalAssessment,
    reasoning: ClinicalReasoning,
    differential: Optional[DifferentialAssessment] = None,
) -> NextBestMeasurementAssessment:
    """
    ClinicalAssessment + ClinicalReasoning'den (+ opsiyonel olarak
    DifferentialAssessment'tan çapraz-referans için) deterministik bir
    NextBestMeasurementAssessment üretir.

    Aynı girdi için HER ZAMAN aynı sonucu üretir. HİÇBİR klinik öncelik
    hesaplanmaz -- sadece üç sabit veri-tamlığı kategorisi (bkz. modül
    docstring'i) sabit bir sırada temsil edilir.
    """
    if not isinstance(assessment, ClinicalAssessment):
        raise MedicalDataValidationError(f"assessment bir ClinicalAssessment nesnesi olmalı: {assessment!r}")
    if not isinstance(reasoning, ClinicalReasoning):
        raise MedicalDataValidationError(f"reasoning bir ClinicalReasoning nesnesi olmalı: {reasoning!r}")
    if reasoning.patient_id != assessment.patient_id:
        raise MedicalDataValidationError(
            f"reasoning ve assessment farklı hastalara ait: {reasoning.patient_id!r} != {assessment.patient_id!r}"
        )
    if differential is not None:
        if not isinstance(differential, DifferentialAssessment):
            raise MedicalDataValidationError(
                f"differential bir DifferentialAssessment nesnesi olmalı: {differential!r}"
            )
        if differential.patient_id != assessment.patient_id:
            raise MedicalDataValidationError(
                f"differential ve assessment farklı hastalara ait: "
                f"{differential.patient_id!r} != {assessment.patient_id!r}"
            )

    patient_id = assessment.patient_id
    candidates: list[MeasurementCandidate] = []

    candidates.extend(_build_missing_measurement_candidates(patient_id, assessment, differential))

    candidates.extend(_build_insufficient_trend_candidates(
        patient_id, assessment.vital_sign_series, reasoning, differential,
        measurement_kind="vital_sign",
        required_resource_id_fn=lambda label: _required_resource_id_for_vital(_vital_type_value_for_label(label)),
    ))
    candidates.extend(_build_insufficient_trend_candidates(
        patient_id, assessment.laboratory_result_series, reasoning, differential,
        measurement_kind="laboratory",
        required_resource_id_fn=lambda _label: _LABORATORY_RESOURCE_ID,
    ))

    candidates.extend(_build_conflict_resolution_candidates(
        patient_id, assessment.vital_sign_series, reasoning, differential,
        measurement_kind="vital_sign",
        required_resource_id_fn=lambda label: _required_resource_id_for_vital(_vital_type_value_for_label(label)),
    ))
    candidates.extend(_build_conflict_resolution_candidates(
        patient_id, assessment.laboratory_result_series, reasoning, differential,
        measurement_kind="laboratory",
        required_resource_id_fn=lambda _label: _LABORATORY_RESOURCE_ID,
    ))

    return NextBestMeasurementAssessment(
        patient_id=patient_id,
        candidates=candidates,
        scope_limitations=list(_SCOPE_LIMITATION_STATEMENTS),
    )


def build_next_best_measurement_assessment_from_record(
    record: PatientRecord, *, include_differential: bool = True,
) -> NextBestMeasurementAssessment:
    """
    Kolaylık fonksiyonu: PatientRecord -> ClinicalAssessment ->
    ClinicalReasoning -> (opsiyonel) DifferentialAssessment ->
    NextBestMeasurementAssessment zincirini kurar.
    """
    assessment = build_clinical_assessment(record)
    reasoning = build_clinical_reasoning(assessment)
    differential = None
    if include_differential:
        from medical.differential_assessment import build_differential_assessment
        differential = build_differential_assessment(reasoning, assessment)
    return build_next_best_measurement_assessment(assessment, reasoning, differential)


# ---------------------------------------------------------------------
# Bölüm B ile köprü: Kaynak Farkındalıklı Yapılabilirlik Kontrolü
# ---------------------------------------------------------------------

def assess_measurement_feasibility(
    candidate: MeasurementCandidate, registry: ResourceRegistry,
) -> MeasurementFeasibility:
    """
    Bir MeasurementCandidate'ı, verilen ResourceRegistry'ye göre
    değerlendirir. registry.get_state(), kayıtlı olmayan bir resource_id
    için HER ZAMAN UNKNOWN döner (bkz. resource_availability.py) -- bu
    yüzden burada da "bilinmiyor" ASLA "mevcut değil" olarak
    YORUMLANMAZ.
    """
    if not isinstance(candidate, MeasurementCandidate):
        raise MedicalDataValidationError(f"candidate bir MeasurementCandidate nesnesi olmalı: {candidate!r}")
    if not isinstance(registry, ResourceRegistry):
        raise MedicalDataValidationError(f"registry bir ResourceRegistry nesnesi olmalı: {registry!r}")

    state = registry.get_state(candidate.required_resource_id)
    if state == ResourceState.AVAILABLE:
        status = FeasibilityStatus.FEASIBLE
        limitation = None
    elif state == ResourceState.UNAVAILABLE:
        status = FeasibilityStatus.NOT_FEASIBLE
        limitation = (
            f"Gerekli kaynak '{candidate.required_resource_id}' şu an mevcut değil olarak işaretli."
        )
    else:
        status = FeasibilityStatus.UNKNOWN
        limitation = (
            f"Gerekli kaynak '{candidate.required_resource_id}' için durum bilgisi mevcut değil "
            "(kayıtlı değil) -- yapılabilirlik bilinmiyor."
        )

    return MeasurementFeasibility(
        candidate=candidate,
        patient_id=candidate.patient_id,
        required_resource_id=candidate.required_resource_id,
        resource_state=state,
        status=status,
        limitation=limitation,
    )


def assess_all_feasibility(
    next_best: NextBestMeasurementAssessment, registry: ResourceRegistry,
) -> list[MeasurementFeasibility]:
    """
    next_best.candidates ZATEN deterministik sırada -- bu sıra burada
    KORUNUR, yeniden sıralama yapılmaz.
    """
    return [assess_measurement_feasibility(c, registry) for c in next_best.candidates]
