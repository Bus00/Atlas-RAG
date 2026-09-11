"""
medical/differential_assessment.py
----------------------------------------
V2 Faz 7: Ayırıcı Değerlendirme Temeli (Differential Assessment Foundation).

BU KATMAN TANI KOYMAZ. Bu depoda (repository) hiçbir OTORİTER KLİNİK KURAL
TABANI (authoritative clinical rule base / medical knowledge base) mevcut
değildir -- dolayısıyla bu katman hiçbir hastalık adı, klinik durum adı,
olasılık veya risk skoru ÜRETEMEZ ve ÜRETMEZ. Proje talimatı madde
"REPOSITORY SAFETY" gereği: "If the repository does not contain
authoritative medical rules, build the safe structural/architectural
foundation for differential assessment rather than fabricating medical
rules." -- bu modül TAM OLARAK bunu yapar.

"PossibleClinicalConcern" burada bir HASTALIK ADAYI DEĞİLDİR. Sadece,
ClinicalReasoning/ClinicalAssessment'ta zaten var olan bir VERİ PATERNİNİ
(örn. bir ölçüm serisinin veri-düzeyinde artış/azalış göstermesi, ya da bir
gözlem ile bir ölçümün aynı zaman damgasında kaydedilmiş olması) insan
gözden geçirmesi için İŞARETLEYEN, tamamen yapısal bir kayıttır. Hiçbir
klinik terim ("ateş", "enfeksiyon", "anormal" vb.) veya hastalık adı ASLA
üretilmez -- yalnızca etiket adları (örn. "temperature", "glucose") ve
veri-düzeyi eğilim kelimeleri (increasing/decreasing) kullanılır.

MİMARİ:
    Patient -> PatientRecord -> ClinicalAssessment -> ClinicalReasoning
    -> DifferentialAssessment -> (ileride) Next Best Measurement /
       Medication / Safety / ...

Bu modül, Faz 6'nın ClinicalReasoning'ini (izlenebilirlik/evidence için) VE
Faz 5'in ClinicalAssessment'ını (Trend gibi yapılandırılmış -- string
olmayan -- veriye erişim için) birlikte tüketir. Faz 3/4/5/6 dosyaları
DEĞİŞTİRİLMEDİ; sadece OKUNUR.

NEDEN HEM ClinicalReasoning HEM ClinicalAssessment?
ClinicalReasoning.items içindeki `statement` alanları insan-okunabilir
serbest metindir -- bunları geri ayrıştırıp (parse) "artış mı azalış mı"
çıkarmak kırılgan ve hataya açık olurdu. Bunun yerine, DOĞRU/yapılandırılmış
kaynak olan ClinicalAssessment.vital_sign_series[...].trend (bir Trend enum
değeri) kullanılır; ReasoningItem'lar ise izlenebilirlik (hangi kanıt bu
paterni destekliyor) için referans olarak kullanılır. Bu, "en temiz/en
küçük" uygulama ilkesiyle tutarlıdır -- metin ayrıştırma yerine zaten var
olan yapılandırılmış veriye dayanmak.

DETERMİNİZM:
  - datetime.now() / random KULLANILMAZ.
  - concern_id, sabit ve deterministik bir string olarak üretilir (örn.
    "trend::temperature", "cooccurrence::dizziness::temperature::2026-01-01T08:00:00"),
    UUID veya rastgele bir kimlik ÜRETİLMEZ.
  - Kaynak diziler (assessment.vital_sign_series / .laboratory_result_series
    / reasoning.items) ZATEN Faz 5/6 tarafından deterministik sırada
    üretilir (bkz. o modüllerin docstring'i); bu modül o sırayı KORUR,
    yeniden sıralama yapmaz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from medical.clinical_assessment import ClinicalAssessment, MeasurementSeries, Trend, build_clinical_assessment
from medical.clinical_reasoning import (
    ClinicalReasoning,
    ReasoningCategory,
    ReasoningItem,
    ReasoningItemKind,
    build_clinical_reasoning,
)
from medical.models.errors import MedicalDataValidationError
from medical.patient_record import PatientRecord


class UncertaintyLevel(str, Enum):
    """
    Bu, bir HASTALIK olasılığı/güveni DEĞİLDİR -- sadece bir veri
    paterninin ne kadar SAĞLAM/tutarlı olduğunu gösteren yapısal bir
    göstergedir. Bu depoda otoriter bir klinik kural tabanı olmadığı için,
    hiçbir PossibleClinicalConcern DATA_LEVEL_ONLY'nin ÜZERİNE çıkamaz --
    bu, mimari olarak dayatılan bir TAVAN'dır (bkz. build_differential_assessment).
    """
    CONFLICTING_DATA = "conflicting_data"  # paterni destekleyen veri, aynı zamanda çelişkili ölçümler içeriyor
    DATA_LEVEL_ONLY = "data_level_only"    # paternin kendisi tutarlı, ama klinik yorum için otoriter kural YOK


@dataclass
class PossibleClinicalConcern:
    """
    Bir HASTALIK ADAYI DEĞİLDİR. ClinicalReasoning/ClinicalAssessment'ta
    zaten var olan, insan gözden geçirmesi için işaretlenen tamamen
    yapısal bir veri paterni kaydıdır.

    supporting_evidence / contradicting_evidence / missing_information,
    aynı ReasoningItem tipini kullanır -- böylece her ikisi de aynı
    izlenebilirlik (provenance) garantisini taşır (patient_id, timestamp,
    source zaten ReasoningItem.evidence içinde mevcuttur).
    """
    concern_id: str
    description: str
    patient_id: str
    supporting_evidence: list[ReasoningItem] = field(default_factory=list)
    contradicting_evidence: list[ReasoningItem] = field(default_factory=list)
    missing_information: list[ReasoningItem] = field(default_factory=list)
    uncertainty: UncertaintyLevel = UncertaintyLevel.DATA_LEVEL_ONLY


@dataclass
class DifferentialAssessment:
    """
    ClinicalReasoning üzerinden üretilen, tamamen yapısal/deterministik
    "ayırıcı değerlendirme" çıktısı. Hiçbir tanı/olasılık/eşik/ilaç alanı
    YOKTUR (bkz. modül docstring'i). `possible_clinical_concerns` boş bir
    liste olabilir -- bu, "hasta sorunsuz" ANLAMINA GELMEZ, sadece bu
    katmanın mevcut veride işaretleyecek bir paternin bulunmadığını
    gösterir (bkz. scope_limitations).
    """
    patient_id: str
    possible_clinical_concerns: list[PossibleClinicalConcern] = field(default_factory=list)
    scope_limitations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Sabit, veriden BAĞIMSIZ kapsam uyarıları (her zaman eklenir)
# ---------------------------------------------------------------------

_NO_CLINICAL_RULE_BASE_STATEMENT = (
    "Bu deseni klinik olarak yorumlayacak, otoriter bir klinik kural tabanı "
    "(medical knowledge base) bu depoda mevcut değil -- bu yüzden bu paternin "
    "ötesinde hiçbir çıkarım yapılmaz."
)

_SCOPE_LIMITATION_STATEMENTS = [
    "Bu katmandan hiçbir hastalık tanısı veya ayırıcı tanı çıkarılamaz; "
    "'possible clinical concern' bir hastalık adayı DEĞİLDİR, sadece işaretlenmiş bir veri paternidir.",
    "Bu katman hiçbir hastalık olasılığı, risk yüzdesi veya klinik skor hesaplamaz.",
    "Bu katman hiçbir sonraki-en-iyi-ölçüm (next best measurement) kararı, ilaç önerisi, "
    "dozaj veya tedavi süresi içermez.",
    "Zamansal eş-oluşuma dayanan hiçbir kalıp, nedensellik iddiası taşımaz.",
    "Hiçbir olası ilgi alanı (possible clinical concern) tespit edilmemiş olması, hastanın "
    "sorunsuz olduğu anlamına GELMEZ -- sadece bu katmanın mevcut yapısal veride "
    "işaretleyebileceği bir desen bulunmadığını gösterir.",
    _NO_CLINICAL_RULE_BASE_STATEMENT,
]


def _missing_information_placeholder(patient_id: str) -> ReasoningItem:
    """
    Her concern'e eklenen, sabit bir MISSING_INFORMATION kaydı: otoriter
    klinik kuralların depoda mevcut olmaması, kendisi başlı başına bir
    "eksik bilgi" durumudur (bkz. proje talimatı: "build the safe
    structural/architectural foundation ... rather than fabricating
    medical rules").
    """
    return ReasoningItem(
        category=ReasoningCategory.SCOPE_LIMITATION,
        kind=ReasoningItemKind.MISSING_INFORMATION,
        statement=_NO_CLINICAL_RULE_BASE_STATEMENT,
        patient_id=patient_id,
    )


# ---------------------------------------------------------------------
# Ana giriş noktaları
# ---------------------------------------------------------------------

def build_differential_assessment(
    reasoning: ClinicalReasoning, assessment: ClinicalAssessment,
) -> DifferentialAssessment:
    """
    Bir ClinicalReasoning + onun kaynağı olan ClinicalAssessment'tan
    deterministik bir DifferentialAssessment üretir.

    Aynı girdi için HER ZAMAN aynı sonucu üretir -- datetime.now(), random
    veya başka bir non-deterministik kaynak KULLANILMAZ.

    Bu fonksiyon HİÇBİR klinik çıkarım yapmaz; sadece ClinicalReasoning'te
    zaten temsil edilen veri-düzeyi paternlerini (trend, zamansal
    eş-oluşum), insan gözden geçirmesi için "possible clinical concern"
    olarak işaretler.
    """
    if not isinstance(reasoning, ClinicalReasoning):
        raise MedicalDataValidationError(
            f"reasoning bir ClinicalReasoning nesnesi olmalı: {reasoning!r}"
        )
    if not isinstance(assessment, ClinicalAssessment):
        raise MedicalDataValidationError(
            f"assessment bir ClinicalAssessment nesnesi olmalı: {assessment!r}"
        )
    if reasoning.patient_id != assessment.patient_id:
        raise MedicalDataValidationError(
            "reasoning ve assessment farklı hastalara ait: "
            f"{reasoning.patient_id!r} != {assessment.patient_id!r}"
        )

    patient_id = reasoning.patient_id
    concerns: list[PossibleClinicalConcern] = []

    concerns.extend(_build_trend_concerns(
        patient_id, assessment.vital_sign_series, reasoning, evidence_kind="vital_sign_measurement",
    ))
    concerns.extend(_build_trend_concerns(
        patient_id, assessment.laboratory_result_series, reasoning, evidence_kind="laboratory_measurement",
    ))
    concerns.extend(_build_cooccurrence_concerns(patient_id, reasoning))

    return DifferentialAssessment(
        patient_id=patient_id,
        possible_clinical_concerns=concerns,
        scope_limitations=list(_SCOPE_LIMITATION_STATEMENTS),
    )


def build_differential_assessment_from_record(record: PatientRecord) -> DifferentialAssessment:
    """Kolaylık fonksiyonu: PatientRecord -> ClinicalAssessment -> ClinicalReasoning -> DifferentialAssessment."""
    assessment = build_clinical_assessment(record)
    reasoning = build_clinical_reasoning(assessment)
    return build_differential_assessment(reasoning, assessment)


# ---------------------------------------------------------------------
# Yardımcı üreticiler
# ---------------------------------------------------------------------

def _find_items(reasoning: ClinicalReasoning, category: ReasoningCategory, label: str) -> list[ReasoningItem]:
    """
    reasoning.items içinde, evidence[0].label'ı `label`e eşit olan ve
    kategorisi `category` olan kayıtları bulur. reasoning.items ZATEN
    deterministik sırada üretildiği için (bkz. clinical_reasoning.py),
    dönen liste de deterministiktir.
    """
    return [
        item for item in reasoning.items
        if item.category == category and item.evidence and item.evidence[0].label == label
    ]


def _build_trend_concerns(
    patient_id: str,
    series_by_label: dict[str, MeasurementSeries],
    reasoning: ClinicalReasoning,
    *,
    evidence_kind: str,
) -> list[PossibleClinicalConcern]:
    """
    SADECE increasing/decreasing eğilimler bir concern üretir -- stable ve
    insufficient_data, tanım gereği "değişmeyen" veya "yetersiz" bir
    paterndir, insan gözden geçirmesi için işaretlenecek bir "veri
    değişikliği" YOKTUR. series_by_label ZATEN Faz 5 tarafından
    deterministik sırada üretilmiştir (bkz. clinical_assessment.py) --
    burada yeniden sıralama yapılmaz.
    """
    concerns: list[PossibleClinicalConcern] = []
    for label, series in series_by_label.items():
        if series.trend not in (Trend.INCREASING, Trend.DECREASING):
            continue

        supporting = _find_items(reasoning, ReasoningCategory.CHRONOLOGY, label)
        supporting += [
            item for item in reasoning.items
            if item.category == ReasoningCategory.DATA_LEVEL_TREND
            and item.kind == ReasoningItemKind.DATA_INTERPRETATION
            and item.evidence and item.evidence[0].label == label
        ]
        contradicting = [
            item for item in reasoning.items
            if item.category == ReasoningCategory.MEASUREMENT_CONFLICT
            and item.evidence and item.evidence[0].label == label
        ]
        missing = [_missing_information_placeholder(patient_id)]
        uncertainty = UncertaintyLevel.CONFLICTING_DATA if contradicting else UncertaintyLevel.DATA_LEVEL_ONLY

        concerns.append(PossibleClinicalConcern(
            concern_id=f"trend::{label}",
            description=(
                f"'{label}' ölçüm serisi veri düzeyinde '{series.trend.value}' eğilimi gösteriyor. "
                "Bu, tek başına hiçbir klinik anlam taşımaz -- sadece insan gözden geçirmesi için "
                "işaretlenen bir veri değişikliği paternidir."
            ),
            patient_id=patient_id,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            missing_information=missing,
            uncertainty=uncertainty,
        ))
    return concerns


def _build_cooccurrence_concerns(patient_id: str, reasoning: ClinicalReasoning) -> list[PossibleClinicalConcern]:
    """
    reasoning.items içindeki her TEMPORAL_COOCCURRENCE kaydı, bire-bir bir
    concern'e dönüşür. reasoning.items ZATEN deterministik sırada
    üretildiği için (bkz. clinical_reasoning.py: gözlem kronolojik x vital
    kanonik x lab alfabetik), burada yeniden sıralama yapılmaz.
    """
    concerns: list[PossibleClinicalConcern] = []
    cooccurrence_items = [
        item for item in reasoning.items
        if item.category == ReasoningCategory.TEMPORAL_COOCCURRENCE
    ]
    for item in cooccurrence_items:
        obs_evidence, measurement_evidence = item.evidence[0], item.evidence[1]
        concern_id = (
            f"cooccurrence::{obs_evidence.label}::{measurement_evidence.label}::"
            f"{obs_evidence.timestamp.isoformat()}"
        )
        concerns.append(PossibleClinicalConcern(
            concern_id=concern_id,
            description=(
                f"'{obs_evidence.label}' gözlemi ile '{measurement_evidence.label}' ölçümü aynı zaman "
                f"damgasında kaydedildi. Bu, sadece zamansal eş-oluşumdur; nedensellik iddia edilmez -- "
                "insan gözden geçirmesi için işaretlenmiştir."
            ),
            patient_id=patient_id,
            supporting_evidence=[item],
            contradicting_evidence=[],
            missing_information=[_missing_information_placeholder(patient_id)],
            uncertainty=UncertaintyLevel.DATA_LEVEL_ONLY,
        ))
    return concerns
