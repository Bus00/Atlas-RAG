"""
medical/clinical_reasoning.py
----------------------------------
V2 Faz 6: Klinik Akıl Yürütme Temeli (Clinical Reasoning Foundation).

BU KATMAN TANI KOYMAZ. Hiçbir şekilde:
  - hastalık tanısı / ayırıcı tanı
  - hastalık olasılığı / risk yüzdesi / klinik skor
  - klinik eşik değeri / referans aralığı yorumu
  - ilaç önerisi / dozaj / tedavi süresi
  - kontrendikasyon / ilaç etkileşimi
  - acil durum eşiği / otonom klinik karar
  - nedensellik iddiası (temporal yakınlık asla "X, Y'ye neden oldu" gibi
    yorumlanmaz)
üretmez veya içermez.

AMAÇ:
Faz 5'in ürettiği ClinicalAssessment'ı (organize edilmiş, deterministik
veri) girdi olarak alıp, veri NOKTALARI ARASINDAKİ YAPISAL İLİŞKİLERİ
("iki ölçüm aynı zaman damgasında çelişiyor", "bir gözlem ve bir ölçüm
aynı anda kaydedildi", "N veri noktası aynı hastaya ait" vb.) açık, izlenebilir
ve test edilebilir bir nesne modeliyle temsil eden bir katmandır.

Bu, "klinik akıl yürütme" DEĞİL -- "veri ilişkilerini şeffaf şekilde temsil
etme" katmanıdır. Faz talimatındaki ayrım: bu Faz 6, tanı/olasılık/tedavi
üretmeyen bir TEMELdir; gelecekteki Differential Assessment / Next Best
Measurement / Medication katmanları bunun üzerine inşa edilecektir (henüz
uygulanmadı, kasıtlı olarak).

MİMARİ:
    Patient -> PatientRecord -> ClinicalAssessment -> ClinicalReasoning
    -> (ileride) Differential Assessment / Next Best Measurement / ...

Faz 3/4/5 modelleri DEĞİŞTİRİLMEDİ. Bu modül SADECE ClinicalAssessment'ın
mevcut alanlarını (observations, vital_sign_series, laboratory_result_series,
missing_vital_sign_types, ...) okur; kompozisyon tercih edildi, değişiklik
değil.

DETERMİNİZM (proje talimatı gereği):
  - datetime.now() veya başka bir "şu an" bilgisi KULLANILMAZ.
  - random KULLANILMAZ, rastgele ID üretilmez.
  - Çıktı sırası set/dict hash yineleme sırasına GÜVENMEZ. ClinicalAssessment
    (Faz 5) zaten vital_sign_series / laboratory_result_series sözlüklerini
    deterministik bir sırayla (sabit enum tanım sırası / alfabetik sıra)
    doldurur ve normal Python dict'leri ekleme sırasını korur (Python 3.7+
    garantisi, hash rastgeleleştirmesinden ETKİLENMEZ) -- bu modül o
    garantiye dayanarak assessment.vital_sign_series.items() /
    assessment.laboratory_result_series.items() üzerinde DOĞRUDAN yineleme
    yapar, yeniden sıralama yapmaz.
  - Gözlem/ölçüm eşleşmeleri (temporal co-occurrence) iç içe geçmiş, sabit
    sıralı döngülerle üretilir (bkz. _build_temporal_cooccurrence_items).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from medical.clinical_assessment import ClinicalAssessment, MeasurementSeries, Trend, build_clinical_assessment
from medical.models.errors import MedicalDataValidationError
from medical.patient_record import PatientRecord


class ReasoningItemKind(str, Enum):
    """
    Faz 5'teki ReasoningTrace'in dört kategorisiyle aynı ayrımı, tekil
    (itemized) seviyede korur -- her ReasoningItem, hangi türden bir
    ifade olduğunu AÇIKÇA taşır (düz metin listesi değil, yapılandırılmış
    nesne).
    """
    OBSERVED_FACT = "observed_fact"                # doğrudan veriden okunan ham gerçek
    DATA_INTERPRETATION = "data_interpretation"     # SADECE veri düzeyinde ilişki (klinik yorum değil)
    MISSING_INFORMATION = "missing_information"     # eksik bilginin kendisi (asla "negatif" değildir)
    UNSUPPORTED_CONCLUSION = "unsupported_conclusion"  # bu katmandan ÇIKARILAMAYACAK sonuçlar


class ReasoningCategory(str, Enum):
    """Temsil edilen veri-ilişkisi türü. Klinik bir sınıflandırma DEĞİLDİR."""
    DATA_AVAILABILITY = "data_availability"           # bir kategorinin mevcut/eksik olması
    CHRONOLOGY = "chronology"                          # kronolojik sıralama
    REPEATED_MEASUREMENT = "repeated_measurement"       # aynı etiket için birden fazla kayıt
    DATA_LEVEL_TREND = "data_level_trend"               # artış/azalış/sabit (SADECE veri düzeyi)
    MEASUREMENT_CONFLICT = "measurement_conflict"       # aynı zaman damgasında farklı değerler
    TEMPORAL_COOCCURRENCE = "temporal_cooccurrence"     # gözlem + ölçüm aynı zaman damgasında
    PATIENT_LINKAGE = "patient_linkage"                 # verilerin aynı hastaya ait olması
    SCOPE_LIMITATION = "scope_limitation"               # bu katmanın kapsam dışı bıraktığı şeyler


@dataclass
class ReasoningEvidence:
    """
    Bir ReasoningItem'ın dayandığı ham veri noktasına işaret eden, hafif
    bir izlenebilirlik (provenance) referansı. Veriyi KOPYALAMAZ/yeniden
    yorumlamaz -- sadece hangi kayda atıfta bulunulduğunu gösterir.
    """
    kind: str  # örn. "observation", "vital_sign_measurement", "laboratory_measurement"
    label: Optional[str] = None       # örn. "temperature", "glucose", gözlem kategorisi değeri
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    value: Optional[float] = None     # sadece sayısal ölçümler için; yorumlanmamış ham değer


@dataclass
class ReasoningItem:
    """
    Tek bir yapısal veri-ilişkisi ifadesi. `statement`, insan tarafından
    okunabilir, deterministik bir açıklamadır -- ama nesnenin kendisi
    (`category`, `kind`, `evidence`) programatik olarak da işlenebilir.
    """
    category: ReasoningCategory
    kind: ReasoningItemKind
    statement: str
    patient_id: str
    evidence: list[ReasoningEvidence] = field(default_factory=list)


@dataclass
class ClinicalReasoning:
    """
    ClinicalAssessment üzerinden üretilen, tamamen yapısal/deterministik
    akıl yürütme çıktısı. Hiçbir tanı/olasılık/eşik/ilaç alanı YOKTUR
    (bkz. modül docstring'i).
    """
    patient_id: str
    items: list[ReasoningItem] = field(default_factory=list)

    def items_by_kind(self, kind: ReasoningItemKind) -> list[ReasoningItem]:
        return [item for item in self.items if item.kind == kind]

    @property
    def observed(self) -> list[ReasoningItem]:
        return self.items_by_kind(ReasoningItemKind.OBSERVED_FACT)

    @property
    def data_interpretation(self) -> list[ReasoningItem]:
        return self.items_by_kind(ReasoningItemKind.DATA_INTERPRETATION)

    @property
    def missing(self) -> list[ReasoningItem]:
        return self.items_by_kind(ReasoningItemKind.MISSING_INFORMATION)

    @property
    def unsupported(self) -> list[ReasoningItem]:
        return self.items_by_kind(ReasoningItemKind.UNSUPPORTED_CONCLUSION)


# ---------------------------------------------------------------------
# Ana giriş noktaları
# ---------------------------------------------------------------------

def build_clinical_reasoning(assessment: ClinicalAssessment) -> ClinicalReasoning:
    """
    Bir ClinicalAssessment'tan deterministik bir ClinicalReasoning üretir.

    Aynı ClinicalAssessment (aynı içerikle) için HER ZAMAN aynı sonucu
    üretir -- datetime.now(), random veya başka bir non-deterministik
    kaynak KULLANILMAZ.

    Bu fonksiyon HİÇBİR klinik çıkarım yapmaz; sadece ClinicalAssessment'taki
    veri noktaları arasındaki YAPISAL ilişkileri (kronoloji, tekrar, eğilim,
    çelişki, eksiklik, zamansal eş-oluşum, hasta bağlantısı) temsil eder.
    """
    if not isinstance(assessment, ClinicalAssessment):
        raise MedicalDataValidationError(
            f"assessment bir ClinicalAssessment nesnesi olmalı: {assessment!r}"
        )

    patient_id = assessment.patient_id
    items: list[ReasoningItem] = []

    total_facts = (
        len(assessment.observations)
        + sum(series.measurement_count for series in assessment.vital_sign_series.values())
        + sum(series.measurement_count for series in assessment.laboratory_result_series.values())
    )

    items.extend(_build_patient_linkage_items(patient_id, total_facts))
    items.extend(_build_observation_items(patient_id, assessment))
    items.extend(_build_series_items(
        patient_id, assessment.vital_sign_series, evidence_kind="vital_sign_measurement",
    ))
    items.extend(_build_missing_vital_items(patient_id, assessment.missing_vital_sign_types))
    items.extend(_build_series_items(
        patient_id, assessment.laboratory_result_series, evidence_kind="laboratory_measurement",
    ))
    if not assessment.laboratory_result_series:
        items.append(ReasoningItem(
            category=ReasoningCategory.DATA_AVAILABILITY,
            kind=ReasoningItemKind.MISSING_INFORMATION,
            statement="Hiç laboratuvar sonucu mevcut değil.",
            patient_id=patient_id,
        ))
    items.extend(_build_temporal_cooccurrence_items(patient_id, assessment))
    items.extend(_build_scope_limitation_items(patient_id))

    return ClinicalReasoning(patient_id=patient_id, items=items)


def build_clinical_reasoning_from_record(record: PatientRecord) -> ClinicalReasoning:
    """Kolaylık fonksiyonu: PatientRecord -> ClinicalAssessment -> ClinicalReasoning zincirini kurar."""
    assessment = build_clinical_assessment(record)
    return build_clinical_reasoning(assessment)


# ---------------------------------------------------------------------
# Yardımcı üreticiler (her biri tek bir ilişki türünden sorumlu)
# ---------------------------------------------------------------------

def _build_patient_linkage_items(patient_id: str, total_facts: int) -> list[ReasoningItem]:
    if total_facts == 0:
        return [ReasoningItem(
            category=ReasoningCategory.PATIENT_LINKAGE,
            kind=ReasoningItemKind.MISSING_INFORMATION,
            statement="Bu hasta için hiçbir yapısal veri (gözlem/vital/lab) mevcut değil.",
            patient_id=patient_id,
        )]
    return [ReasoningItem(
        category=ReasoningCategory.PATIENT_LINKAGE,
        kind=ReasoningItemKind.OBSERVED_FACT,
        statement=f"{total_facts} veri noktası (gözlem/vital/lab) aynı hastayla (patient_id) ilişkilendirildi.",
        patient_id=patient_id,
    )]


def _build_observation_items(patient_id: str, assessment: ClinicalAssessment) -> list[ReasoningItem]:
    if not assessment.observations:
        return [ReasoningItem(
            category=ReasoningCategory.DATA_AVAILABILITY,
            kind=ReasoningItemKind.MISSING_INFORMATION,
            statement="Hiç gözlem (observation) mevcut değil.",
            patient_id=patient_id,
        )]

    items: list[ReasoningItem] = []
    # present_observation_categories zaten ObservationCategory enum tanım
    # sırasına göre deterministik olarak üretildi (bkz. clinical_assessment.py).
    for category_value in assessment.present_observation_categories:
        matching = [obs for obs in assessment.observations if obs.category.value == category_value]
        evidence = [
            ReasoningEvidence(kind="observation", label=category_value, timestamp=obs.timestamp, source=obs.source)
            for obs in matching
        ]
        items.append(ReasoningItem(
            category=ReasoningCategory.DATA_AVAILABILITY,
            kind=ReasoningItemKind.OBSERVED_FACT,
            statement=f"'{category_value}' kategorisinde {len(matching)} gözlem mevcut.",
            patient_id=patient_id,
            evidence=evidence,
        ))
    return items


def _build_series_items(
    patient_id: str, series_by_label: dict[str, MeasurementSeries], *, evidence_kind: str,
) -> list[ReasoningItem]:
    """
    series_by_label ZATEN Faz 5 tarafından deterministik sırada üretilmiştir
    (bkz. modül docstring'i) -- burada yeniden sıralama yapılmaz, doğrudan
    .items() üzerinde yinelenir (normal dict, ekleme sırasını korur).
    """
    items: list[ReasoningItem] = []
    for label, series in series_by_label.items():
        evidence = [
            ReasoningEvidence(kind=evidence_kind, label=label, timestamp=p.timestamp, source=p.source, value=p.value)
            for p in series.points
        ]
        items.append(ReasoningItem(
            category=ReasoningCategory.CHRONOLOGY,
            kind=ReasoningItemKind.OBSERVED_FACT,
            statement=(
                f"'{label}' için {series.measurement_count} ölçüm kronolojik sırada mevcut "
                f"(ilk: {series.points[0].timestamp.isoformat()}, son: {series.points[-1].timestamp.isoformat()})."
            ),
            patient_id=patient_id,
            evidence=evidence,
        ))

        if series.has_repeated_measurements:
            items.append(ReasoningItem(
                category=ReasoningCategory.REPEATED_MEASUREMENT,
                kind=ReasoningItemKind.OBSERVED_FACT,
                statement=f"'{label}' için {series.measurement_count} tekrarlı ölçüm mevcut.",
                patient_id=patient_id,
                evidence=evidence,
            ))

        if series.trend != Trend.INSUFFICIENT_DATA:
            items.append(ReasoningItem(
                category=ReasoningCategory.DATA_LEVEL_TREND,
                kind=ReasoningItemKind.DATA_INTERPRETATION,
                statement=(
                    f"'{label}' ölçümleri veri düzeyinde '{series.trend.value}' eğilimi gösteriyor "
                    "(klinik anlam ifade etmez)."
                ),
                patient_id=patient_id,
                evidence=evidence,
            ))
        else:
            items.append(ReasoningItem(
                category=ReasoningCategory.DATA_LEVEL_TREND,
                kind=ReasoningItemKind.MISSING_INFORMATION,
                statement=(
                    f"'{label}' için eğilim belirlemeye yetecek kadar ölçüm yok "
                    "(en az 2 ölçüm gerekir)."
                ),
                patient_id=patient_id,
                evidence=evidence,
            ))

        if series.has_conflicting_measurements:
            conflict_evidence = [
                ReasoningEvidence(kind=evidence_kind, label=label, timestamp=p.timestamp, source=p.source, value=p.value)
                for p in series.points
                if p.timestamp in series.conflicting_timestamps
            ]
            items.append(ReasoningItem(
                category=ReasoningCategory.MEASUREMENT_CONFLICT,
                kind=ReasoningItemKind.DATA_INTERPRETATION,
                statement=(
                    f"'{label}' için aynı zaman damgasında çelişkili değerler tespit edildi "
                    f"({len(series.conflicting_timestamps)} zaman damgası) -- her iki ölçüm de korunuyor, "
                    "otomatik olarak bir tanesi seçilmedi/çözümlenmedi."
                ),
                patient_id=patient_id,
                evidence=conflict_evidence,
            ))
    return items


def _build_missing_vital_items(patient_id: str, missing_vital_sign_types: list[str]) -> list[ReasoningItem]:
    # missing_vital_sign_types zaten VitalSignType enum tanım sırasına göre
    # deterministik (bkz. clinical_assessment.py).
    return [
        ReasoningItem(
            category=ReasoningCategory.DATA_AVAILABILITY,
            kind=ReasoningItemKind.MISSING_INFORMATION,
            statement=f"'{label}' için hiç ölçüm mevcut değil.",
            patient_id=patient_id,
        )
        for label in missing_vital_sign_types
    ]


def _build_temporal_cooccurrence_items(patient_id: str, assessment: ClinicalAssessment) -> list[ReasoningItem]:
    """
    Bir gözlem ile bir ölçümün AYNI zaman damgasında kaydedilmiş olması,
    SADECE zamansal eş-oluşum (temporal co-occurrence) olarak temsil edilir.
    HİÇBİR nedensellik iddiası içermez -- statement metni bunu açıkça belirtir.

    Deterministik sıralama: gözlemler zaten kronolojik (assessment.observations),
    vital/lab serileri zaten kanonik sırada -- iç içe döngü, tamamen sabit bir
    sırayla üretilir: gözlem (kronolojik) x vital etiket (kanonik) x lab etiket
    (alfabetik), her birinde ölçüm noktaları kronolojik.
    """
    items: list[ReasoningItem] = []
    for obs in assessment.observations:
        for label, series in assessment.vital_sign_series.items():
            for point in series.points:
                if point.timestamp == obs.timestamp:
                    items.append(ReasoningItem(
                        category=ReasoningCategory.TEMPORAL_COOCCURRENCE,
                        kind=ReasoningItemKind.OBSERVED_FACT,
                        statement=(
                            f"'{obs.category.value}' gözlemi ile '{label}' ölçümü aynı zaman damgasında "
                            f"({obs.timestamp.isoformat()}) kaydedildi. Bu, zamansal eş-oluşumdur; "
                            "nedensellik iddia edilmez."
                        ),
                        patient_id=patient_id,
                        evidence=[
                            ReasoningEvidence(kind="observation", label=obs.category.value, timestamp=obs.timestamp, source=obs.source),
                            ReasoningEvidence(kind="vital_sign_measurement", label=label, timestamp=point.timestamp, source=point.source, value=point.value),
                        ],
                    ))
        for test_name, series in assessment.laboratory_result_series.items():
            for point in series.points:
                if point.timestamp == obs.timestamp:
                    items.append(ReasoningItem(
                        category=ReasoningCategory.TEMPORAL_COOCCURRENCE,
                        kind=ReasoningItemKind.OBSERVED_FACT,
                        statement=(
                            f"'{obs.category.value}' gözlemi ile '{test_name}' laboratuvar sonucu aynı zaman "
                            f"damgasında ({obs.timestamp.isoformat()}) kaydedildi. Bu, zamansal eş-oluşumdur; "
                            "nedensellik iddia edilmez."
                        ),
                        patient_id=patient_id,
                        evidence=[
                            ReasoningEvidence(kind="observation", label=obs.category.value, timestamp=obs.timestamp, source=obs.source),
                            ReasoningEvidence(kind="laboratory_measurement", label=test_name, timestamp=point.timestamp, source=point.source, value=point.value),
                        ],
                    ))
    return items


def _build_scope_limitation_items(patient_id: str) -> list[ReasoningItem]:
    """
    Sabit, veriden BAĞIMSIZ, her zaman eklenen kapsam-sınırı uyarıları.
    Boş bir hasta kaydı bile yanlışlıkla "sorun yok" olarak okunmasın diye
    HER ZAMAN mevcutturlar.
    """
    statements = [
        "Bu akıl yürütme katmanından hiçbir hastalık tanısı veya ayırıcı tanı çıkarılamaz.",
        "Bu katman hiçbir hastalık olasılığı, risk yüzdesi veya klinik skor hesaplamaz.",
        "Zamansal yakınlık (temporal proximity) nedensellik anlamına gelmez; korelasyon nedensellik değildir.",
        "Bu katman hiçbir ilaç önerisi, dozaj veya tedavi süresi içermez.",
        "Eksik bilgi (missing) hiçbir şekilde normal veya negatif bir bulgu olarak yorumlanmamalıdır.",
    ]
    return [
        ReasoningItem(
            category=ReasoningCategory.SCOPE_LIMITATION,
            kind=ReasoningItemKind.UNSUPPORTED_CONCLUSION,
            statement=statement,
            patient_id=patient_id,
        )
        for statement in statements
    ]
