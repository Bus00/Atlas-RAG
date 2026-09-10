"""
medical/clinical_assessment.py
----------------------------------
V2 Faz 5: Klinik Değerlendirme Temeli (Clinical Assessment Foundation).

BU KATMAN TANI KOYMAZ. Hiçbir şekilde:
  - hastalık tanısı
  - hastalık olasılığı/yüzdesi
  - klinik eşik değeri (örn. "ateş = 38°C üzeri anormal")
  - klinik referans aralığı
  - ilaç önerisi / dozaj / tedavi süresi
  - kontrendikasyon / ilaç etkileşimi
  - acil durum eşiği
üretmez veya içermez.

AMAÇ:
PatientRecord'daki (Faz 4) ham/yapısal veriyi -- gözlemler, vital bulgular,
laboratuvar sonuçları -- deterministik, açıklanabilir ve test edilebilir
şekilde ORGANİZE eden bir katman. "Klinik karar" değil, "veri düzenleme ve
saydam sunum" katmanıdır.

MİMARİ (proje talimatı gereği):
    Patient -> PatientRecord -> ClinicalAssessment -> (ileride) klinik
    çıkarım/differential/next-best-measurement katmanları.

Faz 3 modelleri ve Faz 4 PatientRecord DEĞİŞTİRİLMEDİ. Bu modül SADECE
PatientRecord'un mevcut get_observations()/get_vital_signs()/
get_laboratory_results() metodlarını okur, üstüne organizasyon katar.

TASARIM KARARI -- "PossibleClinicalConcern" BİLİNÇLİ OLARAK EKLENMEDİ:
Proje talimatı "possible conceptual structures" arasında PossibleClinicalConcern'i
örnek olarak sayıyor, ancak "yalnızca gerçekten uyuyorsa kullan" diyor. Bu
sınıf, doğası gereği bir veri noktasından bir "endişe/olasılık" ÇIKARIMI
yapmayı ima eder -- ki bu tam olarak Faz 5'in yasakladığı şeydir (hastalık
olasılığı, klinik eşik, tanı). Bu yüzden burada kasıtlı olarak
UYGULANMADI. Bunun yerine tüm "yorum" ReasoningTrace.data_interpretation
altında, açıkça "veri düzeyinde" olduğu belirtilerek ve HİÇBİR "concern"
nesnesi üretmeden temsil ediliyor. Klinik anlamda bir "concern" kavramı,
otoriter klinik kurallar tanımlandığında ileri bir fazda eklenmelidir.

DETERMİNİZM:
build_clinical_assessment(), aynı PatientRecord için HER ZAMAN aynı
ClinicalAssessment'ı üretir:
  - datetime.now() / herhangi bir "şu an" bilgisi KULLANILMAZ.
  - Çıktıdaki tüm koleksiyonlar ya sabit enum tanım sırasına (VitalSignType,
    ObservationCategory) ya da sorted() ile elde edilen deterministik bir
    string sırasına göre üretilir -- set/dict yineleme sırasının hash
    rastgeleleştirmesinden (PYTHONHASHSEED) etkilenmemesi için.
  - PatientRecord.get_*() zaten kararlı (stable) sıralama garantisi verir
    (bkz. medical/patient_record.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from medical.models.errors import MedicalDataValidationError
from medical.models.laboratory_result import LaboratoryResult
from medical.models.observation import Observation, ObservationCategory
from medical.models.vital_sign import VitalSign, VitalSignType
from medical.patient_record import PatientRecord

# Blood pressure, Faz 3'te tek bir sayısal 'value' yerine systolic/diastolic
# olarak temsil edilir (bkz. medical/models/vital_sign.py). Bu yüzden ölçüm
# serileri seviyesinde tek bir "blood_pressure" etiketi yerine iki ayrı
# etiket kullanılır -- her ikisi de sayısal bir seri olarak ele alınabilsin
# diye (tek, tutarlı bir MeasurementSeries arayüzü).
BLOOD_PRESSURE_SYSTOLIC_LABEL = "blood_pressure_systolic"
BLOOD_PRESSURE_DIASTOLIC_LABEL = "blood_pressure_diastolic"


class Trend(str, Enum):
    """
    SADECE veri düzeyinde, basit bir eğilim göstergesi -- klinik anlam
    İDDİA ETMEZ. İlk ve son kronolojik değer karşılaştırılarak hesaplanır
    (bilinçli olarak basit tutuldu; gürültülü/ara değerleri dikkate almaz --
    bu, Faz 5'in kapsam sınırıdır, gelecekte istatistiksel bir yaklaşım
    ayrı bir fazda eklenebilir).
    """
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class MeasurementPoint:
    """Tek bir sayısal ölçümün, kaynak/zaman bilgisiyle birlikte taşıyıcısı."""
    value: float
    timestamp: datetime
    source: str


@dataclass
class MeasurementSeries:
    """
    Tek bir etiket (örn. "temperature", "glucose",
    "blood_pressure_systolic") için kronolojik ölçüm serisi.

    trend: SADECE veri düzeyinde (bkz. Trend dokümantasyonu). Klinik
    "anormallik" iddiası YOKTUR.

    conflicting_timestamps: aynı timestamp'te birden fazla FARKLI değer
    kaydedilmişse (örn. iki farklı cihazdan çelişkili okuma), bu
    timestamp'ler burada listelenir. ClinicalAssessment bunları OTOMATİK
    olarak çözümlemez/seçmez -- sadece çelişkinin var olduğunu saydam
    şekilde bildirir.
    """
    label: str
    points: list[MeasurementPoint] = field(default_factory=list)
    trend: Trend = Trend.INSUFFICIENT_DATA
    conflicting_timestamps: list[datetime] = field(default_factory=list)

    @property
    def measurement_count(self) -> int:
        return len(self.points)

    @property
    def has_repeated_measurements(self) -> bool:
        return len(self.points) > 1

    @property
    def has_conflicting_measurements(self) -> bool:
        return len(self.conflicting_timestamps) > 0


@dataclass
class ReasoningTrace:
    """
    Saydam akıl yürütme izi. Dört kategori KASITLI olarak birbirinden
    ayrı tutulur (proje talimatı: "Observed facts, ATLAS interpretations,
    missing information, and unsupported conclusions must remain
    distinguishable"):

      observed             -- verideki doğrudan, ham gerçekler
                               (örn. "3 sıcaklık ölçümü mevcut")
      data_interpretation  -- SADECE veri düzeyinde çıkarım (örn. eğilim),
                               klinik yorum İÇERMEZ
      missing              -- hangi bilginin mevcut olmadığı (asla "yok" ==
                               "negatif" olarak yorumlanmaz, sadece
                               eksikliğin kendisi bildirilir)
      unsupported          -- bu değerlendirmeden ÇIKARILAMAYACAK sonuçlar
                               (tanı, olasılık, ilaç vb.) -- sabit,
                               her zaman eklenen uyarılar
    """
    observed: list[str] = field(default_factory=list)
    data_interpretation: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


@dataclass
class ClinicalAssessment:
    """
    PatientRecord üzerinden üretilen, tamamen yapısal/deterministik
    değerlendirme çıktısı. Hiçbir tanı/olasılık/eşik/ilaç alanı YOKTUR
    (bkz. modül docstring'i).
    """
    patient_id: str
    observations: list[Observation] = field(default_factory=list)
    vital_sign_series: dict[str, MeasurementSeries] = field(default_factory=dict)
    laboratory_result_series: dict[str, MeasurementSeries] = field(default_factory=dict)
    present_observation_categories: list[str] = field(default_factory=list)
    present_vital_sign_types: list[str] = field(default_factory=list)
    missing_vital_sign_types: list[str] = field(default_factory=list)
    present_laboratory_test_names: list[str] = field(default_factory=list)
    reasoning_trace: ReasoningTrace = field(default_factory=ReasoningTrace)


# ---------------------------------------------------------------------
# Sayısal değer çıkarma (Faz 3 modellerini DEĞİŞTİRMEDEN, sadece okuma)
# ---------------------------------------------------------------------

def _vital_sign_labeled_values(vital: VitalSign) -> list[tuple[str, float]]:
    """
    Bir VitalSign'ı, (etiket, sayısal_değer) çiftlerine açar.
    blood_pressure için İKİ çift döner (systolic + diastolic); diğer
    vital türleri için TEK çift döner.
    """
    if vital.vital_type == VitalSignType.BLOOD_PRESSURE:
        return [
            (BLOOD_PRESSURE_SYSTOLIC_LABEL, float(vital.systolic)),
            (BLOOD_PRESSURE_DIASTOLIC_LABEL, float(vital.diastolic)),
        ]
    return [(vital.vital_type.value, float(vital.value))]


def _compute_trend(points: list[MeasurementPoint]) -> Trend:
    """
    SADECE ilk ve son kronolojik değeri karşılaştırır. points, çağıran
    kod tarafından ZATEN timestamp'e göre artan sırada verilmiş olmalı
    (bkz. _build_series).
    """
    if len(points) < 2:
        return Trend.INSUFFICIENT_DATA
    first_value = points[0].value
    last_value = points[-1].value
    if last_value > first_value:
        return Trend.INCREASING
    if last_value < first_value:
        return Trend.DECREASING
    return Trend.STABLE


def _find_conflicting_timestamps(points: list[MeasurementPoint]) -> list[datetime]:
    """
    Aynı timestamp'te birden fazla FARKLI değer varsa, o timestamp'i
    çelişkili olarak işaretler. Hangi değerin "doğru" olduğuna dair
    HİÇBİR karar verilmez -- sadece çelişkinin varlığı bildirilir.
    """
    values_by_timestamp: dict[datetime, set[float]] = {}
    order: list[datetime] = []
    for point in points:
        if point.timestamp not in values_by_timestamp:
            values_by_timestamp[point.timestamp] = set()
            order.append(point.timestamp)
        values_by_timestamp[point.timestamp].add(point.value)

    # `order` zaten points'in (kararlı sıralanmış) sırasını izliyor --
    # dict/set yineleme sırasına güvenilmiyor, sadece üyelik kontrolü için
    # kullanılıyor.
    return [ts for ts in order if len(values_by_timestamp[ts]) > 1]


def _build_series(label: str, points: list[MeasurementPoint]) -> MeasurementSeries:
    # points zaten çağıran kod tarafından timestamp'e göre sıralanmış olarak
    # geliyor (PatientRecord.get_vital_signs()/get_laboratory_results()
    # kararlı sıralama garantisi verir).
    return MeasurementSeries(
        label=label,
        points=points,
        trend=_compute_trend(points),
        conflicting_timestamps=_find_conflicting_timestamps(points),
    )


# ---------------------------------------------------------------------
# Ana giriş noktası
# ---------------------------------------------------------------------

def build_clinical_assessment(record: PatientRecord) -> ClinicalAssessment:
    """
    Bir PatientRecord'dan deterministik bir ClinicalAssessment üretir.

    Aynı PatientRecord (aynı içerikle) için HER ZAMAN aynı sonucu üretir --
    datetime.now() veya başka bir "şu an" bilgisi kullanılmaz.

    Bu fonksiyon HİÇBİR klinik çıkarım yapmaz; sadece PatientRecord'daki
    veriyi kronolojik olarak organize eder ve saydam bir ReasoningTrace
    üretir.
    """
    if not isinstance(record, PatientRecord):
        raise MedicalDataValidationError(
            f"record bir PatientRecord nesnesi olmalı: {record!r}"
        )

    observations = record.get_observations()
    vital_signs = record.get_vital_signs()
    laboratory_results = record.get_laboratory_results()

    # --- vital sign serileri: sabit VitalSignType tanım sırasına göre ---
    vital_points_by_label: dict[str, list[MeasurementPoint]] = {}
    for vital in vital_signs:
        for label, numeric_value in _vital_sign_labeled_values(vital):
            vital_points_by_label.setdefault(label, []).append(
                MeasurementPoint(value=numeric_value, timestamp=vital.timestamp, source=vital.source)
            )

    canonical_vital_labels: list[str] = []
    for vital_type in VitalSignType:  # enum tanım sırası -- deterministik
        if vital_type == VitalSignType.BLOOD_PRESSURE:
            canonical_vital_labels.append(BLOOD_PRESSURE_SYSTOLIC_LABEL)
            canonical_vital_labels.append(BLOOD_PRESSURE_DIASTOLIC_LABEL)
        else:
            canonical_vital_labels.append(vital_type.value)

    vital_sign_series: dict[str, MeasurementSeries] = {}
    for label in canonical_vital_labels:
        if label in vital_points_by_label:
            vital_sign_series[label] = _build_series(label, vital_points_by_label[label])

    present_vital_sign_types: list[str] = []
    missing_vital_sign_types: list[str] = []
    for vital_type in VitalSignType:  # enum tanım sırası -- deterministik
        has_data = (
            (vital_type == VitalSignType.BLOOD_PRESSURE
             and (BLOOD_PRESSURE_SYSTOLIC_LABEL in vital_sign_series
                  or BLOOD_PRESSURE_DIASTOLIC_LABEL in vital_sign_series))
            or (vital_type != VitalSignType.BLOOD_PRESSURE and vital_type.value in vital_sign_series)
        )
        if has_data:
            present_vital_sign_types.append(vital_type.value)
        else:
            missing_vital_sign_types.append(vital_type.value)

    # --- laboratuvar serileri: test_name serbest metin -- sorted() ile
    #     deterministik sıra (enum yok, alfabetik sıra tek güvenilir seçenek) ---
    lab_points_by_name: dict[str, list[MeasurementPoint]] = {}
    for lab in laboratory_results:
        lab_points_by_name.setdefault(lab.test_name, []).append(
            MeasurementPoint(value=float(lab.value), timestamp=lab.timestamp, source=lab.source)
        )

    laboratory_result_series: dict[str, MeasurementSeries] = {}
    for test_name in sorted(lab_points_by_name.keys()):
        laboratory_result_series[test_name] = _build_series(test_name, lab_points_by_name[test_name])

    present_laboratory_test_names = sorted(lab_points_by_name.keys())

    # --- gözlem kategorileri: sabit ObservationCategory tanım sırasına göre ---
    present_categories_set = {obs.category for obs in observations}
    present_observation_categories = [
        category.value for category in ObservationCategory  # enum tanım sırası
        if category in present_categories_set
    ]

    reasoning_trace = _build_reasoning_trace(
        observations=observations,
        vital_sign_series=vital_sign_series,
        missing_vital_sign_types=missing_vital_sign_types,
        laboratory_result_series=laboratory_result_series,
    )

    return ClinicalAssessment(
        patient_id=record.patient_id,
        observations=observations,
        vital_sign_series=vital_sign_series,
        laboratory_result_series=laboratory_result_series,
        present_observation_categories=present_observation_categories,
        present_vital_sign_types=present_vital_sign_types,
        missing_vital_sign_types=missing_vital_sign_types,
        present_laboratory_test_names=present_laboratory_test_names,
        reasoning_trace=reasoning_trace,
    )


def _build_reasoning_trace(
    *,
    observations: list[Observation],
    vital_sign_series: dict[str, MeasurementSeries],
    missing_vital_sign_types: list[str],
    laboratory_result_series: dict[str, MeasurementSeries],
) -> ReasoningTrace:
    observed: list[str] = []
    data_interpretation: list[str] = []
    missing: list[str] = []
    unsupported: list[str] = []

    # --- observed: gözlemler ---
    if observations:
        observed.append(f"{len(observations)} gözlem mevcut.")
    else:
        missing.append("Hiç gözlem (observation) mevcut değil.")

    # --- observed / data_interpretation: vital bulgular (sabit sırayla) ---
    for label, series in vital_sign_series.items():
        observed.append(
            f"'{label}' için {series.measurement_count} ölçüm mevcut "
            f"(ilk: {series.points[0].timestamp.isoformat()}, "
            f"son: {series.points[-1].timestamp.isoformat()})."
        )
        if series.trend != Trend.INSUFFICIENT_DATA:
            data_interpretation.append(
                f"'{label}' ölçümleri veri düzeyinde '{series.trend.value}' eğilimi gösteriyor "
                "(klinik anlam ifade etmez)."
            )
        if series.has_conflicting_measurements:
            data_interpretation.append(
                f"'{label}' için aynı zaman damgasında çelişkili değerler tespit edildi "
                f"({len(series.conflicting_timestamps)} zaman damgası) -- otomatik olarak çözümlenmedi."
            )

    for label in missing_vital_sign_types:
        missing.append(f"'{label}' için hiç ölçüm mevcut değil.")

    # --- observed / data_interpretation: laboratuvar sonuçları (alfabetik) ---
    for test_name, series in laboratory_result_series.items():
        observed.append(
            f"'{test_name}' laboratuvar testi için {series.measurement_count} sonuç mevcut "
            f"(ilk: {series.points[0].timestamp.isoformat()}, "
            f"son: {series.points[-1].timestamp.isoformat()})."
        )
        if series.trend != Trend.INSUFFICIENT_DATA:
            data_interpretation.append(
                f"'{test_name}' sonuçları veri düzeyinde '{series.trend.value}' eğilimi gösteriyor "
                "(klinik anlam ifade etmez)."
            )
        if series.has_conflicting_measurements:
            data_interpretation.append(
                f"'{test_name}' için aynı zaman damgasında çelişkili değerler tespit edildi "
                f"({len(series.conflicting_timestamps)} zaman damgası) -- otomatik olarak çözümlenmedi."
            )

    if not laboratory_result_series:
        missing.append("Hiç laboratuvar sonucu mevcut değil.")

    # --- unsupported: sabit, her zaman eklenen uyarılar ---
    # "Eksik bilgi asla 'negatif' olarak yorumlanmaz" ilkesi (proje talimatı)
    # burada AÇIKÇA belirtiliyor, sessizce varsayılmıyor.
    unsupported.append(
        "Eksik bilgi (missing) hiçbir şekilde 'normal' veya 'negatif' bir bulgu "
        "olarak yorumlanmamalıdır -- sadece verinin mevcut olmadığını gösterir."
    )
    unsupported.append(
        "Bu değerlendirmeden hiçbir hastalık tanısı veya ayırıcı tanı çıkarılamaz."
    )
    unsupported.append(
        "Bu değerlendirme hiçbir hastalık olasılığı/yüzdesi veya klinik eşik hesaplamaz."
    )
    unsupported.append(
        "Bu değerlendirme hiçbir ilaç önerisi, dozaj veya tedavi süresi içermez."
    )

    return ReasoningTrace(
        observed=observed,
        data_interpretation=data_interpretation,
        missing=missing,
        unsupported=unsupported,
    )
