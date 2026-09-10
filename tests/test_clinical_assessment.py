"""
tests/test_clinical_assessment.py
--------------------------------------
V2 Faz 5: medical/clinical_assessment.py (ClinicalAssessment) testleri.

Bu testler yalnızca YAPISAL organizasyon, deterministik sıralama/eğilim ve
"observed / interpretation / missing / unsupported" ayrımını kontrol eder --
hiçbir klinik çıkarım test edilmez, çünkü katmanın kendisi hiçbir klinik
çıkarım yapmaz (bkz. medical/clinical_assessment.py docstring'i).
"""
from __future__ import annotations

from datetime import datetime

import pytest

from medical.clinical_assessment import (
    BLOOD_PRESSURE_DIASTOLIC_LABEL,
    BLOOD_PRESSURE_SYSTOLIC_LABEL,
    ClinicalAssessment,
    MeasurementPoint,
    MeasurementSeries,
    ReasoningTrace,
    Trend,
    build_clinical_assessment,
)
from medical.models.errors import MedicalDataValidationError
from medical.models.laboratory_result import LaboratoryResult
from medical.models.observation import Observation, ObservationCategory
from medical.models.patient import Patient
from medical.models.vital_sign import VitalSign, VitalSignType
from medical.patient_record import PatientRecord

T1 = datetime(2026, 1, 1, 8, 0, 0)
T2 = datetime(2026, 1, 1, 12, 0, 0)
T3 = datetime(2026, 1, 1, 18, 0, 0)


def _patient(patient_id: str = "P-200") -> Patient:
    return Patient(patient_id=patient_id)


def _record(patient_id: str = "P-200") -> PatientRecord:
    return PatientRecord(patient=_patient(patient_id))


def _observation(ts: datetime, category: ObservationCategory = ObservationCategory.SYMPTOM,
                  description: str = "hasta ateş bildiriyor", source: str = "patient_report") -> Observation:
    return Observation(category=category, description=description, timestamp=ts, source=source)


def _temp(ts: datetime, value: float, source: str = "thermometer") -> VitalSign:
    return VitalSign(vital_type=VitalSignType.TEMPERATURE, timestamp=ts, source=source, unit="°C", value=value)


def _spo2(ts: datetime, value: float, source: str = "pulse_oximeter") -> VitalSign:
    return VitalSign(vital_type=VitalSignType.SPO2, timestamp=ts, source=source, unit="%", value=value)


def _bp(ts: datetime, systolic: float, diastolic: float, source: str = "bp_monitor") -> VitalSign:
    return VitalSign(
        vital_type=VitalSignType.BLOOD_PRESSURE, timestamp=ts, source=source, unit="mmHg",
        systolic=systolic, diastolic=diastolic,
    )


def _glucose(ts: datetime, value: float, source: str = "onboard_lab") -> LaboratoryResult:
    return LaboratoryResult(test_name="glucose", value=value, unit="mg/dL", timestamp=ts, source=source)


# ---------------------------------------------------------------------
# 1. Empty PatientRecord
# ---------------------------------------------------------------------

def test_empty_patient_record_produces_empty_assessment():
    assessment = build_clinical_assessment(_record())
    assert assessment.patient_id == "P-200"
    assert assessment.observations == []
    assert assessment.vital_sign_series == {}
    assert assessment.laboratory_result_series == {}
    assert assessment.present_observation_categories == []
    assert assessment.present_vital_sign_types == []
    assert assessment.present_laboratory_test_names == []


def test_empty_patient_record_reports_all_vital_types_missing():
    assessment = build_clinical_assessment(_record())
    assert set(assessment.missing_vital_sign_types) == {t.value for t in VitalSignType}


def test_build_clinical_assessment_rejects_non_patient_record():
    with pytest.raises(MedicalDataValidationError):
        build_clinical_assessment("not a record")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# 2. Assessment from observations
# ---------------------------------------------------------------------

def test_assessment_includes_observations():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.PAIN))
    assessment = build_clinical_assessment(record)
    assert len(assessment.observations) == 1
    assert assessment.observations[0].category == ObservationCategory.PAIN


def test_assessment_present_observation_categories():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.PAIN))
    record.add_observation(_observation(T2, category=ObservationCategory.DIZZINESS))
    assessment = build_clinical_assessment(record)
    assert assessment.present_observation_categories == [
        ObservationCategory.PAIN.value, ObservationCategory.DIZZINESS.value,
    ]  # ObservationCategory enum tanım sırasına göre (PAIN önce tanımlı)


# ---------------------------------------------------------------------
# 3. Assessment from vital signs
# ---------------------------------------------------------------------

def test_assessment_includes_vital_sign_series():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    assessment = build_clinical_assessment(record)
    assert "temperature" in assessment.vital_sign_series
    assert assessment.vital_sign_series["temperature"].measurement_count == 1


def test_assessment_blood_pressure_split_into_systolic_and_diastolic_series():
    record = _record()
    record.add_vital_sign(_bp(T1, 120, 80))
    assessment = build_clinical_assessment(record)
    assert BLOOD_PRESSURE_SYSTOLIC_LABEL in assessment.vital_sign_series
    assert BLOOD_PRESSURE_DIASTOLIC_LABEL in assessment.vital_sign_series
    assert assessment.vital_sign_series[BLOOD_PRESSURE_SYSTOLIC_LABEL].points[0].value == 120
    assert assessment.vital_sign_series[BLOOD_PRESSURE_DIASTOLIC_LABEL].points[0].value == 80
    assert "blood_pressure" in assessment.present_vital_sign_types


# ---------------------------------------------------------------------
# 4. Assessment from laboratory results
# ---------------------------------------------------------------------

def test_assessment_includes_laboratory_result_series():
    record = _record()
    record.add_laboratory_result(_glucose(T1, 95.0))
    assessment = build_clinical_assessment(record)
    assert "glucose" in assessment.laboratory_result_series
    assert assessment.laboratory_result_series["glucose"].measurement_count == 1
    assert assessment.present_laboratory_test_names == ["glucose"]


def test_assessment_laboratory_series_sorted_alphabetically():
    record = _record()
    record.add_laboratory_result(LaboratoryResult(test_name="WBC", value=6.0, unit="10^9/L", timestamp=T1, source="onboard_lab"))
    record.add_laboratory_result(LaboratoryResult(test_name="ALT", value=20.0, unit="U/L", timestamp=T1, source="onboard_lab"))
    assessment = build_clinical_assessment(record)
    assert assessment.present_laboratory_test_names == ["ALT", "WBC"]


# ---------------------------------------------------------------------
# 5. Chronological ordering
# ---------------------------------------------------------------------

def test_vital_sign_series_points_chronologically_ordered():
    record = _record()
    record.add_vital_sign(_temp(T3, 39.0))
    record.add_vital_sign(_temp(T1, 36.5))
    record.add_vital_sign(_temp(T2, 37.5))
    assessment = build_clinical_assessment(record)
    timestamps = [p.timestamp for p in assessment.vital_sign_series["temperature"].points]
    assert timestamps == [T1, T2, T3]


def test_laboratory_series_points_chronologically_ordered():
    record = _record()
    record.add_laboratory_result(_glucose(T3, 130.0))
    record.add_laboratory_result(_glucose(T1, 85.0))
    assessment = build_clinical_assessment(record)
    timestamps = [p.timestamp for p in assessment.laboratory_result_series["glucose"].points]
    assert timestamps == [T1, T3]


# ---------------------------------------------------------------------
# 6. Repeated measurements
# ---------------------------------------------------------------------

def test_repeated_measurements_flagged():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.8))
    record.add_vital_sign(_temp(T2, 37.0))
    assessment = build_clinical_assessment(record)
    series = assessment.vital_sign_series["temperature"]
    assert series.has_repeated_measurements is True
    assert series.measurement_count == 2


def test_single_measurement_not_flagged_as_repeated():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.8))
    assessment = build_clinical_assessment(record)
    assert assessment.vital_sign_series["temperature"].has_repeated_measurements is False


# ---------------------------------------------------------------------
# 7. Missing information
# ---------------------------------------------------------------------

def test_missing_vital_sign_types_reported_when_partial_data():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    assessment = build_clinical_assessment(record)
    assert "temperature" not in assessment.missing_vital_sign_types
    assert "spo2" in assessment.missing_vital_sign_types
    assert "heart_rate" in assessment.missing_vital_sign_types


def test_missing_information_never_becomes_a_value_in_series():
    """Eksik bir vital tipi için series sözlüğünde hiçbir anahtar OLUŞTURULMAZ."""
    record = _record()
    assessment = build_clinical_assessment(record)
    assert "spo2" not in assessment.vital_sign_series


def test_reasoning_trace_missing_section_mentions_absent_categories():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    assessment = build_clinical_assessment(record)
    missing_text = " ".join(assessment.reasoning_trace.missing)
    assert "spo2" in missing_text
    assert "laboratuvar" in missing_text.lower()


# ---------------------------------------------------------------------
# 8. Conflicting data
# ---------------------------------------------------------------------

def test_conflicting_vital_sign_measurements_flagged():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.5, source="thermometer_a"))
    record.add_vital_sign(_temp(T1, 39.0, source="thermometer_b"))  # aynı timestamp, farklı değer
    assessment = build_clinical_assessment(record)
    series = assessment.vital_sign_series["temperature"]
    assert series.has_conflicting_measurements is True
    assert T1 in series.conflicting_timestamps


def test_same_timestamp_same_value_not_flagged_as_conflict():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0, source="thermometer_a"))
    record.add_vital_sign(_temp(T1, 37.0, source="thermometer_b"))
    assessment = build_clinical_assessment(record)
    series = assessment.vital_sign_series["temperature"]
    assert series.has_conflicting_measurements is False


def test_conflicting_laboratory_results_flagged():
    record = _record()
    record.add_laboratory_result(_glucose(T1, 90.0, source="lab_a"))
    record.add_laboratory_result(_glucose(T1, 140.0, source="lab_b"))
    assessment = build_clinical_assessment(record)
    series = assessment.laboratory_result_series["glucose"]
    assert series.has_conflicting_measurements is True


# ---------------------------------------------------------------------
# 9. Basic data-level trends
# ---------------------------------------------------------------------

def test_trend_increasing():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.5))
    record.add_vital_sign(_temp(T2, 37.5))
    record.add_vital_sign(_temp(T3, 38.5))
    assessment = build_clinical_assessment(record)
    assert assessment.vital_sign_series["temperature"].trend == Trend.INCREASING


def test_trend_decreasing():
    record = _record()
    record.add_vital_sign(_spo2(T1, 99.0))
    record.add_vital_sign(_spo2(T2, 96.0))
    assessment = build_clinical_assessment(record)
    assert assessment.vital_sign_series["spo2"].trend == Trend.DECREASING


def test_trend_stable():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    record.add_vital_sign(_temp(T2, 37.0))
    assessment = build_clinical_assessment(record)
    assert assessment.vital_sign_series["temperature"].trend == Trend.STABLE


def test_trend_insufficient_data_for_single_point():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    assessment = build_clinical_assessment(record)
    assert assessment.vital_sign_series["temperature"].trend == Trend.INSUFFICIENT_DATA


def test_trend_never_labeled_as_clinically_abnormal():
    """Trend değerleri sadece veri-düzeyinde etiketlerdir; klinik terim İÇERMEZ."""
    for trend in Trend:
        assert trend.value in ("increasing", "decreasing", "stable", "insufficient_data")


# ---------------------------------------------------------------------
# 10. Patient ID preservation
# ---------------------------------------------------------------------

def test_patient_id_preserved():
    record = _record(patient_id="P-999")
    assessment = build_clinical_assessment(record)
    assert assessment.patient_id == "P-999"


# ---------------------------------------------------------------------
# 11. Timestamp preservation
# ---------------------------------------------------------------------

def test_timestamps_preserved_in_series():
    record = _record()
    record.add_vital_sign(_temp(T2, 37.4))
    assessment = build_clinical_assessment(record)
    assert assessment.vital_sign_series["temperature"].points[0].timestamp == T2


def test_timestamps_preserved_in_observations():
    record = _record()
    record.add_observation(_observation(T2))
    assessment = build_clinical_assessment(record)
    assert assessment.observations[0].timestamp == T2


# ---------------------------------------------------------------------
# 12. Source preservation
# ---------------------------------------------------------------------

def test_source_preserved_in_vital_sign_series():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.1, source="onboard_medic"))
    assessment = build_clinical_assessment(record)
    assert assessment.vital_sign_series["temperature"].points[0].source == "onboard_medic"


def test_source_preserved_in_laboratory_series():
    record = _record()
    record.add_laboratory_result(_glucose(T1, 95.0, source="shore_lab"))
    assessment = build_clinical_assessment(record)
    assert assessment.laboratory_result_series["glucose"].points[0].source == "shore_lab"


# ---------------------------------------------------------------------
# 13. Deterministic output
# ---------------------------------------------------------------------

def test_same_patient_record_produces_identical_assessment_every_time():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.PAIN))
    record.add_vital_sign(_temp(T2, 38.0))
    record.add_vital_sign(_temp(T1, 37.0))
    record.add_laboratory_result(_glucose(T1, 95.0))

    a1 = build_clinical_assessment(record)
    a2 = build_clinical_assessment(record)

    assert a1.patient_id == a2.patient_id
    assert a1.present_observation_categories == a2.present_observation_categories
    assert a1.present_vital_sign_types == a2.present_vital_sign_types
    assert a1.missing_vital_sign_types == a2.missing_vital_sign_types
    assert a1.present_laboratory_test_names == a2.present_laboratory_test_names
    assert [p.value for p in a1.vital_sign_series["temperature"].points] == \
           [p.value for p in a2.vital_sign_series["temperature"].points]
    assert a1.vital_sign_series["temperature"].trend == a2.vital_sign_series["temperature"].trend
    assert a1.reasoning_trace.observed == a2.reasoning_trace.observed
    assert a1.reasoning_trace.data_interpretation == a2.reasoning_trace.data_interpretation
    assert a1.reasoning_trace.missing == a2.reasoning_trace.missing
    assert a1.reasoning_trace.unsupported == a2.reasoning_trace.unsupported


# ---------------------------------------------------------------------
# 14. Explicit absence of diagnosis / probability / medication / dosage /
#     treatment duration / clinical thresholds
# ---------------------------------------------------------------------

def test_clinical_assessment_has_no_diagnosis_or_medication_attributes():
    record = _record()
    record.add_vital_sign(_temp(T1, 39.5))  # bilerek "yüksek görünen" bir değer -- yine de yorumlanmamalı
    assessment = build_clinical_assessment(record)
    for forbidden in (
        "diagnosis", "differential_diagnosis", "disease_probability", "probability",
        "medication_recommendation", "dosage", "treatment_duration",
        "clinical_threshold", "reference_range", "contraindication", "drug_interaction",
        "emergency_level", "severity_score",
    ):
        assert not hasattr(assessment, forbidden)


def test_measurement_series_has_no_clinical_interpretation_attributes():
    series = MeasurementSeries(label="temperature", points=[MeasurementPoint(value=39.5, timestamp=T1, source="x")])
    for forbidden in ("is_abnormal", "clinical_interpretation", "severity", "diagnosis"):
        assert not hasattr(series, forbidden)


def test_high_value_does_not_change_trend_semantics():
    """
    Yüksek bir sayısal değer (örn. 39.5°C), trend hesaplamasını hiçbir
    şekilde 'abnormal'/'fever' gibi bir etikete DÖNÜŞTÜRMEZ -- sadece
    increasing/decreasing/stable/insufficient_data döner.
    """
    record = _record()
    record.add_vital_sign(_temp(T1, 39.5))
    assessment = build_clinical_assessment(record)
    assert assessment.vital_sign_series["temperature"].trend == Trend.INSUFFICIENT_DATA


# ---------------------------------------------------------------------
# 15. Structured reasoning trace
# ---------------------------------------------------------------------

def test_reasoning_trace_structure_present():
    assessment = build_clinical_assessment(_record())
    assert isinstance(assessment.reasoning_trace, ReasoningTrace)
    assert isinstance(assessment.reasoning_trace.observed, list)
    assert isinstance(assessment.reasoning_trace.data_interpretation, list)
    assert isinstance(assessment.reasoning_trace.missing, list)
    assert isinstance(assessment.reasoning_trace.unsupported, list)


def test_reasoning_trace_unsupported_always_present_even_with_no_data():
    """
    'unsupported' uyarıları veri olsun olmasın HER ZAMAN mevcut olmalı --
    böylece boş bir kayıt bile yanlışlıkla 'sorun yok' gibi okunamaz.
    """
    assessment = build_clinical_assessment(_record())
    assert len(assessment.reasoning_trace.unsupported) > 0
    joined = " ".join(assessment.reasoning_trace.unsupported).lower()
    assert "tanı" in joined
    assert "ilaç" in joined


def test_reasoning_trace_observed_reflects_actual_data():
    record = _record()
    record.add_observation(_observation(T1))
    record.add_vital_sign(_temp(T1, 37.0))
    assessment = build_clinical_assessment(record)
    joined = " ".join(assessment.reasoning_trace.observed)
    assert "1 gözlem" in joined
    assert "temperature" in joined


def test_reasoning_trace_data_interpretation_mentions_trend_only_with_enough_data():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.5))
    record.add_vital_sign(_temp(T2, 38.0))
    assessment = build_clinical_assessment(record)
    joined = " ".join(assessment.reasoning_trace.data_interpretation)
    assert "increasing" in joined
    assert "klinik anlam ifade etmez" in joined


# ---------------------------------------------------------------------
# 16. Compatibility with Phase 2 routing
# ---------------------------------------------------------------------

def test_phase2_domain_routing_unaffected_by_phase5():
    """
    Faz 5, routing/domain_router.py'yi hiç değiştirmedi -- burada sadece
    hızlı bir sağlık kontrolü (tam regresyon paketi tests/test_domain_router.py
    ve tests/test_pipeline_domain_routing.py'de zaten mevcut).
    """
    from routing.domain_router import Domain, classify_domain

    assert classify_domain("hastanın nabzı kaç?") == Domain.MEDICAL
    assert classify_domain("gemi limana ne zaman varacak?") == Domain.MARITIME


# ---------------------------------------------------------------------
# Combined / integration case
# ---------------------------------------------------------------------

def test_full_assessment_with_observations_vitals_and_labs():
    record = _record(patient_id="P-777")
    record.add_observation(_observation(T1, category=ObservationCategory.BREATHING_DIFFICULTY))
    record.add_vital_sign(_temp(T1, 36.8))
    record.add_vital_sign(_temp(T2, 38.2))
    record.add_vital_sign(_bp(T1, 118, 76))
    record.add_laboratory_result(_glucose(T1, 92.0))

    assessment = build_clinical_assessment(record)

    assert assessment.patient_id == "P-777"
    assert len(assessment.observations) == 1
    assert assessment.vital_sign_series["temperature"].trend == Trend.INCREASING
    assert BLOOD_PRESSURE_SYSTOLIC_LABEL in assessment.vital_sign_series
    assert "glucose" in assessment.laboratory_result_series
    assert "heart_rate" in assessment.missing_vital_sign_types
    assert "spo2" in assessment.missing_vital_sign_types
