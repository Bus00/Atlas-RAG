"""
tests/test_clinical_reasoning.py
--------------------------------------
V2 Faz 6: medical/clinical_reasoning.py (ClinicalReasoning) testleri.

Bu testler yalnızca YAPISAL veri ilişkilerini (kronoloji, tekrar, veri
düzeyinde eğilim, çelişki, eksiklik, zamansal eş-oluşum, hasta bağlantısı)
ve "observed / interpretation / missing / unsupported" ayrımını kontrol
eder -- hiçbir klinik çıkarım/tanı/nedensellik test edilmez, çünkü katmanın
kendisi bunları üretmez (bkz. medical/clinical_reasoning.py docstring'i).
"""
from __future__ import annotations

from datetime import datetime

import pytest

from medical.clinical_assessment import build_clinical_assessment
from medical.clinical_reasoning import (
    ClinicalReasoning,
    ReasoningCategory,
    ReasoningEvidence,
    ReasoningItem,
    ReasoningItemKind,
    build_clinical_reasoning,
    build_clinical_reasoning_from_record,
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


def _patient(patient_id: str = "P-300") -> Patient:
    return Patient(patient_id=patient_id)


def _record(patient_id: str = "P-300") -> PatientRecord:
    return PatientRecord(patient=_patient(patient_id))


def _observation(ts: datetime, category: ObservationCategory = ObservationCategory.SYMPTOM,
                  description: str = "hasta ateş bildiriyor", source: str = "patient_report") -> Observation:
    return Observation(category=category, description=description, timestamp=ts, source=source)


def _temp(ts: datetime, value: float, source: str = "thermometer") -> VitalSign:
    return VitalSign(vital_type=VitalSignType.TEMPERATURE, timestamp=ts, source=source, unit="°C", value=value)


def _spo2(ts: datetime, value: float, source: str = "pulse_oximeter") -> VitalSign:
    return VitalSign(vital_type=VitalSignType.SPO2, timestamp=ts, source=source, unit="%", value=value)


def _glucose(ts: datetime, value: float, source: str = "onboard_lab") -> LaboratoryResult:
    return LaboratoryResult(test_name="glucose", value=value, unit="mg/dL", timestamp=ts, source=source)


def _reasoning_for(record: PatientRecord) -> ClinicalReasoning:
    return build_clinical_reasoning(build_clinical_assessment(record))


# ---------------------------------------------------------------------
# 1. Empty ClinicalAssessment
# ---------------------------------------------------------------------

def test_empty_assessment_produces_reasoning_with_patient_id():
    reasoning = _reasoning_for(_record())
    assert reasoning.patient_id == "P-300"
    assert len(reasoning.items) > 0  # missing-info + scope-limitation items still present


def test_empty_assessment_reports_no_structural_data():
    reasoning = _reasoning_for(_record())
    missing_statements = " ".join(item.statement for item in reasoning.missing)
    assert "hiçbir yapısal veri" in missing_statements


def test_build_clinical_reasoning_rejects_non_assessment():
    with pytest.raises(MedicalDataValidationError):
        build_clinical_reasoning("not an assessment")  # type: ignore[arg-type]


def test_build_clinical_reasoning_from_record_chains_correctly():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    reasoning = build_clinical_reasoning_from_record(record)
    assert reasoning.patient_id == "P-300"
    assert any("temperature" in item.statement for item in reasoning.observed)


# ---------------------------------------------------------------------
# 2. Assessment containing observations
# ---------------------------------------------------------------------

def test_reasoning_includes_observation_facts():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.PAIN))
    reasoning = _reasoning_for(record)
    obs_items = [i for i in reasoning.observed if i.category == ReasoningCategory.DATA_AVAILABILITY
                 and "pain" in i.statement]
    assert len(obs_items) == 1
    assert obs_items[0].evidence[0].kind == "observation"
    assert obs_items[0].evidence[0].label == "pain"


# ---------------------------------------------------------------------
# 3. Assessment containing vital measurements
# ---------------------------------------------------------------------

def test_reasoning_includes_vital_chronology_fact():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    reasoning = _reasoning_for(record)
    chron_items = [i for i in reasoning.observed if i.category == ReasoningCategory.CHRONOLOGY]
    assert any("temperature" in i.statement for i in chron_items)


# ---------------------------------------------------------------------
# 4. Assessment containing laboratory results
# ---------------------------------------------------------------------

def test_reasoning_includes_laboratory_chronology_fact():
    record = _record()
    record.add_laboratory_result(_glucose(T1, 95.0))
    reasoning = _reasoning_for(record)
    chron_items = [i for i in reasoning.observed if i.category == ReasoningCategory.CHRONOLOGY]
    assert any("glucose" in i.statement for i in chron_items)


# ---------------------------------------------------------------------
# 5. Repeated measurements
# ---------------------------------------------------------------------

def test_repeated_measurement_item_present_for_multiple_entries():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.8))
    record.add_vital_sign(_temp(T2, 37.2))
    reasoning = _reasoning_for(record)
    repeated = [i for i in reasoning.items if i.category == ReasoningCategory.REPEATED_MEASUREMENT]
    assert len(repeated) == 1
    assert "2" in repeated[0].statement


def test_no_repeated_measurement_item_for_single_entry():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.8))
    reasoning = _reasoning_for(record)
    repeated = [i for i in reasoning.items if i.category == ReasoningCategory.REPEATED_MEASUREMENT]
    assert repeated == []


# ---------------------------------------------------------------------
# 6/7/8. Increasing / decreasing / stable data-level trend
# ---------------------------------------------------------------------

def test_increasing_trend_item():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.0))
    reasoning = _reasoning_for(record)
    trend_items = [i for i in reasoning.data_interpretation if i.category == ReasoningCategory.DATA_LEVEL_TREND]
    assert any("increasing" in i.statement for i in trend_items)


def test_decreasing_trend_item():
    record = _record()
    record.add_vital_sign(_spo2(T1, 99.0))
    record.add_vital_sign(_spo2(T2, 95.0))
    reasoning = _reasoning_for(record)
    trend_items = [i for i in reasoning.data_interpretation if i.category == ReasoningCategory.DATA_LEVEL_TREND]
    assert any("decreasing" in i.statement for i in trend_items)


def test_stable_trend_item():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    record.add_vital_sign(_temp(T2, 37.0))
    reasoning = _reasoning_for(record)
    trend_items = [i for i in reasoning.data_interpretation if i.category == ReasoningCategory.DATA_LEVEL_TREND]
    assert any("stable" in i.statement for i in trend_items)


# ---------------------------------------------------------------------
# 9. Insufficient data
# ---------------------------------------------------------------------

def test_insufficient_data_trend_becomes_missing_information():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    reasoning = _reasoning_for(record)
    trend_missing = [i for i in reasoning.missing if i.category == ReasoningCategory.DATA_LEVEL_TREND]
    assert len(trend_missing) == 1
    assert "temperature" in trend_missing[0].statement


# ---------------------------------------------------------------------
# 10. Conflicting same-timestamp measurements
# ---------------------------------------------------------------------

def test_conflicting_measurements_produce_conflict_item_with_both_sources():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.5, source="thermometer_a"))
    record.add_vital_sign(_temp(T1, 39.0, source="thermometer_b"))
    reasoning = _reasoning_for(record)
    conflicts = [i for i in reasoning.data_interpretation if i.category == ReasoningCategory.MEASUREMENT_CONFLICT]
    assert len(conflicts) == 1
    sources = {e.source for e in conflicts[0].evidence}
    assert sources == {"thermometer_a", "thermometer_b"}


def test_no_conflict_item_when_values_agree():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0, source="a"))
    record.add_vital_sign(_temp(T1, 37.0, source="b"))
    reasoning = _reasoning_for(record)
    conflicts = [i for i in reasoning.items if i.category == ReasoningCategory.MEASUREMENT_CONFLICT]
    assert conflicts == []


def test_conflict_does_not_choose_a_winner():
    """Çelişki maddesi, hangi ölçümün 'doğru' olduğuna dair hiçbir alan İÇERMEZ."""
    record = _record()
    record.add_vital_sign(_temp(T1, 36.5, source="a"))
    record.add_vital_sign(_temp(T1, 39.0, source="b"))
    reasoning = _reasoning_for(record)
    conflict = [i for i in reasoning.items if i.category == ReasoningCategory.MEASUREMENT_CONFLICT][0]
    for forbidden in ("winner", "correct_value", "chosen_value", "resolved_value"):
        assert not hasattr(conflict, forbidden)
    assert len(conflict.evidence) == 2  # her iki ölçüm de korunuyor


# ---------------------------------------------------------------------
# 11. Source preservation
# ---------------------------------------------------------------------

def test_source_preserved_in_evidence():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.1, source="onboard_medic"))
    reasoning = _reasoning_for(record)
    chron_item = [i for i in reasoning.observed if i.category == ReasoningCategory.CHRONOLOGY][0]
    assert chron_item.evidence[0].source == "onboard_medic"


# ---------------------------------------------------------------------
# 12. Timestamp preservation
# ---------------------------------------------------------------------

def test_timestamp_preserved_in_evidence():
    record = _record()
    record.add_vital_sign(_temp(T2, 37.4))
    reasoning = _reasoning_for(record)
    chron_item = [i for i in reasoning.observed if i.category == ReasoningCategory.CHRONOLOGY][0]
    assert chron_item.evidence[0].timestamp == T2


# ---------------------------------------------------------------------
# 13. Patient ID preservation
# ---------------------------------------------------------------------

def test_patient_id_preserved_on_reasoning_and_every_item():
    record = _record(patient_id="P-555")
    record.add_vital_sign(_temp(T1, 37.0))
    record.add_observation(_observation(T1))
    reasoning = _reasoning_for(record)
    assert reasoning.patient_id == "P-555"
    assert all(item.patient_id == "P-555" for item in reasoning.items)


# ---------------------------------------------------------------------
# 14. Missing information
# ---------------------------------------------------------------------

def test_missing_vital_types_produce_missing_items():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    reasoning = _reasoning_for(record)
    missing_labels = {
        i.statement for i in reasoning.missing if i.category == ReasoningCategory.DATA_AVAILABILITY
    }
    assert any("spo2" in s for s in missing_labels)
    assert any("heart_rate" in s for s in missing_labels)
    assert not any("'temperature'" in s for s in missing_labels)


def test_missing_never_labeled_as_negative_finding():
    record = _record()
    reasoning = _reasoning_for(record)
    joined = " ".join(i.statement for i in reasoning.missing).lower()
    assert "negatif" not in joined
    assert "normal" not in joined


# ---------------------------------------------------------------------
# 15. Deterministic output
# ---------------------------------------------------------------------

def test_same_assessment_produces_identical_reasoning_every_time():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.PAIN))
    record.add_vital_sign(_temp(T2, 38.0))
    record.add_vital_sign(_temp(T1, 37.0))
    record.add_laboratory_result(_glucose(T1, 95.0))
    assessment = build_clinical_assessment(record)

    r1 = build_clinical_reasoning(assessment)
    r2 = build_clinical_reasoning(assessment)

    assert [i.statement for i in r1.items] == [i.statement for i in r2.items]
    assert [i.category for i in r1.items] == [i.category for i in r2.items]
    assert [i.kind for i in r1.items] == [i.kind for i in r2.items]


def test_reasoning_deterministic_across_hash_seeds_smoke():
    """
    Aynı süreç içinde, aynı girdi için tekrar tekrar üretim yapıldığında
    dict/set sırasına bağlı bir kaymanın olmadığını doğrular (tam
    hash-seed testi ayrı bir script ile de doğrulanmıştır, bkz. final rapor).
    """
    record = _record()
    record.add_vital_sign(_temp(T1, 36.5))
    record.add_vital_sign(_temp(T2, 38.0))
    record.add_laboratory_result(_glucose(T1, 90.0))
    assessment = build_clinical_assessment(record)
    results = [tuple(i.statement for i in build_clinical_reasoning(assessment).items) for _ in range(5)]
    assert len(set(results)) == 1


# ---------------------------------------------------------------------
# 16. Stable ordering
# ---------------------------------------------------------------------

def test_item_ordering_is_stable_and_matches_canonical_vital_order():
    record = _record()
    record.add_vital_sign(_spo2(T1, 98.0))          # eklenme sırası: spo2 önce
    record.add_vital_sign(_temp(T1, 37.0))          # sonra temperature
    reasoning = _reasoning_for(record)
    chron_labels_in_order = [
        i.evidence[0].label for i in reasoning.observed if i.category == ReasoningCategory.CHRONOLOGY
    ]
    # VitalSignType kanonik tanım sırası: TEMPERATURE önce SPO2'den -- ekleme
    # sırasından BAĞIMSIZ olarak temperature önce gelmeli.
    assert chron_labels_in_order.index("temperature") < chron_labels_in_order.index("spo2")


# ---------------------------------------------------------------------
# 17. Multiple data types together
# ---------------------------------------------------------------------

def test_full_reasoning_with_observations_vitals_and_labs():
    record = _record(patient_id="P-888")
    record.add_observation(_observation(T1, category=ObservationCategory.BREATHING_DIFFICULTY))
    record.add_vital_sign(_temp(T1, 36.8))
    record.add_vital_sign(_temp(T2, 38.2))
    record.add_laboratory_result(_glucose(T1, 92.0))

    reasoning = _reasoning_for(record)

    assert reasoning.patient_id == "P-888"
    assert any(i.category == ReasoningCategory.PATIENT_LINKAGE and i.kind == ReasoningItemKind.OBSERVED_FACT
               for i in reasoning.items)
    assert any(i.category == ReasoningCategory.CHRONOLOGY and "temperature" in i.statement for i in reasoning.items)
    assert any(i.category == ReasoningCategory.CHRONOLOGY and "glucose" in i.statement for i in reasoning.items)
    assert any(i.category == ReasoningCategory.DATA_LEVEL_TREND and "increasing" in i.statement for i in reasoning.items)


# ---------------------------------------------------------------------
# 18. Observation/measurement temporal relationships
# ---------------------------------------------------------------------

def test_temporal_cooccurrence_between_observation_and_vital():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.DIZZINESS))
    record.add_vital_sign(_temp(T1, 38.5))  # aynı zaman damgası
    reasoning = _reasoning_for(record)
    cooc = [i for i in reasoning.observed if i.category == ReasoningCategory.TEMPORAL_COOCCURRENCE]
    assert len(cooc) == 1
    assert cooc[0].evidence[0].kind == "observation"
    assert cooc[0].evidence[1].kind == "vital_sign_measurement"


def test_no_cooccurrence_item_when_timestamps_differ():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.DIZZINESS))
    record.add_vital_sign(_temp(T2, 38.5))  # farklı zaman damgası
    reasoning = _reasoning_for(record)
    cooc = [i for i in reasoning.items if i.category == ReasoningCategory.TEMPORAL_COOCCURRENCE]
    assert cooc == []


def test_temporal_cooccurrence_with_laboratory_result():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.WEAKNESS))
    record.add_laboratory_result(_glucose(T1, 60.0))
    reasoning = _reasoning_for(record)
    cooc = [i for i in reasoning.observed if i.category == ReasoningCategory.TEMPORAL_COOCCURRENCE]
    assert len(cooc) == 1
    assert cooc[0].evidence[1].kind == "laboratory_measurement"


# ---------------------------------------------------------------------
# 19. No causal inference
# ---------------------------------------------------------------------

def test_cooccurrence_statement_explicitly_disclaims_causation():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.DIZZINESS))
    record.add_vital_sign(_temp(T1, 38.5))
    reasoning = _reasoning_for(record)
    cooc = [i for i in reasoning.observed if i.category == ReasoningCategory.TEMPORAL_COOCCURRENCE][0]
    assert "nedensellik iddia edilmez" in cooc.statement


def test_no_causal_language_anywhere_in_statements():
    record = _record()
    record.add_observation(_observation(T1))
    record.add_vital_sign(_temp(T1, 39.0))
    record.add_vital_sign(_temp(T2, 40.0))
    reasoning = _reasoning_for(record)
    forbidden_terms = ["neden oldu", "sebep oldu", "yol açtı", "caused by", "due to", "leads to"]
    for item in reasoning.items:
        lowered = item.statement.lower()
        for term in forbidden_terms:
            assert term not in lowered


def test_scope_limitation_includes_correlation_vs_causation_disclaimer():
    reasoning = _reasoning_for(_record())
    joined = " ".join(i.statement for i in reasoning.unsupported)
    assert "nedensellik" in joined


# ---------------------------------------------------------------------
# 20. No diagnosis fields
# ---------------------------------------------------------------------

def test_no_diagnosis_attributes_on_reasoning_objects():
    record = _record()
    record.add_vital_sign(_temp(T1, 39.5))
    reasoning = _reasoning_for(record)
    for obj in [reasoning] + reasoning.items:
        for forbidden in ("diagnosis", "differential_diagnosis", "disease", "condition_name"):
            assert not hasattr(obj, forbidden)


# ---------------------------------------------------------------------
# 21. No probability/risk fields
# ---------------------------------------------------------------------

def test_no_probability_or_risk_attributes():
    reasoning = _reasoning_for(_record())
    for obj in [reasoning] + reasoning.items:
        for forbidden in ("probability", "risk_score", "risk_percentage", "likelihood", "severity_score"):
            assert not hasattr(obj, forbidden)


# ---------------------------------------------------------------------
# 22. No medication fields
# ---------------------------------------------------------------------

def test_no_medication_attributes():
    reasoning = _reasoning_for(_record())
    for obj in [reasoning] + reasoning.items:
        for forbidden in ("medication", "medication_recommendation", "drug", "prescription"):
            assert not hasattr(obj, forbidden)


# ---------------------------------------------------------------------
# 23. No dosage/treatment fields
# ---------------------------------------------------------------------

def test_no_dosage_or_treatment_attributes():
    reasoning = _reasoning_for(_record())
    for obj in [reasoning] + reasoning.items:
        for forbidden in ("dosage", "dose", "treatment_duration", "treatment_plan"):
            assert not hasattr(obj, forbidden)


# ---------------------------------------------------------------------
# 24. No clinical threshold logic
# ---------------------------------------------------------------------

def test_no_clinical_threshold_attributes_or_hardcoded_reference_values():
    reasoning = _reasoning_for(_record())
    for obj in [reasoning] + reasoning.items:
        for forbidden in ("clinical_threshold", "reference_range", "normal_range", "abnormal", "is_abnormal"):
            assert not hasattr(obj, forbidden)


def test_high_or_low_values_never_classified_by_trend_alone():
    """
    Yüksek/düşük bir değer (örn. 39.5°C ya da SpO2 70), trend hesaplamasını
    hiçbir şekilde bir klinik etikete (fever/hipoksi vb.) DÖNÜŞTÜRMEZ.
    """
    record = _record()
    record.add_vital_sign(_spo2(T1, 99.0))
    record.add_vital_sign(_spo2(T2, 70.0))  # bilerek düşük görünen bir değer
    reasoning = _reasoning_for(record)
    trend_items = [i for i in reasoning.data_interpretation if i.category == ReasoningCategory.DATA_LEVEL_TREND]
    joined = " ".join(i.statement for i in trend_items).lower()
    for forbidden_clinical_word in ("hipoksi", "kritik", "tehlikeli", "anormal", "abnormal"):
        assert forbidden_clinical_word not in joined
    assert any("decreasing" in i.statement for i in trend_items)


# ---------------------------------------------------------------------
# 25. Existing Phase 2 routing compatibility
# ---------------------------------------------------------------------

def test_phase2_domain_routing_unaffected_by_phase6():
    """
    Faz 6, routing/domain_router.py'yi hiç değiştirmedi -- burada sadece
    hızlı bir sağlık kontrolü (tam regresyon paketi tests/test_domain_router.py
    ve tests/test_pipeline_domain_routing.py'de zaten mevcut).
    """
    from routing.domain_router import Domain, classify_domain

    assert classify_domain("hastanın tansiyonu kaç?") == Domain.MEDICAL
    assert classify_domain("gemi hangi liman için yola çıktı?") == Domain.MARITIME


# ---------------------------------------------------------------------
# Reasoning item / evidence structural sanity
# ---------------------------------------------------------------------

def test_reasoning_item_and_evidence_are_plain_dataclasses_with_expected_fields():
    evidence = ReasoningEvidence(kind="observation", label="pain", timestamp=T1, source="patient_report", value=None)
    item = ReasoningItem(
        category=ReasoningCategory.DATA_AVAILABILITY,
        kind=ReasoningItemKind.OBSERVED_FACT,
        statement="test",
        patient_id="P-1",
        evidence=[evidence],
    )
    assert item.evidence[0] is evidence
    assert item.category == ReasoningCategory.DATA_AVAILABILITY
    assert item.kind == ReasoningItemKind.OBSERVED_FACT
