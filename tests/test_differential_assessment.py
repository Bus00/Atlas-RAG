"""
tests/test_differential_assessment.py
-------------------------------------------
V2 Faz 7: medical/differential_assessment.py (DifferentialAssessment) testleri.

Bu testler yalnızca YAPISAL patern işaretlemeyi (trend, zamansal
eş-oluşum), destekleyici/çelişen kanıtı, eksik bilgiyi, belirsizlik
seviyesini ve izlenebilirliği kontrol eder -- hiçbir hastalık adı, tanı,
olasılık test edilmez, çünkü bu katman bunları üretmez (bkz.
medical/differential_assessment.py docstring'i).
"""
from __future__ import annotations

from datetime import datetime

import pytest

from medical.clinical_assessment import build_clinical_assessment
from medical.clinical_reasoning import ReasoningCategory, build_clinical_reasoning
from medical.differential_assessment import (
    DifferentialAssessment,
    PossibleClinicalConcern,
    UncertaintyLevel,
    build_differential_assessment,
    build_differential_assessment_from_record,
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


def _patient(patient_id: str = "P-400") -> Patient:
    return Patient(patient_id=patient_id)


def _record(patient_id: str = "P-400") -> PatientRecord:
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


def _differential_for(record: PatientRecord) -> DifferentialAssessment:
    return build_differential_assessment_from_record(record)


def _concern(differential: DifferentialAssessment, concern_id: str) -> PossibleClinicalConcern:
    matches = [c for c in differential.possible_clinical_concerns if c.concern_id == concern_id]
    assert len(matches) == 1, f"expected exactly one concern with id {concern_id}, found {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------
# Empty / minimal input
# ---------------------------------------------------------------------

def test_empty_record_produces_no_concerns_but_has_scope_limitations():
    differential = _differential_for(_record())
    assert differential.patient_id == "P-400"
    assert differential.possible_clinical_concerns == []
    assert len(differential.scope_limitations) > 0


def test_no_concerns_never_implies_patient_is_fine():
    differential = _differential_for(_record())
    joined = " ".join(differential.scope_limitations).lower()
    assert "sorunsuz" in joined  # açıkça reddeden ifade mevcut


def test_build_differential_assessment_rejects_wrong_types():
    record = _record()
    assessment = build_clinical_assessment(record)
    reasoning = build_clinical_reasoning(assessment)
    with pytest.raises(MedicalDataValidationError):
        build_differential_assessment("not reasoning", assessment)  # type: ignore[arg-type]
    with pytest.raises(MedicalDataValidationError):
        build_differential_assessment(reasoning, "not assessment")  # type: ignore[arg-type]


def test_build_differential_assessment_rejects_mismatched_patient_ids():
    record_a = _record(patient_id="P-A")
    record_b = _record(patient_id="P-B")
    assessment_a = build_clinical_assessment(record_a)
    reasoning_b = build_clinical_reasoning(build_clinical_assessment(record_b))
    with pytest.raises(MedicalDataValidationError):
        build_differential_assessment(reasoning_b, assessment_a)


# ---------------------------------------------------------------------
# Stable / insufficient-data trends never produce a concern
# ---------------------------------------------------------------------

def test_stable_trend_produces_no_concern():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    record.add_vital_sign(_temp(T2, 37.0))
    differential = _differential_for(record)
    assert differential.possible_clinical_concerns == []


def test_single_measurement_insufficient_data_produces_no_concern():
    record = _record()
    record.add_vital_sign(_temp(T1, 37.0))
    differential = _differential_for(record)
    assert differential.possible_clinical_concerns == []


# ---------------------------------------------------------------------
# Increasing / decreasing trend concerns
# ---------------------------------------------------------------------

def test_increasing_vital_trend_produces_concern():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.5))
    differential = _differential_for(record)
    concern = _concern(differential, "trend::temperature")
    assert "increasing" in concern.description
    assert concern.uncertainty == UncertaintyLevel.DATA_LEVEL_ONLY


def test_decreasing_vital_trend_produces_concern():
    record = _record()
    record.add_vital_sign(_spo2(T1, 99.0))
    record.add_vital_sign(_spo2(T2, 94.0))
    differential = _differential_for(record)
    concern = _concern(differential, "trend::spo2")
    assert "decreasing" in concern.description


def test_increasing_laboratory_trend_produces_concern():
    record = _record()
    record.add_laboratory_result(_glucose(T1, 90.0))
    record.add_laboratory_result(_glucose(T2, 140.0))
    differential = _differential_for(record)
    concern = _concern(differential, "trend::glucose")
    assert "increasing" in concern.description


# ---------------------------------------------------------------------
# Supporting evidence
# ---------------------------------------------------------------------

def test_trend_concern_supporting_evidence_includes_chronology_and_trend_items():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.5))
    differential = _differential_for(record)
    concern = _concern(differential, "trend::temperature")
    categories = {item.category for item in concern.supporting_evidence}
    assert ReasoningCategory.CHRONOLOGY in categories
    assert ReasoningCategory.DATA_LEVEL_TREND in categories


def test_cooccurrence_concern_supporting_evidence_references_both_facts():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.DIZZINESS))
    record.add_vital_sign(_temp(T1, 38.5))
    differential = _differential_for(record)
    matches = [c for c in differential.possible_clinical_concerns if c.concern_id.startswith("cooccurrence::")]
    assert len(matches) == 1
    concern = matches[0]
    assert len(concern.supporting_evidence) == 1
    assert concern.supporting_evidence[0].category == ReasoningCategory.TEMPORAL_COOCCURRENCE
    assert len(concern.supporting_evidence[0].evidence) == 2


# ---------------------------------------------------------------------
# Contradicting evidence
# ---------------------------------------------------------------------

def test_conflicting_measurements_produce_contradicting_evidence_and_raise_uncertainty():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.5))
    record.add_vital_sign(_temp(T2, 30.0, source="faulty_sensor"))  # aynı T2'de çelişkili değer
    differential = _differential_for(record)
    concern = _concern(differential, "trend::temperature")
    assert len(concern.contradicting_evidence) == 1
    assert concern.contradicting_evidence[0].category == ReasoningCategory.MEASUREMENT_CONFLICT
    assert concern.uncertainty == UncertaintyLevel.CONFLICTING_DATA


def test_no_contradicting_evidence_when_no_conflict_exists():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.5))
    differential = _differential_for(record)
    concern = _concern(differential, "trend::temperature")
    assert concern.contradicting_evidence == []
    assert concern.uncertainty == UncertaintyLevel.DATA_LEVEL_ONLY


def test_contradicting_evidence_does_not_choose_a_winner():
    """Çelişki kanıtı, hangi ölçümün 'doğru' olduğuna dair hiçbir seçim İÇERMEZ."""
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.5))
    record.add_vital_sign(_temp(T2, 30.0, source="faulty_sensor"))
    differential = _differential_for(record)
    concern = _concern(differential, "trend::temperature")
    conflict_item = concern.contradicting_evidence[0]
    sources = {e.source for e in conflict_item.evidence}
    assert sources == {"thermometer", "faulty_sensor"}
    for forbidden in ("winner", "resolved_value", "correct_source"):
        assert not hasattr(conflict_item, forbidden)


# ---------------------------------------------------------------------
# Missing information
# ---------------------------------------------------------------------

def test_every_concern_carries_missing_clinical_rule_base_notice():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.5))
    differential = _differential_for(record)
    concern = _concern(differential, "trend::temperature")
    assert len(concern.missing_information) == 1
    missing_item = concern.missing_information[0]
    assert "klinik kural tabanı" in missing_item.statement
    assert missing_item.kind.value == "missing_information"


def test_missing_information_never_phrased_as_negative_finding():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.5))
    differential = _differential_for(record)
    concern = _concern(differential, "trend::temperature")
    joined = concern.missing_information[0].statement.lower()
    assert "negatif" not in joined
    assert "normal" not in joined


# ---------------------------------------------------------------------
# Uncertainty preservation
# ---------------------------------------------------------------------

def test_uncertainty_never_exceeds_data_level_only_without_conflict():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.5))
    record.add_laboratory_result(_glucose(T1, 85.0))
    record.add_laboratory_result(_glucose(T2, 130.0))
    differential = _differential_for(record)
    for concern in differential.possible_clinical_concerns:
        assert concern.uncertainty in (UncertaintyLevel.DATA_LEVEL_ONLY, UncertaintyLevel.CONFLICTING_DATA)


def test_uncertainty_enum_has_no_clinical_confidence_language():
    """UncertaintyLevel değerleri asla 'likely'/'probable'/'confirmed' gibi klinik güven terimleri İÇERMEZ."""
    forbidden = {"likely", "probable", "confirmed", "definite", "certain", "high", "low", "medium"}
    for level in UncertaintyLevel:
        assert level.value not in forbidden


# ---------------------------------------------------------------------
# Provenance preservation
# ---------------------------------------------------------------------

def test_provenance_patient_id_preserved_on_assessment_and_every_concern():
    record = _record(patient_id="P-777")
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.5))
    differential = _differential_for(record)
    assert differential.patient_id == "P-777"
    for concern in differential.possible_clinical_concerns:
        assert concern.patient_id == "P-777"
        for item in concern.supporting_evidence + concern.contradicting_evidence + concern.missing_information:
            assert item.patient_id == "P-777"


def test_provenance_timestamp_and_source_preserved_in_supporting_evidence():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0, source="onboard_medic"))
    record.add_vital_sign(_temp(T2, 38.5, source="onboard_medic"))
    differential = _differential_for(record)
    concern = _concern(differential, "trend::temperature")
    chronology_item = [i for i in concern.supporting_evidence if i.category == ReasoningCategory.CHRONOLOGY][0]
    timestamps = {e.timestamp for e in chronology_item.evidence}
    sources = {e.source for e in chronology_item.evidence}
    assert timestamps == {T1, T2}
    assert sources == {"onboard_medic"}


# ---------------------------------------------------------------------
# Deterministic output / stable ordering
# ---------------------------------------------------------------------

def test_same_input_produces_identical_differential_assessment():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.DIZZINESS))
    record.add_vital_sign(_temp(T2, 38.0))
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_laboratory_result(_glucose(T1, 95.0))
    record.add_laboratory_result(_glucose(T2, 140.0))
    assessment = build_clinical_assessment(record)
    reasoning = build_clinical_reasoning(assessment)

    d1 = build_differential_assessment(reasoning, assessment)
    d2 = build_differential_assessment(reasoning, assessment)

    assert [c.concern_id for c in d1.possible_clinical_concerns] == [c.concern_id for c in d2.possible_clinical_concerns]
    assert [c.description for c in d1.possible_clinical_concerns] == [c.description for c in d2.possible_clinical_concerns]
    assert [c.uncertainty for c in d1.possible_clinical_concerns] == [c.uncertainty for c in d2.possible_clinical_concerns]
    assert d1.scope_limitations == d2.scope_limitations


def test_concern_ordering_is_stable_vitals_then_labs_then_cooccurrence():
    record = _record()
    record.add_laboratory_result(_glucose(T1, 85.0))
    record.add_laboratory_result(_glucose(T2, 140.0))
    record.add_vital_sign(_spo2(T1, 99.0))
    record.add_vital_sign(_spo2(T2, 94.0))
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.5))
    record.add_observation(_observation(T1, category=ObservationCategory.DIZZINESS))

    differential = _differential_for(record)
    ids = [c.concern_id for c in differential.possible_clinical_concerns]
    # VitalSignType kanonik sırasına göre temperature spo2'den önce gelmeli;
    # lab (glucose) alfabetik olarak vital'lardan sonra; cooccurrence en sonda.
    assert ids.index("trend::temperature") < ids.index("trend::spo2")
    assert ids.index("trend::spo2") < ids.index("trend::glucose")
    assert ids.index("trend::glucose") < ids.index("cooccurrence::dizziness::temperature::2026-01-01T08:00:00")


# ---------------------------------------------------------------------
# Multiple data types together
# ---------------------------------------------------------------------

def test_full_differential_with_all_data_types():
    record = _record(patient_id="P-999")
    record.add_observation(_observation(T1, category=ObservationCategory.BREATHING_DIFFICULTY))
    record.add_vital_sign(_temp(T1, 36.5))
    record.add_vital_sign(_temp(T2, 38.5))
    record.add_vital_sign(_spo2(T1, 98.0))
    record.add_vital_sign(_spo2(T2, 93.0))
    record.add_laboratory_result(_glucose(T1, 90.0))
    record.add_laboratory_result(_glucose(T2, 130.0))

    differential = _differential_for(record)
    ids = {c.concern_id for c in differential.possible_clinical_concerns}
    assert "trend::temperature" in ids
    assert "trend::spo2" in ids
    assert "trend::glucose" in ids
    assert any(cid.startswith("cooccurrence::") for cid in ids)


# ---------------------------------------------------------------------
# No causal inference
# ---------------------------------------------------------------------

def test_cooccurrence_concern_description_disclaims_causation():
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.DIZZINESS))
    record.add_vital_sign(_temp(T1, 38.5))
    differential = _differential_for(record)
    matches = [c for c in differential.possible_clinical_concerns if c.concern_id.startswith("cooccurrence::")]
    assert "nedensellik iddia edilmez" in matches[0].description


def test_no_causal_language_anywhere_in_concern_descriptions():
    record = _record()
    record.add_observation(_observation(T1))
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 39.0))
    differential = _differential_for(record)
    forbidden_terms = ["neden oldu", "sebep oldu", "yol açtı", "caused by", "due to", "leads to"]
    for concern in differential.possible_clinical_concerns:
        lowered = concern.description.lower()
        for term in forbidden_terms:
            assert term not in lowered


def test_scope_limitations_include_causation_disclaimer():
    differential = _differential_for(_record())
    joined = " ".join(differential.scope_limitations)
    assert "nedensellik" in joined


# ---------------------------------------------------------------------
# No diagnosis fields
# ---------------------------------------------------------------------

def test_no_diagnosis_attributes_anywhere():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 39.5))
    differential = _differential_for(record)
    objects = [differential] + differential.possible_clinical_concerns
    for obj in objects:
        for forbidden in ("diagnosis", "differential_diagnosis", "disease_name", "condition_name", "disease"):
            assert not hasattr(obj, forbidden)


def test_no_disease_names_or_clinical_terms_in_any_description():
    """
    Bu depoda otoriter bir klinik kural tabanı olmadığı için, hiçbir
    hastalık adı/klinik terim ÜRETİLEMEZ (üretilecek bir kaynak yok).
    Bu test, bilinen birkaç yaygın klinik terimin YANLIŞLIKLA bile
    sızmadığını doğrular.
    """
    record = _record()
    record.add_observation(_observation(T1, category=ObservationCategory.BREATHING_DIFFICULTY))
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 39.5))
    record.add_vital_sign(_spo2(T1, 98.0))
    record.add_vital_sign(_spo2(T2, 88.0))
    differential = _differential_for(record)
    forbidden_terms = [
        "pneumonia", "zatürre", "sepsis", "enfeksiyon", "infection", "fever", "ateş hastalığı",
        "hipoksi", "hypoxia", "diabetes", "diyabet", "kalp yetmezliği", "heart failure",
    ]
    for concern in differential.possible_clinical_concerns:
        lowered = concern.description.lower()
        for term in forbidden_terms:
            assert term not in lowered


# ---------------------------------------------------------------------
# No probability/risk fields
# ---------------------------------------------------------------------

def test_no_probability_or_risk_attributes_anywhere():
    differential = _differential_for(_record())
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 39.5))
    differential2 = _differential_for(record)
    for obj in [differential, differential2] + differential2.possible_clinical_concerns:
        for forbidden in ("probability", "risk_score", "risk_percentage", "likelihood", "confidence_score"):
            assert not hasattr(obj, forbidden)


# ---------------------------------------------------------------------
# No medication / dosage / treatment / clinical threshold fields
# ---------------------------------------------------------------------

def test_no_medication_dosage_treatment_or_threshold_attributes():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 39.5))
    differential = _differential_for(record)
    objects = [differential] + differential.possible_clinical_concerns
    for obj in objects:
        for forbidden in (
            "medication", "medication_recommendation", "drug", "prescription",
            "dosage", "dose", "treatment_duration", "treatment_plan",
            "clinical_threshold", "reference_range", "normal_range",
            "next_best_measurement", "emergency_level",
        ):
            assert not hasattr(obj, forbidden)


# ---------------------------------------------------------------------
# Existing Phase 2-6 compatibility
# ---------------------------------------------------------------------

def test_phase2_domain_routing_unaffected_by_phase7():
    from routing.domain_router import Domain, classify_domain

    assert classify_domain("hastanın ateşi kaç?") == Domain.MEDICAL
    assert classify_domain("gemi hangi bunker ikmalini aldı?") == Domain.MARITIME


def test_differential_assessment_consumes_clinical_reasoning_without_modifying_it():
    """Faz 7, ClinicalReasoning nesnesini SADECE okur -- mutasyona uğratmaz."""
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.5))
    assessment = build_clinical_assessment(record)
    reasoning = build_clinical_reasoning(assessment)
    item_count_before = len(reasoning.items)
    build_differential_assessment(reasoning, assessment)
    assert len(reasoning.items) == item_count_before
