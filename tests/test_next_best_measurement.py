"""
tests/test_next_best_measurement.py
------------------------------------------
Faz (Next Best Measurement & Resource-Aware Decision Support), Bölüm A:
medical/next_best_measurement.py testleri (Bölüm B ile köprü dahil).

Bu testler yalnızca YAPISAL aday üretimini, deterministik sıralamayı,
kanıt/izlenebilirlik korumasını ve kaynak-yapılabilirlik eşleşmesini
kontrol eder -- hiçbir tanı/olasılık/tedavi/ilaç test edilmez, çünkü bu
katman bunları üretmez (bkz. medical/next_best_measurement.py docstring'i).
"""
from __future__ import annotations

from datetime import datetime

import pytest

from medical.clinical_assessment import build_clinical_assessment
from medical.clinical_reasoning import ReasoningCategory, build_clinical_reasoning
from medical.differential_assessment import build_differential_assessment
from medical.models.errors import MedicalDataValidationError
from medical.models.laboratory_result import LaboratoryResult
from medical.models.observation import Observation, ObservationCategory
from medical.models.patient import Patient
from medical.models.vital_sign import VitalSign, VitalSignType
from medical.next_best_measurement import (
    FeasibilityStatus,
    MeasurementCandidate,
    MeasurementConsiderationReason,
    NextBestMeasurementAssessment,
    assess_all_feasibility,
    assess_measurement_feasibility,
    build_next_best_measurement_assessment,
    build_next_best_measurement_assessment_from_record,
)
from medical.patient_record import PatientRecord
from medical.resource_availability import Resource, ResourceCategory, ResourceState, build_resource_registry

T1 = datetime(2026, 1, 1, 8, 0, 0)
T2 = datetime(2026, 1, 1, 12, 0, 0)


def _patient(patient_id: str = "P-500") -> Patient:
    return Patient(patient_id=patient_id)


def _record(patient_id: str = "P-500") -> PatientRecord:
    return PatientRecord(patient=_patient(patient_id))


def _temp(ts: datetime, value: float, source: str = "thermometer") -> VitalSign:
    return VitalSign(vital_type=VitalSignType.TEMPERATURE, timestamp=ts, source=source, unit="°C", value=value)


def _spo2(ts: datetime, value: float, source: str = "pulse_oximeter") -> VitalSign:
    return VitalSign(vital_type=VitalSignType.SPO2, timestamp=ts, source=source, unit="%", value=value)


def _glucose(ts: datetime, value: float, source: str = "onboard_lab") -> LaboratoryResult:
    return LaboratoryResult(test_name="glucose", value=value, unit="mg/dL", timestamp=ts, source=source)


def _nb_for(record: PatientRecord) -> NextBestMeasurementAssessment:
    return build_next_best_measurement_assessment_from_record(record)


def _candidate(nb: NextBestMeasurementAssessment, candidate_id: str) -> MeasurementCandidate:
    matches = [c for c in nb.candidates if c.candidate_id == candidate_id]
    assert len(matches) == 1, f"expected exactly one candidate with id {candidate_id}, found {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------
# 1/2. Creation with minimal / complete patient state
# ---------------------------------------------------------------------

def test_minimal_empty_record_produces_only_missing_measurement_candidates():
    nb = _nb_for(_record())
    assert nb.patient_id == "P-500"
    reasons = {c.reason for c in nb.candidates}
    assert reasons == {MeasurementConsiderationReason.MISSING_MEASUREMENT}
    assert len(nb.candidates) == len(VitalSignType)  # hiçbir vital tipi kaydedilmemiş


def test_complete_state_reduces_missing_candidates():
    record = _record()
    for vital_type_builder in (_temp, _spo2):
        record.add_vital_sign(vital_type_builder(T1, 1.0))
        record.add_vital_sign(vital_type_builder(T2, 2.0))
    nb = _nb_for(record)
    missing_labels = {c.measurement_label for c in nb.candidates
                       if c.reason == MeasurementConsiderationReason.MISSING_MEASUREMENT}
    assert "temperature" not in missing_labels
    assert "spo2" not in missing_labels
    assert "heart_rate" in missing_labels


def test_build_next_best_measurement_assessment_rejects_wrong_types():
    record = _record()
    assessment = build_clinical_assessment(record)
    reasoning = build_clinical_reasoning(assessment)
    with pytest.raises(MedicalDataValidationError):
        build_next_best_measurement_assessment("not assessment", reasoning)  # type: ignore[arg-type]
    with pytest.raises(MedicalDataValidationError):
        build_next_best_measurement_assessment(assessment, "not reasoning")  # type: ignore[arg-type]


def test_build_rejects_mismatched_patient_ids_between_assessment_and_reasoning():
    record_a, record_b = _record("P-A"), _record("P-B")
    assessment_a = build_clinical_assessment(record_a)
    reasoning_b = build_clinical_reasoning(build_clinical_assessment(record_b))
    with pytest.raises(MedicalDataValidationError):
        build_next_best_measurement_assessment(assessment_a, reasoning_b)


# ---------------------------------------------------------------------
# 3. Measurement candidate representation
# ---------------------------------------------------------------------

def test_candidate_has_expected_structural_fields():
    nb = _nb_for(_record())
    candidate = _candidate(nb, "missing::temperature")
    assert candidate.measurement_label == "temperature"
    assert candidate.measurement_kind == "vital_sign"
    assert candidate.required_resource_id == "device:temperature"
    assert candidate.patient_id == "P-500"
    assert "yapısal" in candidate.rationale or "veri" in candidate.rationale


# ---------------------------------------------------------------------
# 4. Missing measurement detection
# ---------------------------------------------------------------------

def test_missing_vital_detected_as_candidate():
    nb = _nb_for(_record())
    ids = {c.candidate_id for c in nb.candidates}
    assert "missing::temperature" in ids
    assert "missing::heart_rate" in ids
    assert "missing::blood_pressure" in ids
    assert "missing::respiratory_rate" in ids
    assert "missing::spo2" in ids


def test_no_missing_measurement_candidates_for_laboratory():
    """Laboratuvar açık-uçlu olduğu için 'eksik lab paneli' İCAT EDİLMEZ."""
    nb = _nb_for(_record())
    assert not any(c.measurement_kind == "laboratory" and c.reason == MeasurementConsiderationReason.MISSING_MEASUREMENT
                   for c in nb.candidates)


# ---------------------------------------------------------------------
# 5. Repeated measurement handling
# ---------------------------------------------------------------------

def test_two_measurements_do_not_produce_insufficient_trend_candidate():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 37.0))
    nb = _nb_for(record)
    assert not any(c.candidate_id == "insufficient_trend::temperature" for c in nb.candidates)


def test_single_measurement_produces_insufficient_trend_candidate():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    nb = _nb_for(record)
    candidate = _candidate(nb, "insufficient_trend::temperature")
    assert candidate.reason == MeasurementConsiderationReason.INSUFFICIENT_FOR_TREND


# ---------------------------------------------------------------------
# 6. Timestamp ordering (of underlying evidence within a candidate)
# ---------------------------------------------------------------------

def test_supporting_evidence_reflects_chronological_series():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    nb = _nb_for(record)
    candidate = _candidate(nb, "insufficient_trend::temperature")
    chronology_items = [i for i in candidate.supporting_evidence if i.category == ReasoningCategory.CHRONOLOGY]
    assert len(chronology_items) == 1
    assert chronology_items[0].evidence[0].timestamp == T1


# ---------------------------------------------------------------------
# 7/8. Deterministic candidate ordering / stable output across runs
# ---------------------------------------------------------------------

def test_candidate_ordering_missing_then_insufficient_then_conflict():
    record = _record()
    record.add_vital_sign(_spo2(T1, 98.0))          # spo2: yeterli tek ölçüm -> insufficient_trend
    record.add_vital_sign(_temp(T1, 36.0, source="a"))
    record.add_vital_sign(_temp(T1, 39.0, source="b"))  # temperature: aynı T1'de çelişki
    nb = _nb_for(record)
    ids = [c.candidate_id for c in nb.candidates]
    missing_idx = [i for i, cid in enumerate(ids) if cid.startswith("missing::")]
    insufficient_idx = [i for i, cid in enumerate(ids) if cid.startswith("insufficient_trend::")]
    conflict_idx = [i for i, cid in enumerate(ids) if cid.startswith("conflict_resolution::")]
    assert max(missing_idx) < min(insufficient_idx)
    assert max(insufficient_idx) < min(conflict_idx)


def test_vital_canonical_order_before_lab_alphabetical_order_within_same_reason():
    record = _record()
    record.add_laboratory_result(_glucose(T1, 90.0))  # lab: tek ölçüm -> insufficient_trend
    record.add_vital_sign(_spo2(T1, 98.0))             # vital: tek ölçüm -> insufficient_trend
    nb = _nb_for(record)
    ids = [c.candidate_id for c in nb.candidates if c.candidate_id.startswith("insufficient_trend::")]
    assert ids.index("insufficient_trend::spo2") < ids.index("insufficient_trend::glucose")


def test_same_input_produces_identical_candidates_every_run():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.0))
    record.add_laboratory_result(_glucose(T1, 95.0))
    assessment = build_clinical_assessment(record)
    reasoning = build_clinical_reasoning(assessment)
    differential = build_differential_assessment(reasoning, assessment)

    nb1 = build_next_best_measurement_assessment(assessment, reasoning, differential)
    nb2 = build_next_best_measurement_assessment(assessment, reasoning, differential)

    assert [c.candidate_id for c in nb1.candidates] == [c.candidate_id for c in nb2.candidates]
    assert [c.rationale for c in nb1.candidates] == [c.rationale for c in nb2.candidates]
    assert nb1.scope_limitations == nb2.scope_limitations


def test_deterministic_across_hash_seeds_smoke():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 38.0))
    record.add_laboratory_result(_glucose(T1, 95.0))
    results = [tuple(c.candidate_id for c in _nb_for(record).candidates) for _ in range(5)]
    assert len(set(results)) == 1


# ---------------------------------------------------------------------
# 9/10/11/12. Resource creation / availability / unavailable / unknown
#             (covered structurally in test_resource_availability.py;
#             here we test the next_best_measurement <-> resource bridge)
# ---------------------------------------------------------------------

def test_resource_bridge_available_resource():
    record = _record()
    nb = _nb_for(record)
    registry = build_resource_registry([Resource("device:temperature", ResourceCategory.MEASUREMENT_DEVICE, ResourceState.AVAILABLE)])
    feasibility = assess_measurement_feasibility(_candidate(nb, "missing::temperature"), registry)
    assert feasibility.status == FeasibilityStatus.FEASIBLE
    assert feasibility.limitation is None


# ---------------------------------------------------------------------
# 13. Measurement requiring a resource
# ---------------------------------------------------------------------

def test_every_candidate_declares_a_required_resource():
    record = _record()
    record.add_laboratory_result(_glucose(T1, 90.0))
    nb = _nb_for(record)
    assert all(c.required_resource_id for c in nb.candidates)


def test_laboratory_candidates_require_generic_laboratory_capability():
    record = _record()
    record.add_laboratory_result(_glucose(T1, 90.0))
    nb = _nb_for(record)
    candidate = _candidate(nb, "insufficient_trend::glucose")
    assert candidate.required_resource_id == "laboratory_capability"


def test_blood_pressure_missing_candidate_requires_single_device_resource():
    nb = _nb_for(_record())
    candidate = _candidate(nb, "missing::blood_pressure")
    assert candidate.required_resource_id == "device:blood_pressure"


# ---------------------------------------------------------------------
# 14/15/16. Feasible / infeasible / unknown feasibility
# ---------------------------------------------------------------------

def test_feasible_when_resource_available():
    nb = _nb_for(_record())
    registry = build_resource_registry([Resource("device:temperature", ResourceCategory.MEASUREMENT_DEVICE, ResourceState.AVAILABLE)])
    feasibility = assess_measurement_feasibility(_candidate(nb, "missing::temperature"), registry)
    assert feasibility.status == FeasibilityStatus.FEASIBLE


def test_infeasible_when_resource_unavailable():
    nb = _nb_for(_record())
    registry = build_resource_registry([Resource("device:temperature", ResourceCategory.MEASUREMENT_DEVICE, ResourceState.UNAVAILABLE)])
    feasibility = assess_measurement_feasibility(_candidate(nb, "missing::temperature"), registry)
    assert feasibility.status == FeasibilityStatus.NOT_FEASIBLE
    assert feasibility.limitation is not None


def test_unknown_feasibility_when_resource_not_registered():
    nb = _nb_for(_record())
    registry = build_resource_registry([])
    feasibility = assess_measurement_feasibility(_candidate(nb, "missing::temperature"), registry)
    assert feasibility.status == FeasibilityStatus.UNKNOWN
    assert feasibility.limitation is not None


# ---------------------------------------------------------------------
# 17/18. Missing resource / missing measurement never becomes negative
# ---------------------------------------------------------------------

def test_missing_measurement_rationale_never_phrased_as_negative():
    nb = _nb_for(_record())
    candidate = _candidate(nb, "missing::temperature")
    lowered = candidate.rationale.lower()
    assert "negatif" not in lowered
    assert "normal" not in lowered


def test_unavailable_resource_never_phrased_as_measurement_not_needed():
    nb = _nb_for(_record())
    registry = build_resource_registry([Resource("device:temperature", ResourceCategory.MEASUREMENT_DEVICE, ResourceState.UNAVAILABLE)])
    feasibility = assess_measurement_feasibility(_candidate(nb, "missing::temperature"), registry)
    lowered = feasibility.limitation.lower()
    assert "ihtiyaç yok" not in lowered
    assert "gerekmiyor" not in lowered


def test_candidate_persists_even_when_resource_infeasible():
    """DO NOT silently remove the measurement -- aday, kaynak yokken bile temsil edilmeye devam eder."""
    nb = _nb_for(_record())
    registry = build_resource_registry([Resource("device:temperature", ResourceCategory.MEASUREMENT_DEVICE, ResourceState.UNAVAILABLE)])
    feasibility_list = assess_all_feasibility(nb, registry)
    assert any(f.candidate.candidate_id == "missing::temperature" for f in feasibility_list)
    assert len(feasibility_list) == len(nb.candidates)  # hiçbir aday sessizce kaldırılmadı


# ---------------------------------------------------------------------
# 19/20/21/22. Provenance / patient_id / timestamp / source preservation
# ---------------------------------------------------------------------

def test_patient_id_preserved_everywhere():
    record = _record(patient_id="P-321")
    record.add_vital_sign(_temp(T1, 36.0))
    nb = _nb_for(record)
    assert nb.patient_id == "P-321"
    for c in nb.candidates:
        assert c.patient_id == "P-321"
    registry = build_resource_registry([])
    for f in assess_all_feasibility(nb, registry):
        assert f.patient_id == "P-321"


def test_timestamp_and_source_preserved_in_supporting_evidence():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0, source="onboard_medic"))
    nb = _nb_for(record)
    candidate = _candidate(nb, "insufficient_trend::temperature")
    chronology_item = [i for i in candidate.supporting_evidence if i.category == ReasoningCategory.CHRONOLOGY][0]
    assert chronology_item.evidence[0].timestamp == T1
    assert chronology_item.evidence[0].source == "onboard_medic"


# ---------------------------------------------------------------------
# 23/24. Conflict preservation / no winner selection
# ---------------------------------------------------------------------

def test_conflict_candidate_preserves_both_conflicting_sources():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0, source="thermometer_a"))
    record.add_vital_sign(_temp(T1, 39.0, source="thermometer_b"))
    nb = _nb_for(record)
    candidate = _candidate(nb, "conflict_resolution::temperature")
    conflict_item = [i for i in candidate.supporting_evidence if i.category == ReasoningCategory.MEASUREMENT_CONFLICT][0]
    sources = {e.source for e in conflict_item.evidence}
    assert sources == {"thermometer_a", "thermometer_b"}


def test_conflict_candidate_never_selects_a_winner():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0, source="thermometer_a"))
    record.add_vital_sign(_temp(T1, 39.0, source="thermometer_b"))
    nb = _nb_for(record)
    candidate = _candidate(nb, "conflict_resolution::temperature")
    for forbidden in ("winner", "resolved_value", "correct_source", "chosen_value"):
        assert not hasattr(candidate, forbidden)
    assert "hiçbir değer otomatik olarak seçilmedi" in candidate.related_missing_information[0].statement


# ---------------------------------------------------------------------
# 25/26/27/28. Compatibility with PatientRecord / ClinicalAssessment /
#              ClinicalReasoning / DifferentialAssessment
# ---------------------------------------------------------------------

def test_compatible_with_patient_record_directly():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    nb = build_next_best_measurement_assessment_from_record(record)
    assert isinstance(nb, NextBestMeasurementAssessment)


def test_compatible_with_clinical_assessment_and_reasoning_explicit_pipeline():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    assessment = build_clinical_assessment(record)
    reasoning = build_clinical_reasoning(assessment)
    nb = build_next_best_measurement_assessment(assessment, reasoning)
    assert nb.patient_id == assessment.patient_id


def test_compatible_with_differential_assessment_cross_referencing():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 39.0))  # increasing trend -> Faz 7 concern trend::temperature
    assessment = build_clinical_assessment(record)
    reasoning = build_clinical_reasoning(assessment)
    differential = build_differential_assessment(reasoning, assessment)
    nb = build_next_best_measurement_assessment(assessment, reasoning, differential)
    # temperature zaten 2 ölçüme sahip (insufficient_trend adayı YOK), ama spo2 hiç yok (missing adayı VAR)
    missing_spo2 = _candidate(nb, "missing::spo2")
    assert missing_spo2.related_concern_ids == []  # spo2 ile ilgili bir concern yok


def test_next_best_measurement_from_record_without_differential():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    nb = build_next_best_measurement_assessment_from_record(record, include_differential=False)
    candidate = _candidate(nb, "insufficient_trend::temperature")
    assert candidate.related_concern_ids == []


# ---------------------------------------------------------------------
# 29/30/31/32/33. Deterministic behavior / no randomness / no current-time /
#                 no network / no LLM dependency
# ---------------------------------------------------------------------

def test_module_has_no_network_or_llm_dependencies():
    import medical.next_best_measurement as module
    import medical.resource_availability as resource_module
    source_nb = open(module.__file__, encoding="utf-8").read()
    source_res = open(resource_module.__file__, encoding="utf-8").read()
    for forbidden in ("requests", "httpx", "urllib", "socket", "openai", "anthropic", "ollama", "datetime.now", "uuid", "random"):
        assert forbidden not in source_nb
        assert forbidden not in source_res


# ---------------------------------------------------------------------
# 34-40. No diagnosis / probability / medication / dosage / treatment /
#        thresholds / fabricated evidence
# ---------------------------------------------------------------------

def test_no_diagnosis_probability_medication_dosage_treatment_threshold_attributes():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 39.5))
    nb = _nb_for(record)
    objects = [nb] + nb.candidates
    for obj in objects:
        for forbidden in (
            "diagnosis", "differential_diagnosis", "disease", "probability", "risk_score",
            "risk_percentage", "information_gain", "medication", "medication_recommendation",
            "dosage", "dose", "treatment_duration", "treatment_plan", "clinical_threshold",
            "reference_range", "emergency_level", "evacuation_decision",
        ):
            assert not hasattr(obj, forbidden)


def test_no_disease_names_or_clinical_claims_in_rationale():
    record = _record()
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 39.5))
    record.add_vital_sign(_spo2(T1, 98.0, source="a"))
    record.add_vital_sign(_spo2(T1, 80.0, source="b"))
    nb = _nb_for(record)
    forbidden_terms = ["pneumonia", "zatürre", "sepsis", "enfeksiyon", "infection", "rules out", "confirms",
                        "dışlar", "doğrular"]
    for candidate in nb.candidates:
        lowered = candidate.rationale.lower()
        for term in forbidden_terms:
            assert term not in lowered


def test_no_information_gain_claim_anywhere():
    nb = _nb_for(_record())
    joined = " ".join(nb.scope_limitations).lower()
    assert "information gain" in joined  # sadece HESAPLANMADIĞINI belirten bir cümlede geçmeli
    assert "hesaplamaz" in joined


# ---------------------------------------------------------------------
# 41/42. Resource limitation explicitly represented / scope limitations exposed
# ---------------------------------------------------------------------

def test_resource_limitation_explicitly_represented_not_hidden():
    nb = _nb_for(_record())
    registry = build_resource_registry([])  # hiçbir kaynak bilinmiyor
    feasibility_list = assess_all_feasibility(nb, registry)
    assert all(f.status == FeasibilityStatus.UNKNOWN for f in feasibility_list)
    assert all(f.limitation is not None for f in feasibility_list)


def test_scope_limitations_always_present_even_for_empty_record():
    nb = _nb_for(_record())
    assert len(nb.scope_limitations) > 0
    joined = " ".join(nb.scope_limitations)
    assert "tanı" in joined
    assert "ilaç" in joined or "tedavi" in joined


# ---------------------------------------------------------------------
# Phase 2 routing / backward-compatibility sanity
# ---------------------------------------------------------------------

def test_phase2_domain_routing_unaffected():
    from routing.domain_router import Domain, classify_domain

    assert classify_domain("hastanın nabzı kaç?") == Domain.MEDICAL
    assert classify_domain("gemi hangi bunker ikmalini aldı?") == Domain.MARITIME
