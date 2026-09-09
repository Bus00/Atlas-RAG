"""
tests/test_patient_record.py
--------------------------------
V2 Faz 4: medical/patient_record.py (PatientRecord container) testleri.

Bu testler yalnızca YAPISAL doğrulama, koleksiyon davranışı ve deterministik
sıralamayı kontrol eder -- klinik yorum/çıkarım test edilmez, çünkü
PatientRecord hiçbir klinik çıkarım yapmaz (bkz. medical/patient_record.py
docstring'i).

Ayrıca Faz 2 (domain routing) davranışının Faz 4 ile bozulmadığını
doğrulayan bir regresyon testi içerir.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from medical.models.errors import MedicalDataValidationError
from medical.models.laboratory_result import LaboratoryResult
from medical.models.observation import Observation, ObservationCategory
from medical.models.patient import Patient
from medical.models.vital_sign import VitalSign, VitalSignType
from medical.patient_record import PatientRecord

T1 = datetime(2026, 1, 1, 8, 0, 0)
T2 = datetime(2026, 1, 1, 12, 0, 0)
T3 = datetime(2026, 1, 1, 18, 0, 0)


def _patient(patient_id: str = "P-100") -> Patient:
    return Patient(patient_id=patient_id)


def _observation(ts: datetime, description: str = "hasta ateş bildiriyor") -> Observation:
    return Observation(
        category=ObservationCategory.SYMPTOM,
        description=description,
        timestamp=ts,
        source="patient_report",
    )


def _vital_sign(ts: datetime, value: float = 37.0) -> VitalSign:
    return VitalSign(
        vital_type=VitalSignType.TEMPERATURE,
        timestamp=ts,
        source="thermometer",
        unit="°C",
        value=value,
    )


def _lab_result(ts: datetime, value: float = 95.0) -> LaboratoryResult:
    return LaboratoryResult(
        test_name="glucose",
        value=value,
        unit="mg/dL",
        timestamp=ts,
        source="onboard_lab",
    )


# ---------------------------------------------------------------------
# Creating a patient record
# ---------------------------------------------------------------------

def test_create_patient_record_minimal():
    record = PatientRecord(patient=_patient())
    assert record.patient_id == "P-100"
    assert record.get_observations() == []
    assert record.get_vital_signs() == []
    assert record.get_laboratory_results() == []


def test_create_patient_record_requires_patient_instance():
    with pytest.raises(MedicalDataValidationError):
        PatientRecord(patient="P-100")  # type: ignore[arg-type]


def test_patient_record_compatible_with_phase3_patient_model():
    """Faz 3 Patient modeliyle uyumluluk: opsiyonel alanlar aynen korunur."""
    p = Patient(patient_id="P-101", display_name="Crew Member B", age=41, allergies=["iodine"])
    record = PatientRecord(patient=p)
    assert record.patient.display_name == "Crew Member B"
    assert record.patient.age == 41
    assert record.patient.allergies == ["iodine"]


# ---------------------------------------------------------------------
# Adding observations
# ---------------------------------------------------------------------

def test_add_single_observation():
    record = PatientRecord(patient=_patient())
    obs = _observation(T1)
    record.add_observation(obs)
    assert record.get_observations() == [obs]


def test_add_multiple_observations():
    record = PatientRecord(patient=_patient())
    obs1, obs2, obs3 = _observation(T1), _observation(T2), _observation(T3)
    record.add_observation(obs1)
    record.add_observation(obs2)
    record.add_observation(obs3)
    assert len(record.get_observations()) == 3


def test_add_observation_rejects_wrong_type():
    record = PatientRecord(patient=_patient())
    with pytest.raises(MedicalDataValidationError):
        record.add_observation("hasta ateş bildiriyor")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Adding vital signs
# ---------------------------------------------------------------------

def test_add_single_vital_sign():
    record = PatientRecord(patient=_patient())
    v = _vital_sign(T1)
    record.add_vital_sign(v)
    assert record.get_vital_signs() == [v]


def test_add_multiple_vital_signs():
    record = PatientRecord(patient=_patient())
    record.add_vital_sign(_vital_sign(T1, value=36.8))
    record.add_vital_sign(_vital_sign(T2, value=37.2))
    assert len(record.get_vital_signs()) == 2


def test_add_vital_sign_rejects_wrong_type():
    record = PatientRecord(patient=_patient())
    with pytest.raises(MedicalDataValidationError):
        record.add_vital_sign({"value": 37.0})  # type: ignore[arg-type]


def test_patient_record_compatible_with_phase3_blood_pressure_representation():
    """Faz 3'teki systolic/diastolic BP temsili PatientRecord üzerinden de çalışır."""
    record = PatientRecord(patient=_patient())
    bp = VitalSign(
        vital_type=VitalSignType.BLOOD_PRESSURE,
        timestamp=T1,
        source="bp_monitor",
        unit="mmHg",
        systolic=120,
        diastolic=80,
    )
    record.add_vital_sign(bp)
    stored = record.get_vital_signs()[0]
    assert stored.systolic == 120
    assert stored.diastolic == 80
    assert stored.value is None


# ---------------------------------------------------------------------
# Adding laboratory results
# ---------------------------------------------------------------------

def test_add_single_laboratory_result():
    record = PatientRecord(patient=_patient())
    lab = _lab_result(T1)
    record.add_laboratory_result(lab)
    assert record.get_laboratory_results() == [lab]


def test_add_multiple_laboratory_results():
    record = PatientRecord(patient=_patient())
    record.add_laboratory_result(_lab_result(T1, value=90.0))
    record.add_laboratory_result(_lab_result(T2, value=110.0))
    assert len(record.get_laboratory_results()) == 2


def test_add_laboratory_result_rejects_wrong_type():
    record = PatientRecord(patient=_patient())
    with pytest.raises(MedicalDataValidationError):
        record.add_laboratory_result(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Deterministic timestamp ordering
# ---------------------------------------------------------------------

def test_observations_returned_in_timestamp_order_regardless_of_insertion_order():
    record = PatientRecord(patient=_patient())
    obs_late = _observation(T3, description="geç kayıt")
    obs_early = _observation(T1, description="erken kayıt")
    obs_mid = _observation(T2, description="orta kayıt")
    record.add_observation(obs_late)
    record.add_observation(obs_early)
    record.add_observation(obs_mid)

    ordered = record.get_observations()
    assert [o.timestamp for o in ordered] == [T1, T2, T3]
    assert ordered[0] is obs_early
    assert ordered[-1] is obs_late


def test_vital_signs_returned_in_timestamp_order():
    record = PatientRecord(patient=_patient())
    v_late, v_early = _vital_sign(T3, value=38.0), _vital_sign(T1, value=36.5)
    record.add_vital_sign(v_late)
    record.add_vital_sign(v_early)

    ordered = record.get_vital_signs()
    assert [v.timestamp for v in ordered] == [T1, T3]


def test_laboratory_results_returned_in_timestamp_order():
    record = PatientRecord(patient=_patient())
    lab_late, lab_early = _lab_result(T3, value=120.0), _lab_result(T1, value=85.0)
    record.add_laboratory_result(lab_late)
    record.add_laboratory_result(lab_early)

    ordered = record.get_laboratory_results()
    assert [lr.timestamp for lr in ordered] == [T1, T3]


def test_timestamp_ordering_is_stable_for_equal_timestamps():
    """
    Aynı timestamp'e sahip iki kayıt için sıralama, ekleme sırasını korur
    (Python sorted()/sort() kararlıdır -- bu davranış bilinçli olarak
    test ediliyor, sadece varsayılmıyor).
    """
    record = PatientRecord(patient=_patient())
    obs_a = _observation(T1, description="ilk eklenen")
    obs_b = _observation(T1, description="ikinci eklenen")
    record.add_observation(obs_a)
    record.add_observation(obs_b)

    ordered = record.get_observations()
    assert ordered[0] is obs_a
    assert ordered[1] is obs_b


# ---------------------------------------------------------------------
# Empty collections
# ---------------------------------------------------------------------

def test_empty_collections_return_empty_lists_not_none():
    record = PatientRecord(patient=_patient())
    assert record.get_observations() == []
    assert record.get_vital_signs() == []
    assert record.get_laboratory_results() == []


# ---------------------------------------------------------------------
# Invalid types / structural validation at construction time
# ---------------------------------------------------------------------

def test_constructor_rejects_non_list_observations():
    with pytest.raises(MedicalDataValidationError):
        PatientRecord(patient=_patient(), observations=_observation(T1))  # type: ignore[arg-type]


def test_constructor_rejects_list_with_wrong_item_type():
    with pytest.raises(MedicalDataValidationError):
        PatientRecord(patient=_patient(), vital_signs=["not a vital sign"])  # type: ignore[arg-type]


def test_constructor_accepts_prevalidated_collections():
    obs = _observation(T1)
    v = _vital_sign(T1)
    lab = _lab_result(T1)
    record = PatientRecord(
        patient=_patient(),
        observations=[obs],
        vital_signs=[v],
        laboratory_results=[lab],
    )
    assert record.get_observations() == [obs]
    assert record.get_vital_signs() == [v]
    assert record.get_laboratory_results() == [lab]


# ---------------------------------------------------------------------
# Source and timestamp preservation
# ---------------------------------------------------------------------

def test_observation_source_and_timestamp_preserved():
    record = PatientRecord(patient=_patient())
    obs = _observation(T2, description="hasta baş dönmesi bildiriyor")
    record.add_observation(obs)
    stored = record.get_observations()[0]
    assert stored.source == "patient_report"
    assert stored.timestamp == T2


def test_vital_sign_source_and_timestamp_preserved():
    record = PatientRecord(patient=_patient())
    v = _vital_sign(T2, value=39.1)
    record.add_vital_sign(v)
    stored = record.get_vital_signs()[0]
    assert stored.source == "thermometer"
    assert stored.timestamp == T2
    assert stored.value == 39.1


def test_laboratory_result_source_and_timestamp_preserved():
    record = PatientRecord(patient=_patient())
    lab = _lab_result(T2, value=101.0)
    record.add_laboratory_result(lab)
    stored = record.get_laboratory_results()[0]
    assert stored.source == "onboard_lab"
    assert stored.timestamp == T2


# ---------------------------------------------------------------------
# get_* returns independent copies (mutation safety)
# ---------------------------------------------------------------------

def test_get_observations_returns_copy_not_internal_reference():
    record = PatientRecord(patient=_patient())
    record.add_observation(_observation(T1))
    result = record.get_observations()
    result.append(_observation(T2))
    # İç durum, döndürülen listenin mutasyonundan etkilenmemeli.
    assert len(record.get_observations()) == 1


# ---------------------------------------------------------------------
# PatientRecord itself never performs clinical reasoning
# ---------------------------------------------------------------------

def test_patient_record_has_no_clinical_reasoning_attributes():
    """
    Kritik ayrım testi (Faz 3 ile aynı ilke): PatientRecord container'ı
    hiçbir tanı/olasılık/aciliyet alanı üretmez veya taşımaz.
    """
    record = PatientRecord(patient=_patient())
    for forbidden in ("diagnosis", "differential_diagnosis", "probability",
                       "severity_score", "emergency_level", "medication_recommendation"):
        assert not hasattr(record, forbidden)


# ---------------------------------------------------------------------
# Regression: Phase 2 domain routing remains intact
# ---------------------------------------------------------------------

def test_phase2_domain_routing_unaffected_by_phase4():
    """
    Faz 4, routing/domain_router.py'yi hiç değiştirmedi. Bu test, Faz 2
    davranışının (MEDICAL sınıflandırması) hâlâ çalıştığını doğrular --
    tam regresyon paketi tests/test_domain_router.py ve
    tests/test_pipeline_domain_routing.py'de zaten mevcut, bu sadece
    Faz 4 dosyasından hızlı bir sağlık kontrolü (sanity check).
    """
    from routing.domain_router import Domain, classify_domain

    assert classify_domain("hastanın ateşi kaç derece?") == Domain.MEDICAL
    assert classify_domain("gemi hangi limanda?") == Domain.MARITIME
