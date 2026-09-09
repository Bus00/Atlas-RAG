"""
tests/test_medical_models.py
--------------------------------
V2 Faz 3: medical/models/ altındaki saf veri modellerinin testleri.
Hiçbir dış servise (Postgres/Chroma/Ollama) bağımlı değil.

Bu testler yalnızca YAPISAL doğrulamayı kontrol eder -- klinik yorum/çıkarım
test edilmez, çünkü modellerin kendisi hiçbir klinik çıkarım yapmaz (bkz.
medical/models/*.py docstring'leri).
"""
from __future__ import annotations

from datetime import datetime

import pytest

from medical.models.errors import MedicalDataValidationError
from medical.models.laboratory_result import COMMON_LAB_TEST_NAMES, LaboratoryResult
from medical.models.observation import Observation, ObservationCategory
from medical.models.patient import Patient
from medical.models.reference_range import ReferenceRange
from medical.models.vital_sign import VitalSign, VitalSignType

NOW = datetime(2026, 1, 1, 12, 0, 0)


# ---------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------

def test_patient_creation_minimal():
    p = Patient(patient_id="P-001")
    assert p.patient_id == "P-001"
    assert p.allergies == []
    assert p.current_medications == []
    assert p.known_history == []
    assert p.age is None


def test_patient_creation_with_optional_fields():
    p = Patient(
        patient_id="P-002",
        display_name="Crew Member A",
        age=34,
        sex="male",
        allergies=["penicillin"],
        current_medications=["metformin"],
        known_history=["bilinen astım öyküsü, hasta tarafından bildirildi"],
        created_at=NOW,
    )
    assert p.age == 34
    assert p.allergies == ["penicillin"]
    assert p.known_history[0].startswith("bilinen astım")


def test_patient_empty_patient_id_rejected():
    with pytest.raises(MedicalDataValidationError):
        Patient(patient_id="")


def test_patient_negative_age_rejected():
    with pytest.raises(MedicalDataValidationError):
        Patient(patient_id="P-003", age=-1)


def test_patient_non_integer_age_rejected():
    with pytest.raises(MedicalDataValidationError):
        Patient(patient_id="P-004", age="34")  # type: ignore[arg-type]


def test_patient_allergies_must_be_non_empty_strings():
    with pytest.raises(MedicalDataValidationError):
        Patient(patient_id="P-005", allergies=[""])


# ---------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------

def test_observation_creation():
    obs = Observation(
        category=ObservationCategory.BREATHING_DIFFICULTY,
        description="hasta nefes darlığı bildiriyor",
        timestamp=NOW,
        source="patient_report",
    )
    assert obs.category == ObservationCategory.BREATHING_DIFFICULTY
    assert obs.timestamp == NOW
    assert obs.source == "patient_report"
    assert obs.reliability is None


def test_observation_does_not_transform_into_diagnosis():
    """
    Kritik ayrım testi: model, gözlemi olduğu gibi saklar -- otomatik bir
    tanı/olasılık üretmez. Bu test, description alanının verildiği gibi
    (yorumlanmadan) korunduğunu doğrular.
    """
    obs = Observation(
        category=ObservationCategory.PAIN,
        description="hasta göğüs ağrısı bildiriyor",
        timestamp=NOW,
        source="patient_report",
    )
    assert obs.description == "hasta göğüs ağrısı bildiriyor"
    assert not hasattr(obs, "diagnosis")
    assert not hasattr(obs, "probability")
    assert not hasattr(obs, "severity_score")


def test_observation_invalid_category_rejected():
    with pytest.raises(MedicalDataValidationError):
        Observation(category="pain", description="x", timestamp=NOW, source="patient_report")  # type: ignore[arg-type]


def test_observation_empty_description_rejected():
    with pytest.raises(MedicalDataValidationError):
        Observation(category=ObservationCategory.PAIN, description="  ", timestamp=NOW, source="patient_report")


def test_observation_invalid_timestamp_type_rejected():
    with pytest.raises(MedicalDataValidationError):
        Observation(category=ObservationCategory.PAIN, description="x", timestamp="2026-01-01", source="patient_report")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# VitalSign
# ---------------------------------------------------------------------

def test_vital_sign_scalar_creation():
    v = VitalSign(
        vital_type=VitalSignType.TEMPERATURE,
        timestamp=NOW,
        source="thermometer",
        unit="°C",
        value=38.4,
    )
    assert v.value == 38.4
    # Model klinik yorum yapmaz -- "infection" gibi bir alan/sonuç üretmez.
    assert not hasattr(v, "interpretation")
    assert not hasattr(v, "clinical_conclusion")


def test_vital_sign_blood_pressure_representation():
    v = VitalSign(
        vital_type=VitalSignType.BLOOD_PRESSURE,
        timestamp=NOW,
        source="bp_monitor",
        unit="mmHg",
        systolic=120,
        diastolic=80,
    )
    assert v.systolic == 120
    assert v.diastolic == 80
    assert v.value is None


def test_vital_sign_blood_pressure_missing_diastolic_rejected():
    with pytest.raises(MedicalDataValidationError):
        VitalSign(
            vital_type=VitalSignType.BLOOD_PRESSURE,
            timestamp=NOW,
            source="bp_monitor",
            unit="mmHg",
            systolic=120,
        )


def test_vital_sign_blood_pressure_with_scalar_value_rejected():
    with pytest.raises(MedicalDataValidationError):
        VitalSign(
            vital_type=VitalSignType.BLOOD_PRESSURE,
            timestamp=NOW,
            source="bp_monitor",
            unit="mmHg",
            value=120,
            systolic=120,
            diastolic=80,
        )


def test_vital_sign_non_bp_with_systolic_rejected():
    with pytest.raises(MedicalDataValidationError):
        VitalSign(
            vital_type=VitalSignType.HEART_RATE,
            timestamp=NOW,
            source="pulse_oximeter",
            unit="bpm",
            systolic=120,
        )


def test_vital_sign_missing_value_rejected():
    with pytest.raises(MedicalDataValidationError):
        VitalSign(vital_type=VitalSignType.SPO2, timestamp=NOW, source="pulse_oximeter", unit="%")


def test_vital_sign_non_numeric_value_rejected():
    with pytest.raises(MedicalDataValidationError):
        VitalSign(
            vital_type=VitalSignType.HEART_RATE,
            timestamp=NOW,
            source="monitor",
            unit="bpm",
            value="fast",  # type: ignore[arg-type]
        )


def test_vital_sign_with_reference_range():
    v = VitalSign(
        vital_type=VitalSignType.SPO2,
        timestamp=NOW,
        source="pulse_oximeter",
        unit="%",
        value=97,
        reference_range=ReferenceRange(low=95, high=100, unit="%"),
    )
    assert v.reference_range.low == 95


# ---------------------------------------------------------------------
# LaboratoryResult
# ---------------------------------------------------------------------

def test_laboratory_result_creation_with_common_test_name():
    assert "Hb" in COMMON_LAB_TEST_NAMES
    lab = LaboratoryResult(test_name="Hb", value=13.5, unit="g/dL", timestamp=NOW, source="onboard_lab")
    assert lab.test_name == "Hb"
    assert lab.reference_range is None


def test_laboratory_result_extensibility_arbitrary_test_name_accepted():
    """
    Genişletilebilirlik testi: COMMON_LAB_TEST_NAMES'te OLMAYAN bir test adı
    da mimari değiştirilmeden kabul edilmeli.
    """
    lab = LaboratoryResult(test_name="D-dimer", value=0.4, unit="mg/L FEU", timestamp=NOW, source="onboard_lab")
    assert lab.test_name == "D-dimer"
    assert "D-dimer" not in COMMON_LAB_TEST_NAMES


def test_laboratory_result_with_optional_reference_range():
    lab = LaboratoryResult(
        test_name="glucose",
        value=95,
        unit="mg/dL",
        timestamp=NOW,
        source="onboard_lab",
        reference_range=ReferenceRange(low=70, high=100, unit="mg/dL"),
    )
    assert lab.reference_range.low == 70
    assert lab.reference_range.high == 100


def test_laboratory_result_empty_test_name_rejected():
    with pytest.raises(MedicalDataValidationError):
        LaboratoryResult(test_name="", value=1.0, unit="mg/dL", timestamp=NOW, source="onboard_lab")


def test_laboratory_result_non_numeric_value_rejected():
    with pytest.raises(MedicalDataValidationError):
        LaboratoryResult(test_name="glucose", value="high", unit="mg/dL", timestamp=NOW, source="onboard_lab")  # type: ignore[arg-type]


def test_laboratory_result_invalid_timestamp_rejected():
    with pytest.raises(MedicalDataValidationError):
        LaboratoryResult(test_name="glucose", value=95, unit="mg/dL", timestamp="2026-01-01", source="onboard_lab")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# ReferenceRange
# ---------------------------------------------------------------------

def test_reference_range_low_greater_than_high_rejected():
    with pytest.raises(MedicalDataValidationError):
        ReferenceRange(low=100, high=50)


def test_reference_range_all_fields_optional():
    rr = ReferenceRange()
    assert rr.low is None
    assert rr.high is None
