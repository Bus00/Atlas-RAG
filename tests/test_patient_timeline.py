"""
tests/test_patient_timeline.py
------------------------------------
medical/patient_timeline.py testleri.

Bu testler yalnızca YAPISAL organizasyonu (kronolojik sıra, aynı zaman
damgası davranışı, hasta ID tutarlılığı, boş zaman çizelgesi) kontrol
eder -- hiçbir klinik çıkarım test edilmez, çünkü bu katman bunları
üretmez (bkz. medical/patient_timeline.py docstring'i).
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
from medical.patient_timeline import (
    PatientTimeline,
    TimelineEntry,
    TimelineEntryType,
    build_patient_timeline,
)

T1 = datetime(2026, 1, 1, 8, 0, 0)
T2 = datetime(2026, 1, 1, 12, 0, 0)
T3 = datetime(2026, 1, 1, 18, 0, 0)


def _patient(patient_id: str = "P-600") -> Patient:
    return Patient(patient_id=patient_id)


def _record(patient_id: str = "P-600") -> PatientRecord:
    return PatientRecord(patient=_patient(patient_id))


def _observation(ts: datetime, category: ObservationCategory = ObservationCategory.PAIN,
                  description: str = "karın ağrısı", source: str = "patient_report") -> Observation:
    return Observation(category=category, description=description, timestamp=ts, source=source)


def _temp(ts: datetime, value: float, source: str = "thermometer") -> VitalSign:
    return VitalSign(vital_type=VitalSignType.TEMPERATURE, timestamp=ts, source=source, unit="°C", value=value)


def _bp(ts: datetime, systolic: float, diastolic: float, source: str = "bp_monitor") -> VitalSign:
    return VitalSign(vital_type=VitalSignType.BLOOD_PRESSURE, timestamp=ts, source=source, unit="mmHg",
                      systolic=systolic, diastolic=diastolic)


def _glucose(ts: datetime, value: float, source: str = "onboard_lab") -> LaboratoryResult:
    return LaboratoryResult(test_name="glucose", value=value, unit="mg/dL", timestamp=ts, source=source)


# ---------------------------------------------------------------------
# Empty timeline
# ---------------------------------------------------------------------

def test_empty_record_produces_empty_timeline():
    timeline = build_patient_timeline(_record())
    assert timeline.patient_id == "P-600"
    assert timeline.entries == []


def test_build_patient_timeline_rejects_non_record():
    with pytest.raises(MedicalDataValidationError):
        build_patient_timeline("not a record")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Chronological ordering
# ---------------------------------------------------------------------

def test_entries_sorted_chronologically_regardless_of_insertion_order():
    record = _record()
    record.add_observation(_observation(T3))
    record.add_vital_sign(_temp(T1, 36.5))
    record.add_laboratory_result(_glucose(T2, 95.0))
    timeline = build_patient_timeline(record)
    timestamps = [e.timestamp for e in timeline.entries]
    assert timestamps == sorted(timestamps)
    assert timestamps == [T1, T2, T3]


def test_multiple_entries_per_type_remain_chronological():
    record = _record()
    record.add_vital_sign(_temp(T3, 39.0))
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_vital_sign(_temp(T2, 37.5))
    timeline = build_patient_timeline(record)
    vital_entries = timeline.entries_by_type(TimelineEntryType.VITAL_SIGN)
    assert [e.timestamp for e in vital_entries] == [T1, T2, T3]


# ---------------------------------------------------------------------
# Same-timestamp deterministic behavior
# ---------------------------------------------------------------------

def test_same_timestamp_entries_use_fixed_type_order_observation_vital_lab():
    record = _record()
    record.add_laboratory_result(_glucose(T1, 95.0))  # eklenme sırası: lab önce
    record.add_vital_sign(_temp(T1, 37.0))              # sonra vital
    record.add_observation(_observation(T1))            # sonra observation
    timeline = build_patient_timeline(record)
    types_in_order = [e.entry_type for e in timeline.entries]
    # Eklenme sırasından BAĞIMSIZ olarak sabit tür-sırası: observation -> vital -> lab
    assert types_in_order == [
        TimelineEntryType.OBSERVATION, TimelineEntryType.VITAL_SIGN, TimelineEntryType.LABORATORY_RESULT,
    ]


def test_same_timestamp_multiple_observations_preserve_insertion_order():
    record = _record()
    obs_a = _observation(T1, description="ilk gözlem")
    obs_b = _observation(T1, description="ikinci gözlem")
    record.add_observation(obs_a)
    record.add_observation(obs_b)
    timeline = build_patient_timeline(record)
    obs_entries = timeline.entries_by_type(TimelineEntryType.OBSERVATION)
    assert obs_entries[0].original_record is obs_a
    assert obs_entries[1].original_record is obs_b


def test_same_timestamp_behavior_is_deterministic_across_runs():
    record = _record()
    record.add_laboratory_result(_glucose(T1, 95.0))
    record.add_vital_sign(_temp(T1, 37.0))
    record.add_observation(_observation(T1))
    results = [tuple(e.entry_type for e in build_patient_timeline(record).entries) for _ in range(5)]
    assert len(set(results)) == 1


# ---------------------------------------------------------------------
# Patient ID consistency
# ---------------------------------------------------------------------

def test_patient_id_preserved_on_timeline_and_every_entry():
    record = _record(patient_id="P-999")
    record.add_observation(_observation(T1))
    record.add_vital_sign(_temp(T1, 37.0))
    record.add_laboratory_result(_glucose(T1, 95.0))
    timeline = build_patient_timeline(record)
    assert timeline.patient_id == "P-999"
    assert all(e.patient_id == "P-999" for e in timeline.entries)


# ---------------------------------------------------------------------
# Entry type representation
# ---------------------------------------------------------------------

def test_observation_entry_representation():
    record = _record()
    obs = _observation(T1, category=ObservationCategory.DIZZINESS, description="baş dönmesi")
    record.add_observation(obs)
    timeline = build_patient_timeline(record)
    entry = timeline.entries[0]
    assert entry.entry_type == TimelineEntryType.OBSERVATION
    assert entry.description == "dizziness: baş dönmesi"
    assert entry.original_record is obs


def test_vital_sign_entry_representation():
    record = _record()
    v = _temp(T1, 38.4, source="onboard_medic")
    record.add_vital_sign(v)
    timeline = build_patient_timeline(record)
    entry = timeline.entries[0]
    assert entry.entry_type == TimelineEntryType.VITAL_SIGN
    assert entry.description == "temperature: 38.4 °C"
    assert entry.source == "onboard_medic"
    assert entry.original_record is v


def test_blood_pressure_entry_representation():
    record = _record()
    v = _bp(T1, 120, 80)
    record.add_vital_sign(v)
    timeline = build_patient_timeline(record)
    entry = timeline.entries[0]
    assert entry.description == "blood_pressure: 120/80 mmHg"


def test_laboratory_result_entry_representation():
    record = _record()
    lab = _glucose(T1, 95.0, source="shore_lab")
    record.add_laboratory_result(lab)
    timeline = build_patient_timeline(record)
    entry = timeline.entries[0]
    assert entry.entry_type == TimelineEntryType.LABORATORY_RESULT
    assert entry.description == "glucose: 95.0 mg/dL"
    assert entry.source == "shore_lab"
    assert entry.original_record is lab


# ---------------------------------------------------------------------
# Does not modify existing data / does not fabricate clinical conclusions
# ---------------------------------------------------------------------

def test_building_timeline_does_not_mutate_patient_record():
    record = _record()
    record.add_observation(_observation(T1))
    record.add_vital_sign(_temp(T1, 37.0))
    obs_count_before = len(record.get_observations())
    vital_count_before = len(record.get_vital_signs())
    build_patient_timeline(record)
    assert len(record.get_observations()) == obs_count_before
    assert len(record.get_vital_signs()) == vital_count_before


def test_no_clinical_interpretation_attributes_on_timeline_or_entries():
    record = _record()
    record.add_vital_sign(_temp(T1, 39.5))
    timeline = build_patient_timeline(record)
    objects = [timeline] + timeline.entries
    for obj in objects:
        for forbidden in ("diagnosis", "probability", "risk_score", "is_abnormal", "severity", "trend"):
            assert not hasattr(obj, forbidden)


# ---------------------------------------------------------------------
# Deterministic output overall
# ---------------------------------------------------------------------

def test_same_record_produces_identical_timeline_every_time():
    record = _record()
    record.add_observation(_observation(T2))
    record.add_vital_sign(_temp(T1, 36.0))
    record.add_laboratory_result(_glucose(T3, 95.0))
    t1 = build_patient_timeline(record)
    t2 = build_patient_timeline(record)
    assert [e.description for e in t1.entries] == [e.description for e in t2.entries]
    assert [e.timestamp for e in t1.entries] == [e.timestamp for e in t2.entries]
    assert [e.entry_type for e in t1.entries] == [e.entry_type for e in t2.entries]


# ---------------------------------------------------------------------
# Phase 2 routing compatibility sanity
# ---------------------------------------------------------------------

def test_phase2_domain_routing_unaffected():
    from routing.domain_router import Domain, classify_domain

    assert classify_domain("hastanın nabzı kaç?") == Domain.MEDICAL
    assert classify_domain("gemi hangi bunker ikmalini aldı?") == Domain.MARITIME
