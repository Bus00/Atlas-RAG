"""
tests/test_medical_inventory.py
--------------------------------------
medical/medical_inventory.py testleri.

Bu testler yalnızca YAPISAL envanter takibini (ilaç/ekipman kaydı, stok,
son kullanma tarihi, ekipman durumu) ve deterministik tarih işlemeyi
kontrol eder -- hiçbir klinik karar (ilaç kullanım önerisi, doz, tedavi)
test edilmez, çünkü bu katman bunları üretmez (bkz.
medical/medical_inventory.py docstring'i).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from medical.medical_inventory import (
    EquipmentItem,
    EquipmentStatus,
    MedicalInventory,
    MedicationItem,
    build_medical_inventory,
    find_equipment_by_status,
    find_expired_medications,
    find_expiring_soon_medications,
    find_zero_stock_medications,
)
from medical.models.errors import MedicalDataValidationError

D_2026_01_01 = datetime(2026, 1, 1)
D_2026_06_01 = datetime(2026, 6, 1)
D_2026_12_01 = datetime(2026, 12, 1)
D_2027_01_01 = datetime(2027, 1, 1)


def _medication(inventory_id: str = "med-1", name: str = "Paracetamol", quantity: float = 10,
                 unit: str = "tablet", expiration_date: datetime = D_2027_01_01,
                 status=None, note=None) -> MedicationItem:
    return MedicationItem(
        inventory_id=inventory_id, name=name, quantity=quantity, unit=unit,
        expiration_date=expiration_date, status=status, note=note,
    )


def _equipment(inventory_id: str = "eq-1", name: str = "Defibrillator", quantity: float = 1,
                unit: str = "unit", status: EquipmentStatus = EquipmentStatus.AVAILABLE,
                maintenance_date=None, note=None) -> EquipmentItem:
    return EquipmentItem(
        inventory_id=inventory_id, name=name, quantity=quantity, unit=unit,
        status=status, maintenance_date=maintenance_date, note=note,
    )


# ---------------------------------------------------------------------
# Medication creation / validation
# ---------------------------------------------------------------------

def test_medication_creation_minimal():
    med = _medication()
    assert med.inventory_id == "med-1"
    assert med.name == "Paracetamol"
    assert med.quantity == 10
    assert med.unit == "tablet"
    assert med.expiration_date == D_2027_01_01
    assert med.status is None
    assert med.note is None


def test_medication_requires_expiration_date():
    """Son kullanma tarihi KESİNLİKLE zorunlu -- None ya da yanlış tip reddedilir."""
    with pytest.raises(MedicalDataValidationError):
        MedicationItem(inventory_id="med-x", name="X", quantity=1, unit="tablet", expiration_date=None)  # type: ignore[arg-type]
    with pytest.raises(MedicalDataValidationError):
        MedicationItem(inventory_id="med-x", name="X", quantity=1, unit="tablet", expiration_date="2026-01-01")  # type: ignore[arg-type]


def test_medication_rejects_empty_inventory_id_or_name():
    with pytest.raises(MedicalDataValidationError):
        _medication(inventory_id="")
    with pytest.raises(MedicalDataValidationError):
        _medication(name="")


def test_medication_rejects_negative_quantity():
    with pytest.raises(MedicalDataValidationError):
        _medication(quantity=-1)


def test_medication_rejects_non_numeric_quantity():
    with pytest.raises(MedicalDataValidationError):
        _medication(quantity="ten")  # type: ignore[arg-type]


def test_medication_allows_zero_quantity():
    med = _medication(quantity=0)
    assert med.quantity == 0


def test_medication_with_status_and_note():
    med = _medication(status="active", note="son envanter kontrolünde bildirildi")
    assert med.status == "active"
    assert med.note == "son envanter kontrolünde bildirildi"


# ---------------------------------------------------------------------
# Equipment creation / validation
# ---------------------------------------------------------------------

def test_equipment_creation_minimal():
    eq = _equipment()
    assert eq.inventory_id == "eq-1"
    assert eq.status == EquipmentStatus.AVAILABLE
    assert eq.maintenance_date is None


def test_equipment_does_not_require_maintenance_date():
    """Ekipman için (son kullanma tarihi kavramı yok, sadece) bakım tarihi opsiyoneldir."""
    eq = _equipment(maintenance_date=None)
    assert eq.maintenance_date is None


def test_equipment_accepts_optional_maintenance_date():
    eq = _equipment(maintenance_date=D_2026_06_01)
    assert eq.maintenance_date == D_2026_06_01


def test_equipment_rejects_invalid_status():
    with pytest.raises(MedicalDataValidationError):
        _equipment(status="available")  # type: ignore[arg-type]


def test_equipment_rejects_negative_quantity():
    with pytest.raises(MedicalDataValidationError):
        _equipment(quantity=-1)


def test_equipment_supports_documented_statuses():
    assert {s.value for s in EquipmentStatus} == {"available", "unavailable", "in_maintenance"}


# ---------------------------------------------------------------------
# Inventory registration / lookup
# ---------------------------------------------------------------------

def test_build_medical_inventory_registers_medications_and_equipment():
    inv = build_medical_inventory(medications=[_medication()], equipment=[_equipment()])
    assert inv.get_medication("med-1") is not None
    assert inv.get_equipment("eq-1") is not None


def test_empty_inventory_is_safe():
    inv = build_medical_inventory()
    assert inv.known_medication_ids() == []
    assert inv.known_equipment_ids() == []
    assert inv.get_medication("nonexistent") is None


def test_register_medication_overwrites_existing_entry():
    inv = MedicalInventory()
    inv.register_medication(_medication(quantity=10))
    inv.register_medication(_medication(quantity=5))
    assert inv.get_medication("med-1").quantity == 5


def test_known_medication_ids_sorted_deterministically():
    inv = build_medical_inventory(medications=[
        _medication(inventory_id="med-z"), _medication(inventory_id="med-a"), _medication(inventory_id="med-m"),
    ])
    assert inv.known_medication_ids() == ["med-a", "med-m", "med-z"]


def test_registry_rejects_non_item_registration():
    inv = MedicalInventory()
    with pytest.raises(MedicalDataValidationError):
        inv.register_medication("not a medication")  # type: ignore[arg-type]
    with pytest.raises(MedicalDataValidationError):
        inv.register_equipment("not equipment")  # type: ignore[arg-type]


def test_inventory_constructor_validates_key_matches_id():
    with pytest.raises(MedicalDataValidationError):
        MedicalInventory(medications={"wrong-key": _medication(inventory_id="med-1")})


def test_build_medical_inventory_rejects_non_list():
    with pytest.raises(MedicalDataValidationError):
        build_medical_inventory(medications="not a list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Expired medications (deterministic, explicit as_of)
# ---------------------------------------------------------------------

def test_find_expired_medications_with_explicit_as_of():
    inv = build_medical_inventory(medications=[
        _medication(inventory_id="med-old", expiration_date=D_2026_01_01),
        _medication(inventory_id="med-fresh", expiration_date=D_2027_01_01),
    ])
    expired = find_expired_medications(inv, as_of=D_2026_06_01)
    assert [m.inventory_id for m in expired] == ["med-old"]


def test_find_expired_medications_requires_as_of_datetime():
    inv = build_medical_inventory(medications=[_medication()])
    with pytest.raises(MedicalDataValidationError):
        find_expired_medications(inv, as_of="2026-06-01")  # type: ignore[arg-type]


def test_no_expired_medications_when_as_of_is_before_expiration():
    inv = build_medical_inventory(medications=[_medication(expiration_date=D_2027_01_01)])
    assert find_expired_medications(inv, as_of=D_2026_01_01) == []


def test_expired_medications_deterministic_ordering():
    inv = build_medical_inventory(medications=[
        _medication(inventory_id="med-z", expiration_date=D_2026_01_01),
        _medication(inventory_id="med-a", expiration_date=D_2026_01_01),
    ])
    expired = find_expired_medications(inv, as_of=D_2026_06_01)
    assert [m.inventory_id for m in expired] == ["med-a", "med-z"]


# ---------------------------------------------------------------------
# Expiring-soon medications (explicit as_of + explicit window)
# ---------------------------------------------------------------------

def test_find_expiring_soon_medications_within_explicit_window():
    inv = build_medical_inventory(medications=[
        _medication(inventory_id="med-soon", expiration_date=D_2026_12_01 + timedelta(days=10)),
        _medication(inventory_id="med-far", expiration_date=D_2026_12_01 + timedelta(days=200)),
    ])
    soon = find_expiring_soon_medications(inv, as_of=D_2026_12_01, within=timedelta(days=30))
    assert [m.inventory_id for m in soon] == ["med-soon"]


def test_expiring_soon_excludes_already_expired():
    inv = build_medical_inventory(medications=[_medication(expiration_date=D_2026_01_01)])
    soon = find_expiring_soon_medications(inv, as_of=D_2026_06_01, within=timedelta(days=30))
    assert soon == []


def test_expiring_soon_requires_explicit_within_parameter():
    inv = build_medical_inventory(medications=[_medication()])
    with pytest.raises(MedicalDataValidationError):
        find_expiring_soon_medications(inv, as_of=D_2026_01_01, within="30 days")  # type: ignore[arg-type]


def test_expiring_soon_different_windows_give_different_results():
    """Sistem kendi başına bir eşik icat etmez -- sonuç tamamen çağıran kodun verdiği pencereye bağlıdır."""
    inv = build_medical_inventory(medications=[
        _medication(inventory_id="med-40d", expiration_date=D_2026_12_01 + timedelta(days=40)),
    ])
    narrow = find_expiring_soon_medications(inv, as_of=D_2026_12_01, within=timedelta(days=10))
    wide = find_expiring_soon_medications(inv, as_of=D_2026_12_01, within=timedelta(days=60))
    assert narrow == []
    assert [m.inventory_id for m in wide] == ["med-40d"]


# ---------------------------------------------------------------------
# Zero stock
# ---------------------------------------------------------------------

def test_find_zero_stock_medications():
    inv = build_medical_inventory(medications=[
        _medication(inventory_id="med-empty", quantity=0),
        _medication(inventory_id="med-full", quantity=20),
    ])
    zero_stock = find_zero_stock_medications(inv)
    assert [m.inventory_id for m in zero_stock] == ["med-empty"]


def test_no_zero_stock_when_none_are_empty():
    inv = build_medical_inventory(medications=[_medication(quantity=5)])
    assert find_zero_stock_medications(inv) == []


# ---------------------------------------------------------------------
# Equipment status queries
# ---------------------------------------------------------------------

def test_find_equipment_by_status():
    inv = build_medical_inventory(equipment=[
        _equipment(inventory_id="eq-avail", status=EquipmentStatus.AVAILABLE),
        _equipment(inventory_id="eq-maint", status=EquipmentStatus.IN_MAINTENANCE),
        _equipment(inventory_id="eq-unavail", status=EquipmentStatus.UNAVAILABLE),
    ])
    assert [e.inventory_id for e in find_equipment_by_status(inv, EquipmentStatus.IN_MAINTENANCE)] == ["eq-maint"]


def test_find_equipment_by_status_empty_result():
    inv = build_medical_inventory(equipment=[_equipment(status=EquipmentStatus.AVAILABLE)])
    assert find_equipment_by_status(inv, EquipmentStatus.UNAVAILABLE) == []


# ---------------------------------------------------------------------
# Deterministic behavior overall
# ---------------------------------------------------------------------

def test_queries_deterministic_across_repeated_calls():
    inv = build_medical_inventory(
        medications=[
            _medication(inventory_id="med-old", expiration_date=D_2026_01_01, quantity=0),
            _medication(inventory_id="med-soon", expiration_date=D_2026_12_01 + timedelta(days=5)),
        ],
        equipment=[_equipment(status=EquipmentStatus.IN_MAINTENANCE)],
    )
    results = []
    for _ in range(5):
        results.append((
            tuple(m.inventory_id for m in find_expired_medications(inv, as_of=D_2026_06_01)),
            tuple(m.inventory_id for m in find_expiring_soon_medications(inv, as_of=D_2026_12_01, within=timedelta(days=30))),
            tuple(m.inventory_id for m in find_zero_stock_medications(inv)),
            tuple(e.inventory_id for e in find_equipment_by_status(inv, EquipmentStatus.IN_MAINTENANCE)),
        ))
    assert len(set(results)) == 1


def test_system_never_assumes_current_date():
    """Sistemin kendi başına 'bugün' ürettiğine dair hiçbir kanıt/attribute yok -- kaynak taranır."""
    import medical.medical_inventory as module
    source = open(module.__file__, encoding="utf-8").read()
    assert "datetime.now" not in source
    assert "date.today" not in source


# ---------------------------------------------------------------------
# No clinical / medication-usage decision content
# ---------------------------------------------------------------------

def test_no_clinical_decision_attributes_on_medication_or_inventory():
    med = _medication()
    inv = build_medical_inventory(medications=[med])
    for obj in (med, inv):
        for forbidden in (
            "dosage", "dose", "treatment_duration", "treatment_plan", "use_recommendation",
            "administer_to", "diagnosis", "disease", "probability",
        ):
            assert not hasattr(obj, forbidden)


def test_equipment_has_no_clinical_decision_attributes():
    eq = _equipment()
    for forbidden in ("diagnosis", "probability", "treatment_plan", "dosage"):
        assert not hasattr(eq, forbidden)


# ---------------------------------------------------------------------
# Phase 2 routing compatibility sanity
# ---------------------------------------------------------------------

def test_phase2_domain_routing_unaffected():
    from routing.domain_router import Domain, classify_domain

    assert classify_domain("hastanın nabzı kaç?") == Domain.MEDICAL
    assert classify_domain("gemi hangi bunker ikmalini aldı?") == Domain.MARITIME
