"""
tests/test_resource_availability.py
------------------------------------------
Faz (Next Best Measurement & Resource-Aware Decision Support), Bölüm B:
medical/resource_availability.py testleri.

Bu testler yalnızca YAPISAL kaynak durumu temsilini kontrol eder --
kayıtlı olmayan bir kaynağın HER ZAMAN UNKNOWN döndüğünü, hiçbir kaynağın
var olduğunun VARSAYILMADIĞINI ve bu modülün hiçbir tıbbi/ilaç içeriği
taşımadığını doğrular.
"""
from __future__ import annotations

import pytest

from medical.models.errors import MedicalDataValidationError
from medical.resource_availability import (
    Resource,
    ResourceCategory,
    ResourceRegistry,
    ResourceState,
    build_resource_registry,
)


def _resource(resource_id: str = "device:temperature", state: ResourceState = ResourceState.AVAILABLE,
              category: ResourceCategory = ResourceCategory.MEASUREMENT_DEVICE) -> Resource:
    return Resource(resource_id=resource_id, category=category, state=state)


# ---------------------------------------------------------------------
# 9. Resource creation
# ---------------------------------------------------------------------

def test_resource_creation_minimal():
    resource = _resource()
    assert resource.resource_id == "device:temperature"
    assert resource.category == ResourceCategory.MEASUREMENT_DEVICE
    assert resource.state == ResourceState.AVAILABLE
    assert resource.note is None


def test_resource_creation_with_note():
    resource = Resource(
        resource_id="laboratory_capability", category=ResourceCategory.LABORATORY_CAPABILITY,
        state=ResourceState.UNAVAILABLE, note="son envanter kontrolünde bildirildi",
    )
    assert resource.note == "son envanter kontrolünde bildirildi"


def test_resource_rejects_empty_id():
    with pytest.raises(MedicalDataValidationError):
        Resource(resource_id="", category=ResourceCategory.OTHER, state=ResourceState.UNKNOWN)


def test_resource_rejects_invalid_category():
    with pytest.raises(MedicalDataValidationError):
        Resource(resource_id="x", category="measurement_device", state=ResourceState.AVAILABLE)  # type: ignore[arg-type]


def test_resource_rejects_invalid_state():
    with pytest.raises(MedicalDataValidationError):
        Resource(resource_id="x", category=ResourceCategory.OTHER, state="available")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# 10/11/12. Resource availability / unavailable / unknown state
# ---------------------------------------------------------------------

def test_registered_available_resource_reports_available():
    registry = build_resource_registry([_resource(state=ResourceState.AVAILABLE)])
    assert registry.get_state("device:temperature") == ResourceState.AVAILABLE


def test_registered_unavailable_resource_reports_unavailable():
    registry = build_resource_registry([_resource(state=ResourceState.UNAVAILABLE)])
    assert registry.get_state("device:temperature") == ResourceState.UNAVAILABLE


def test_registered_unknown_state_resource_reports_unknown():
    registry = build_resource_registry([_resource(state=ResourceState.UNKNOWN)])
    assert registry.get_state("device:temperature") == ResourceState.UNKNOWN


def test_unregistered_resource_always_reports_unknown():
    """KRİTİK: kayıtlı olmayan bir kaynak İÇİN VAR OLDUĞU YA DA OLMADIĞI ASLA VARSAYILMAZ."""
    registry = build_resource_registry([])
    assert registry.get_state("device:never_registered") == ResourceState.UNKNOWN


def test_unregistered_resource_is_not_treated_as_unavailable():
    """'Bilinmiyor', ASLA 'mevcut değil' olarak yorumlanmamalı."""
    registry = build_resource_registry([])
    state = registry.get_state("device:never_registered")
    assert state != ResourceState.UNAVAILABLE
    assert state != ResourceState.AVAILABLE
    assert state == ResourceState.UNKNOWN


# ---------------------------------------------------------------------
# Registry behavior: registration, lookup, deterministic listing
# ---------------------------------------------------------------------

def test_registry_register_adds_resource():
    registry = ResourceRegistry()
    registry.register(_resource("device:heart_rate"))
    assert registry.get("device:heart_rate") is not None
    assert registry.get("device:heart_rate").state == ResourceState.AVAILABLE


def test_registry_register_overwrites_existing_resource():
    registry = ResourceRegistry()
    registry.register(_resource("device:heart_rate", state=ResourceState.UNKNOWN))
    registry.register(_resource("device:heart_rate", state=ResourceState.AVAILABLE))
    assert registry.get_state("device:heart_rate") == ResourceState.AVAILABLE


def test_registry_get_returns_none_for_unregistered_resource():
    registry = build_resource_registry([])
    assert registry.get("device:nonexistent") is None


def test_registry_rejects_non_resource_registration():
    registry = ResourceRegistry()
    with pytest.raises(MedicalDataValidationError):
        registry.register("not a resource")  # type: ignore[arg-type]


def test_known_resource_ids_sorted_deterministically():
    registry = build_resource_registry([
        _resource("device:temperature"), _resource("device:spo2"), _resource("laboratory_capability"),
    ])
    assert registry.known_resource_ids() == ["device:spo2", "device:temperature", "laboratory_capability"]


def test_build_resource_registry_rejects_non_list():
    with pytest.raises(MedicalDataValidationError):
        build_resource_registry("not a list")  # type: ignore[arg-type]


def test_resource_registry_constructor_validates_key_matches_resource_id():
    with pytest.raises(MedicalDataValidationError):
        ResourceRegistry(resources={"wrong_key": _resource("device:temperature")})


# ---------------------------------------------------------------------
# No medication / clinical fabrication in this module
# ---------------------------------------------------------------------

def test_no_medication_attributes_on_resource_or_registry():
    resource = _resource()
    registry = build_resource_registry([resource])
    for obj in (resource, registry):
        for forbidden in ("medication", "dosage", "treatment", "diagnosis", "probability"):
            assert not hasattr(obj, forbidden)


def test_resource_categories_are_purely_structural():
    """Kategori değerleri hiçbir spesifik marka/model/gerçek envanter İÇERMEZ."""
    expected = {"measurement_device", "laboratory_capability", "internet_connectivity",
                "communication_capability", "trained_personnel", "other"}
    assert {c.value for c in ResourceCategory} == expected
