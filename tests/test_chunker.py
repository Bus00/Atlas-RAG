"""
tests/test_chunker.py
------------------------
rag/chunker.py içindeki saf fonksiyonların testleri. Hiçbir dış servise
bağımlı değil.
"""
from __future__ import annotations

from datetime import date

from rag.chunker import EventForChunking, build_chunk_metadata, build_event_chunk_text


def _sample_event(**overrides) -> EventForChunking:
    base = dict(
        vessel_name="MT IVANI",
        report_date=date(2021, 1, 6),
        event_type="maintenance",
        equipment_hint="AE (Auxiliary Engine) 2",
        event_text="Pergantian Oli AE 2 = 56 Liter (AE 2 yağ değişimi)",
        noon_report_id=1,
        noon_report_event_id=1,
        vessel_id=1,
    )
    base.update(overrides)
    return EventForChunking(**base)


def test_chunk_text_contains_vessel_date_and_event_text():
    event = _sample_event()
    text = build_event_chunk_text(event)
    assert "MT IVANI" in text
    assert "2021-01-06" in text
    assert "Pergantian Oli AE 2" in text


def test_chunk_text_translates_event_type_to_turkish():
    event = _sample_event(event_type="failure")
    text = build_event_chunk_text(event)
    assert "arıza" in text


def test_chunk_text_omits_equipment_part_when_none():
    event = _sample_event(equipment_hint=None)
    text = build_event_chunk_text(event)
    assert "ekipman:" not in text


def test_chunk_metadata_has_expected_keys():
    event = _sample_event()
    meta = build_chunk_metadata(event)
    assert meta["vessel"] == "MT IVANI"
    assert meta["year"] == 2021
    assert meta["month"] == 1
    assert meta["event_type"] == "maintenance"
    assert meta["chunk_type"] == "event"
