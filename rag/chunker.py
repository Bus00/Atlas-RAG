"""
rag/chunker.py
----------------
noon_report_event satırlarından embedding'e girecek "chunk" metnini üretir.

Neden ayrı bir fonksiyon: Bu mantığı DB kodundan ayırmak, embedding'e
giden metnin formatını DB veya ChromaDB olmadan test edebilmemizi sağlar
(bkz. tests/test_chunker.py).

V1'de her event = 1 chunk (basit ve öngörülebilir). Noon Report metinleri
zaten kısa olduğundan (~1-2 cümle) daha karmaşık bir chunk'lama (örn. sliding
window) V1 için gereksiz karmaşıklık olurdu.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class EventForChunking:
    vessel_name: str
    report_date: date
    event_type: str  # maintenance | failure | bunker | voyage | other
    equipment_hint: str | None
    event_text: str

    # metadata için, chunk metnine değil ChromaDB metadata'sına gider:
    noon_report_id: int
    noon_report_event_id: int
    vessel_id: int
    source_document: str | None = None


EVENT_TYPE_LABELS_TR = {
    "maintenance": "bakım",
    "failure": "arıza",
    "bunker": "ikmal/bunker",
    "voyage": "sefer",
    "other": "diğer",
}


def build_event_chunk_text(event: EventForChunking) -> str:
    """
    Örnek çıktı:
    "MT IVANI - 2021-01-06 tarihli Noon Report kaydı (olay türü: bakım,
     ekipman: AE (Auxiliary Engine) 2): Pergantian Oli AE 2 = 56 Liter
     (AE 2 yağ değişimi)."
    """
    type_label = EVENT_TYPE_LABELS_TR.get(event.event_type, event.event_type)
    equipment_part = f", ekipman: {event.equipment_hint}" if event.equipment_hint else ""
    return (
        f"{event.vessel_name} - {event.report_date.isoformat()} tarihli Noon Report "
        f"kaydı (olay türü: {type_label}{equipment_part}): {event.event_text}"
    )


def build_chunk_metadata(event: EventForChunking) -> dict:
    """ChromaDB'ye yazılacak metadata sözlüğü. Retrieval sırasında filtreleme için kullanılır."""
    return {
        "vessel": event.vessel_name,
        "vessel_id": event.vessel_id,
        "report_date": event.report_date.isoformat(),
        "year": event.report_date.year,
        "month": event.report_date.month,
        "event_type": event.event_type,
        "equipment_hint": event.equipment_hint or "",
        "noon_report_id": event.noon_report_id,
        "noon_report_event_id": event.noon_report_event_id,
        "chunk_type": "event",
    }
