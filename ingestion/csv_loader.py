"""
ingestion/csv_loader.py
------------------------
data/raw/ altındaki CSV dosyalarını okuyup doğrulayan, saf Python
fonksiyonları. Bilinçli olarak PostgreSQL'e bağımlı DEĞİL — böylece
veritabanı olmadan da (tests/) doğrulanabilir.

Bu modül "ingestion"ın PARSING kısmıdır. DB'ye YAZMA işi
load_noon_reports.py içinde ayrı tutulur (single responsibility).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

VALID_EVENT_TYPES = {"maintenance", "failure", "bunker", "voyage", "other"}


@dataclass
class NoonReportRow:
    report_date: date
    fw_rob: Optional[float]
    bbm_rob: Optional[float]
    me_oil_rob: Optional[float]
    ae_oil_rob: Optional[float]
    fw_consumption: Optional[float]
    bbm_consumption: Optional[float]
    me_oil_consumption: Optional[float]
    ae_oil_consumption: Optional[float]


@dataclass
class EventRow:
    report_date: date
    event_type: str
    equipment_hint: Optional[str]
    event_text: str
    source_image: Optional[str]


@dataclass
class MonthlyFwSummaryRow:
    period_label: str
    period_start: date
    period_end: date
    total_fw_tons: float
    avg_fw_tons_per_day: float
    source_image: Optional[str]


class CsvValidationError(ValueError):
    """CSV içeriğinde beklenmeyen/geçersiz bir değer bulunduğunda fırlatılır."""


def _to_float(value: str, *, field_name: str, row_num: int) -> Optional[float]:
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise CsvValidationError(
            f"Satır {row_num}: '{field_name}' alanı sayıya çevrilemedi: {value!r}"
        ) from exc


def _to_date(value: str, *, row_num: int) -> date:
    value = (value or "").strip()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CsvValidationError(
            f"Satır {row_num}: geçersiz tarih formatı (YYYY-MM-DD bekleniyor): {value!r}"
        ) from exc


def load_noon_report_csv(path: str | Path) -> list[NoonReportRow]:
    rows: list[NoonReportRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader, start=2):  # 1. satır header
            rows.append(
                NoonReportRow(
                    report_date=_to_date(raw["report_date"], row_num=i),
                    fw_rob=_to_float(raw["fw_rob"], field_name="fw_rob", row_num=i),
                    bbm_rob=_to_float(raw["bbm_rob"], field_name="bbm_rob", row_num=i),
                    me_oil_rob=_to_float(raw["me_oil_rob"], field_name="me_oil_rob", row_num=i),
                    ae_oil_rob=_to_float(raw["ae_oil_rob"], field_name="ae_oil_rob", row_num=i),
                    fw_consumption=_to_float(raw["fw_consumption"], field_name="fw_consumption", row_num=i),
                    bbm_consumption=_to_float(raw["bbm_consumption"], field_name="bbm_consumption", row_num=i),
                    me_oil_consumption=_to_float(raw["me_oil_consumption"], field_name="me_oil_consumption", row_num=i),
                    ae_oil_consumption=_to_float(raw["ae_oil_consumption"], field_name="ae_oil_consumption", row_num=i),
                )
            )
    _check_duplicate_dates(rows, path)
    return rows


def _check_duplicate_dates(rows: list[NoonReportRow], path: str | Path) -> None:
    seen: set[date] = set()
    for r in rows:
        if r.report_date in seen:
            raise CsvValidationError(f"{path}: '{r.report_date}' tarihi birden fazla kez geçiyor.")
        seen.add(r.report_date)


def load_events_csv(path: str | Path) -> list[EventRow]:
    rows: list[EventRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader, start=2):
            event_type = (raw["event_type"] or "").strip()
            if event_type not in VALID_EVENT_TYPES:
                raise CsvValidationError(
                    f"Satır {i}: geçersiz event_type {event_type!r}. "
                    f"Geçerli değerler: {sorted(VALID_EVENT_TYPES)}"
                )
            event_text = (raw["event_text"] or "").strip()
            if not event_text:
                raise CsvValidationError(f"Satır {i}: event_text boş olamaz.")
            rows.append(
                EventRow(
                    report_date=_to_date(raw["report_date"], row_num=i),
                    event_type=event_type,
                    equipment_hint=(raw.get("equipment_hint") or "").strip() or None,
                    event_text=event_text,
                    source_image=(raw.get("source_image") or "").strip() or None,
                )
            )
    return rows


def load_monthly_fw_summary_csv(path: str | Path) -> list[MonthlyFwSummaryRow]:
    rows: list[MonthlyFwSummaryRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader, start=2):
            rows.append(
                MonthlyFwSummaryRow(
                    period_label=raw["period_label"].strip(),
                    period_start=_to_date(raw["period_start"], row_num=i),
                    period_end=_to_date(raw["period_end"], row_num=i),
                    total_fw_tons=_to_float(raw["total_fw_tons"], field_name="total_fw_tons", row_num=i) or 0.0,
                    avg_fw_tons_per_day=_to_float(
                        raw["avg_fw_tons_per_day"], field_name="avg_fw_tons_per_day", row_num=i
                    )
                    or 0.0,
                    source_image=(raw.get("source_image") or "").strip() or None,
                )
            )
    return rows
