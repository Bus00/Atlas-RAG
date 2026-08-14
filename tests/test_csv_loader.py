"""
tests/test_csv_loader.py
--------------------------
data/raw/ altındaki gerçek CSV dosyalarının doğru parse edildiğini ve
formatlarının bozulmadığını doğrular. PostgreSQL/ChromaDB/Ollama GEREKMEZ
— bu yüzden `pip install -r requirements.txt` sonrası hiçbir servis
çalıştırmadan `pytest tests/test_csv_loader.py` ile hemen koşulabilir.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ingestion.csv_loader import (
    CsvValidationError,
    load_events_csv,
    load_monthly_fw_summary_csv,
    load_noon_report_csv,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_noon_report_csv_loads_and_has_expected_row_count():
    rows = load_noon_report_csv(DATA_DIR / "mt_ivani_noon_reports.csv")
    assert len(rows) > 0
    # Ocak 2021 31 gün + ... toplam 8 ay, en az 200 satır bekleniyor
    assert len(rows) >= 200


def test_noon_report_csv_first_row_matches_source_image():
    rows = load_noon_report_csv(DATA_DIR / "mt_ivani_noon_reports.csv")
    first = rows[0]
    assert first.report_date == date(2021, 1, 1)
    assert first.fw_rob == 27
    assert first.bbm_rob == 10492
    assert first.me_oil_rob == 152
    assert first.ae_oil_rob == 193


def test_noon_report_csv_no_duplicate_dates():
    rows = load_noon_report_csv(DATA_DIR / "mt_ivani_noon_reports.csv")
    dates = [r.report_date for r in rows]
    assert len(dates) == len(set(dates))


def test_events_csv_all_event_types_valid():
    rows = load_events_csv(DATA_DIR / "mt_ivani_events.csv")
    assert len(rows) > 0
    valid = {"maintenance", "failure", "bunker", "voyage", "other"}
    assert all(r.event_type in valid for r in rows)


def test_events_csv_contains_known_grounding_event():
    rows = load_events_csv(DATA_DIR / "mt_ivani_events.csv")
    grounding = [r for r in rows if r.event_type == "failure" and r.report_date == date(2021, 5, 10)]
    assert len(grounding) == 1
    assert "kandas" in grounding[0].event_text.lower()


def test_monthly_summary_csv_has_8_months():
    rows = load_monthly_fw_summary_csv(DATA_DIR / "mt_ivani_monthly_fw_summary.csv")
    assert len(rows) == 8


def test_invalid_date_raises_validation_error(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "report_date,fw_rob,bbm_rob,me_oil_rob,ae_oil_rob,"
        "fw_consumption,bbm_consumption,me_oil_consumption,ae_oil_consumption\n"
        "2021-13-40,1,2,3,4,5,6,7,8\n"
    )
    with pytest.raises(CsvValidationError):
        load_noon_report_csv(bad_csv)


def test_duplicate_dates_raise_validation_error(tmp_path):
    bad_csv = tmp_path / "dup.csv"
    bad_csv.write_text(
        "report_date,fw_rob,bbm_rob,me_oil_rob,ae_oil_rob,"
        "fw_consumption,bbm_consumption,me_oil_consumption,ae_oil_consumption\n"
        "2021-01-01,1,2,3,4,5,6,7,8\n"
        "2021-01-01,1,2,3,4,5,6,7,8\n"
    )
    with pytest.raises(CsvValidationError):
        load_noon_report_csv(bad_csv)
