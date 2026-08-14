"""
ingestion/load_noon_reports.py
--------------------------------
data/raw/ altındaki 3 CSV dosyasını okuyup PostgreSQL'e yazar:
  - mt_ivani_noon_reports.csv        -> noon_report tablosu
  - mt_ivani_events.csv              -> noon_report_event tablosu
  - mt_ivani_monthly_fw_summary.csv  -> document_chunk tablosuna
                                         "monthly_summary" tipi chunk olarak

Bu script YALNIZCA yapılandırılmış veriyi Postgres'e yazar; embedding/
ChromaDB işi rag/build_index.py içindedir (ayrı adım — RAG pipeline
bölümündeki akışla tutarlı).

Çalıştırma:
    python -m ingestion.load_noon_reports
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import get_cursor  # noqa: E402
from ingestion.csv_loader import (  # noqa: E402
    load_events_csv,
    load_monthly_fw_summary_csv,
    load_noon_report_csv,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# event_text içindeki anahtar kelimelere göre basit kural tabanlı eşleme.
# V1 için bilinçli olarak basit tutuldu — gelişmiş NLP sınıflandırma V2 konusu.
MAINTENANCE_KEYWORDS = {
    "Oil Change": ["yağ", "oli", "yag"],
    "Repair": ["onar", "repair"],
    "Supply / Replenishment": ["ikmal", "supply"],
    "Inspection / Test": ["test", "denetim", "vetting"],
}
FAILURE_KEYWORDS = {
    "Grounding": ["kandas", "karaya", "grounding"],
    "Hull Damage": ["gövde hasar", "hull damage"],
    "Steering / Maneuvering Fault": ["dümen", "manevra", "steering"],
}
EQUIPMENT_CATEGORY_KEYWORDS = {
    "Main Engine": ["me ", "main engine", "ana makine"],
    "Auxiliary Engine": ["ae ", "auxiliary", "yardımcı makine"],
    "Fresh Water System": ["fresh water", "fw", "taze su"],
    "Hull & Steering": ["gövde", "dümen", "hull", "steering"],
    "Cargo System": ["cargo", "yük", "tahliye", "discharge"],
}


def _match_keyword(text: str, mapping: dict[str, list[str]]) -> str | None:
    lowered = text.lower()
    for label, keywords in mapping.items():
        for kw in keywords:
            if kw.lower() in lowered:
                return label
    return None


def _get_or_create_equipment(cur, vessel_id: int, equipment_hint: str | None) -> int | None:
    if not equipment_hint or equipment_hint == "-":
        return None

    cur.execute(
        "SELECT id FROM equipment WHERE vessel_id = %s AND name = %s",
        (vessel_id, equipment_hint),
    )
    row = cur.fetchone()
    if row:
        return row["id"]

    category_name = _match_keyword(equipment_hint, EQUIPMENT_CATEGORY_KEYWORDS) or "Cargo System"
    cur.execute("SELECT id FROM equipment_category WHERE name = %s", (category_name,))
    cat_row = cur.fetchone()
    if not cat_row:
        return None
    category_id = cat_row["id"]

    cur.execute(
        """
        INSERT INTO equipment (vessel_id, equipment_category_id, name)
        VALUES (%s, %s, %s)
        ON CONFLICT (vessel_id, name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        (vessel_id, category_id, equipment_hint),
    )
    return cur.fetchone()["id"]


def load_noon_reports() -> None:
    rows = load_noon_report_csv(DATA_DIR / "mt_ivani_noon_reports.csv")
    print(f"[ingestion] {len(rows)} noon_report satırı okundu (mt_ivani_noon_reports.csv).")

    with get_cursor(commit=True) as cur:
        cur.execute("SELECT id FROM vessel WHERE name = %s", ("MT IVANI",))
        vessel_row = cur.fetchone()
        if not vessel_row:
            raise RuntimeError(
                "vessel tablosunda 'MT IVANI' bulunamadı. Önce migration'ları "
                "(03_seed_reference_data.sql dahil) çalıştırdığınızdan emin olun."
            )
        vessel_id = vessel_row["id"]

        inserted = 0
        for r in rows:
            cur.execute(
                """
                INSERT INTO noon_report (
                    vessel_id, report_date,
                    fw_rob, bbm_rob, me_oil_rob, ae_oil_rob,
                    fw_consumption, bbm_consumption, me_oil_consumption, ae_oil_consumption,
                    is_test_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
                ON CONFLICT (vessel_id, report_date) DO UPDATE SET
                    fw_rob = EXCLUDED.fw_rob,
                    bbm_rob = EXCLUDED.bbm_rob,
                    me_oil_rob = EXCLUDED.me_oil_rob,
                    ae_oil_rob = EXCLUDED.ae_oil_rob,
                    fw_consumption = EXCLUDED.fw_consumption,
                    bbm_consumption = EXCLUDED.bbm_consumption,
                    me_oil_consumption = EXCLUDED.me_oil_consumption,
                    ae_oil_consumption = EXCLUDED.ae_oil_consumption
                """,
                (
                    vessel_id, r.report_date,
                    r.fw_rob, r.bbm_rob, r.me_oil_rob, r.ae_oil_rob,
                    r.fw_consumption, r.bbm_consumption, r.me_oil_consumption, r.ae_oil_consumption,
                ),
            )
            inserted += 1
        print(f"[OK] {inserted} noon_report kaydı yazıldı/güncellendi.")


def load_events() -> None:
    rows = load_events_csv(DATA_DIR / "mt_ivani_events.csv")
    print(f"[ingestion] {len(rows)} event satırı okundu (mt_ivani_events.csv).")

    with get_cursor(commit=True) as cur:
        cur.execute("SELECT id FROM vessel WHERE name = %s", ("MT IVANI",))
        vessel_id = cur.fetchone()["id"]

        inserted, skipped = 0, 0
        for r in rows:
            cur.execute(
                "SELECT id FROM noon_report WHERE vessel_id = %s AND report_date = %s",
                (vessel_id, r.report_date),
            )
            nr_row = cur.fetchone()
            if not nr_row:
                print(f"  [ATLA] {r.report_date} için noon_report bulunamadı, event atlandı.")
                skipped += 1
                continue
            noon_report_id = nr_row["id"]

            equipment_id = _get_or_create_equipment(cur, vessel_id, r.equipment_hint)

            maintenance_type_id = None
            failure_mode_id = None
            if r.event_type == "maintenance":
                mt_name = _match_keyword(r.event_text, MAINTENANCE_KEYWORDS)
                if mt_name:
                    cur.execute("SELECT id FROM maintenance_type WHERE name = %s", (mt_name,))
                    row = cur.fetchone()
                    maintenance_type_id = row["id"] if row else None
            elif r.event_type == "failure":
                fm_name = _match_keyword(r.event_text, FAILURE_KEYWORDS)
                if fm_name:
                    cur.execute("SELECT id FROM failure_mode WHERE name = %s", (fm_name,))
                    row = cur.fetchone()
                    failure_mode_id = row["id"] if row else None

            cur.execute(
                """
                INSERT INTO noon_report_event (
                    noon_report_id, equipment_id, maintenance_type_id, failure_mode_id,
                    event_type, event_date, event_text, equipment_hint, is_test_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE)
                """,
                (
                    noon_report_id, equipment_id, maintenance_type_id, failure_mode_id,
                    r.event_type, r.report_date, r.event_text, r.equipment_hint,
                ),
            )
            inserted += 1
        print(f"[OK] {inserted} noon_report_event kaydı yazıldı ({skipped} atlandı).")


def load_monthly_summaries_as_chunks() -> None:
    """
    Aylık FW özet kutularını doğrudan document_chunk'a "monthly_summary" tipi
    olarak yazar (bunlar günlük noon_report'a değil, dokümanın tamamına ait).
    """
    rows = load_monthly_fw_summary_csv(DATA_DIR / "mt_ivani_monthly_fw_summary.csv")
    print(f"[ingestion] {len(rows)} aylık özet satırı okundu (mt_ivani_monthly_fw_summary.csv).")

    with get_cursor(commit=True) as cur:
        cur.execute("SELECT id FROM vessel WHERE name = %s", ("MT IVANI",))
        vessel_id = cur.fetchone()["id"]

        inserted = 0
        for r in rows:
            cur.execute(
                "SELECT id FROM document WHERE vessel_id = %s AND source_path = %s",
                (vessel_id, r.source_image),
            )
            doc_row = cur.fetchone()
            if not doc_row:
                print(f"  [ATLA] {r.source_image} için document bulunamadı.")
                continue
            document_id = doc_row["id"]

            chunk_text = (
                f"MT IVANI - {r.period_label} dönemi Fresh Water (taze su) tüketim özeti: "
                f"Toplam tüketim {r.total_fw_tons} ton, günlük ortalama tüketim "
                f"{r.avg_fw_tons_per_day:.2f} ton ({r.period_start} - {r.period_end})."
            )

            cur.execute(
                "SELECT COALESCE(MAX(chunk_index), -1) + 1 AS next_idx FROM document_chunk WHERE document_id = %s",
                (document_id,),
            )
            next_idx = cur.fetchone()["next_idx"]

            cur.execute(
                """
                INSERT INTO document_chunk (
                    document_id, vessel_id, report_date,
                    chunk_index, chunk_text, chunk_type
                ) VALUES (%s, %s, %s, %s, %s, 'monthly_summary')
                ON CONFLICT (document_id, chunk_index) DO NOTHING
                """,
                (document_id, vessel_id, r.period_start, next_idx, chunk_text),
            )
            inserted += 1
        print(f"[OK] {inserted} aylık özet chunk'ı yazıldı.")


def main() -> None:
    load_noon_reports()
    load_events()
    load_monthly_summaries_as_chunks()
    print("\n[TAMAM] Ingestion (Postgres yazma aşaması) tamamlandı.")
    print("Sonraki adım: python -m rag.build_index  (embedding + ChromaDB)")


if __name__ == "__main__":
    main()
