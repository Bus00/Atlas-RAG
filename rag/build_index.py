"""
rag/build_index.py
---------------------
İki aşamalı çalışır:

  1) noon_report_event tablosundaki, henüz document_chunk karşılığı
     olmayan satırlar için document_chunk kaydı oluşturur (chunk_type='event').
     (monthly_summary chunk'ları zaten ingestion/load_noon_reports.py
     aşamasında yazılmıştı.)

  2) document_chunk tablosunda embedded_at IS NULL olan tüm satırları
     embed eder, ChromaDB'ye yazar (upsert) ve Postgres'te embedded_at +
     chroma_id alanlarını günceller.

Bu ayrım (Postgres yazma / embed etme) sayesinde script tekrar tekrar
güvenle çalıştırılabilir (idempotent): zaten embed edilmiş chunk'lar
tekrar işlenmez.

Çalıştırma:
    python -m rag.build_index
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import get_cursor  # noqa: E402
from embeddings.embedder import get_default_embedder  # noqa: E402
from rag.chunker import EventForChunking, build_chunk_metadata, build_event_chunk_text  # noqa: E402
from vectorstore.chroma_store import ChromaStore  # noqa: E402


def create_missing_event_chunks() -> int:
    """noon_report_event -> document_chunk (chunk_type='event'). Idempotent."""
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            SELECT nre.id AS event_id, nre.event_type, nre.event_text, nre.equipment_hint,
                   nre.event_date, nr.id AS noon_report_id, nr.document_id,
                   v.id AS vessel_id, v.name AS vessel_name
            FROM noon_report_event nre
            JOIN noon_report nr ON nr.id = nre.noon_report_id
            JOIN vessel v ON v.id = nr.vessel_id
            LEFT JOIN document_chunk dc ON dc.noon_report_event_id = nre.id
            WHERE dc.id IS NULL
            ORDER BY nre.id
            """
        )
        pending = cur.fetchall()

        created = 0
        for row in pending:
            event = EventForChunking(
                vessel_name=row["vessel_name"],
                report_date=row["event_date"],
                event_type=row["event_type"],
                equipment_hint=row["equipment_hint"],
                event_text=row["event_text"],
                noon_report_id=row["noon_report_id"],
                noon_report_event_id=row["event_id"],
                vessel_id=row["vessel_id"],
            )
            chunk_text = build_event_chunk_text(event)

            document_id = row["document_id"]
            if document_id is None:
                # noon_report bir document'e bağlı değilse (olmamalı ama güvenlik için) atla
                continue

            cur.execute(
                "SELECT COALESCE(MAX(chunk_index), -1) + 1 AS next_idx FROM document_chunk WHERE document_id = %s",
                (document_id,),
            )
            next_idx = cur.fetchone()["next_idx"]

            cur.execute(
                """
                INSERT INTO document_chunk (
                    document_id, noon_report_id, noon_report_event_id,
                    vessel_id, report_date, chunk_index, chunk_text, chunk_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'event')
                ON CONFLICT (document_id, chunk_index) DO NOTHING
                """,
                (
                    document_id, row["noon_report_id"], row["event_id"],
                    row["vessel_id"], row["event_date"], next_idx, chunk_text,
                ),
            )
            created += 1
        print(f"[OK] {created} yeni event chunk'ı document_chunk tablosuna yazıldı.")
        return created


def embed_pending_chunks(batch_size: int = 32) -> int:
    """embedded_at IS NULL olan chunk'ları embed edip ChromaDB'ye yazar."""
    embedder = get_default_embedder()
    store = ChromaStore()

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT dc.id, dc.chunk_text, dc.chunk_type, dc.report_date,
                   dc.vessel_id, dc.noon_report_id, dc.noon_report_event_id,
                   v.name AS vessel_name,
                   nre.event_type, nre.equipment_hint
            FROM document_chunk dc
            JOIN vessel v ON v.id = dc.vessel_id
            LEFT JOIN noon_report_event nre ON nre.id = dc.noon_report_event_id
            WHERE dc.embedded_at IS NULL
            ORDER BY dc.id
            """
        )
        pending = cur.fetchall()

    if not pending:
        print("[OK] Embed edilecek yeni chunk yok (hepsi güncel).")
        return 0

    print(f"[embed] {len(pending)} chunk embed edilecek...")
    total_embedded = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        texts = [r["chunk_text"] for r in batch]
        vectors = embedder.embed_texts(texts)

        ids = [f"chunk-{r['id']}" for r in batch]
        metadatas = []
        for r in batch:
            meta = {
                "chunk_id": r["id"],
                "vessel": r["vessel_name"],
                "vessel_id": r["vessel_id"],
                "chunk_type": r["chunk_type"],
                "report_date": r["report_date"].isoformat() if r["report_date"] else "",
                "year": r["report_date"].year if r["report_date"] else 0,
                "month": r["report_date"].month if r["report_date"] else 0,
                "event_type": r["event_type"] or "",
                "equipment_hint": r["equipment_hint"] or "",
            }
            metadatas.append(meta)

        store.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

        with get_cursor(commit=True) as cur:
            for r, chroma_id in zip(batch, ids):
                cur.execute(
                    "UPDATE document_chunk SET embedded_at = CURRENT_TIMESTAMP, chroma_id = %s WHERE id = %s",
                    (chroma_id, r["id"]),
                )
        total_embedded += len(batch)
        print(f"  ... {total_embedded}/{len(pending)}")

    print(f"[OK] Toplam {total_embedded} chunk embed edilip ChromaDB'ye yazıldı.")
    print(f"[bilgi] ChromaDB koleksiyonundaki toplam kayıt sayısı: {store.count()}")
    return total_embedded


def main() -> None:
    create_missing_event_chunks()
    embed_pending_chunks()


if __name__ == "__main__":
    main()
