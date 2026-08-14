"""
retrieval/structured_lookup.py
---------------------------------
PostgreSQL = kesin/yapılandırılmış verinin kaynağı (bkz. proje talimatı
"PostgreSQL ve ChromaDB'nin görevlerini birbirine karıştırma").

Bu modül, kullanıcı sorusunda net bir TARİH ve ROB/consumption ile ilgili
bir anahtar kelime geçiyorsa, ChromaDB'ye hiç gitmeden doğrudan Postgres'ten
kesin sayıyı çeker. RAG pipeline bu sonucu, semantic search'ten gelen
chunk'ların ÜSTÜNE ek bir "kesin veri" bloğu olarak ekler.

Bilinçli olarak basit tutuldu (regex tabanlı tarih yakalama + anahtar
kelime eşleşmesi) — V1 kapsamında gelişmiş bir NLU/intent parser'a gerek yok.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from database.db import get_cursor

DATE_PATTERNS = [
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),          # 2021-05-14
    re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b"),  # 14.05.2021 / 14/05/2021
]

CONSUMPTION_KEYWORDS = ["tüketim", "tuketim", "consumption", "harcan"]
ROB_KEYWORDS = ["rob", "kalan", "remaining"]
FUEL_KEYWORDS = ["yakıt", "yakit", "bbm", "fuel"]
FW_KEYWORDS = ["taze su", "fresh water", "fw"]
OIL_KEYWORDS = ["yağ", "yag", "oil"]


MONTHS_TR = {
    "ocak": 1,
    "şubat": 2,
    "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayıs": 5,
    "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "ağustos": 8,
    "agustos": 8,
    "eylül": 9,
    "eylul": 9,
    "ekim": 10,
    "kasım": 11,
    "kasim": 11,
    "aralık": 12,
    "aralik": 12,
}

MONTHS_EN = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MONTH_KEYWORDS = {**MONTHS_TR, **MONTHS_EN}



@dataclass
class StructuredFact:
    report_date: date
    vessel: str
    fw_rob: float | None
    bbm_rob: float | None
    me_oil_rob: float | None
    ae_oil_rob: float | None
    fw_consumption: float | None
    bbm_consumption: float | None
    me_oil_consumption: float | None
    ae_oil_consumption: float | None

    def to_context_text(self) -> str:
        return (
            f"[Kesin veri - PostgreSQL noon_report kaydı] {self.vessel} - {self.report_date.isoformat()}: "
            f"ROB(kalan) => FW: {self.fw_rob} ton, BBM: {self.bbm_rob} L, ME Oil: {self.me_oil_rob} L, "
            f"AE Oil: {self.ae_oil_rob} L | Consumption(tüketim) => FW: {self.fw_consumption} ton, "
            f"BBM: {self.bbm_consumption} L, ME Oil: {self.me_oil_consumption} L, "
            f"AE Oil: {self.ae_oil_consumption} L"
        )


def _extract_date(question: str) -> date | None:
    for pattern in DATE_PATTERNS[:1]:
        m = pattern.search(question)
        if m:
            y, mo, d = m.groups()
            try:
                return date(int(y), int(mo), int(d))
            except ValueError:
                return None
    m = DATE_PATTERNS[1].search(question)
    if m:
        d, mo, y = m.groups()
        try:
            return date(int(y), int(mo), int(d))
        except ValueError:
            return None
    return None


def _extract_month(question: str) -> int | None:
    lowered = question.lower()

    for month_name, month_number in MONTH_KEYWORDS.items():
        if re.search(rf"\b{re.escape(month_name)}\b", lowered):
            return month_number

    return None



def _mentions_any(question: str, keywords: list[str]) -> bool:
    lowered = question.lower()
    return any(k in lowered for k in keywords)


def looks_like_monthly_structured_query(question: str) -> bool:
    has_month = _extract_month(question) is not None
    has_keyword = _mentions_any(
        question,
        CONSUMPTION_KEYWORDS
        + ROB_KEYWORDS
        + FUEL_KEYWORDS
        + FW_KEYWORDS
        + OIL_KEYWORDS,
    )
    return has_month and has_keyword


def looks_like_structured_query(question: str) -> bool:
    has_date = _extract_date(question) is not None
    has_keyword = _mentions_any(
        question, CONSUMPTION_KEYWORDS + ROB_KEYWORDS + FUEL_KEYWORDS + FW_KEYWORDS + OIL_KEYWORDS
    )
    return has_date and has_keyword


def lookup_monthly_consumption(
    question: str,
    vessel_name: str = "MT IVANI",
) -> dict | None:
    month = _extract_month(question)

    if month is None:
        return None

    # V1 veri setimiz 2021 yılına ait.
    year = 2021

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) AS report_count,
                COALESCE(SUM(nr.bbm_consumption), 0) AS bbm_consumption,
                COALESCE(SUM(nr.fw_consumption), 0) AS fw_consumption,
                COALESCE(SUM(nr.me_oil_consumption), 0) AS me_oil_consumption,
                COALESCE(SUM(nr.ae_oil_consumption), 0) AS ae_oil_consumption
            FROM noon_report nr
            JOIN vessel v ON v.id = nr.vessel_id
            WHERE v.name = %s
              AND EXTRACT(YEAR FROM nr.report_date) = %s
              AND EXTRACT(MONTH FROM nr.report_date) = %s
            """,
            (vessel_name, year, month),
        )

        row = cur.fetchone()

    if not row or row["report_count"] == 0:
        return None

    return {
        "year": year,
        "month": month,
        "report_count": row["report_count"],
        "bbm_consumption": row["bbm_consumption"],
        "fw_consumption": row["fw_consumption"],
        "me_oil_consumption": row["me_oil_consumption"],
        "ae_oil_consumption": row["ae_oil_consumption"],
    }


def lookup_noon_report_by_date(question: str, vessel_name: str = "MT IVANI") -> StructuredFact | None:
    report_date = _extract_date(question)
    if report_date is None:
        return None

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT nr.report_date, v.name AS vessel,
                   nr.fw_rob, nr.bbm_rob, nr.me_oil_rob, nr.ae_oil_rob,
                   nr.fw_consumption, nr.bbm_consumption, nr.me_oil_consumption, nr.ae_oil_consumption
            FROM noon_report nr
            JOIN vessel v ON v.id = nr.vessel_id
            WHERE v.name = %s AND nr.report_date = %s
            """,
            (vessel_name, report_date),
        )
        row = cur.fetchone()

    if not row:
        return None

    return StructuredFact(
        report_date=row["report_date"],
        vessel=row["vessel"],
        fw_rob=row["fw_rob"],
        bbm_rob=row["bbm_rob"],
        me_oil_rob=row["me_oil_rob"],
        ae_oil_rob=row["ae_oil_rob"],
        fw_consumption=row["fw_consumption"],
        bbm_consumption=row["bbm_consumption"],
        me_oil_consumption=row["me_oil_consumption"],
        ae_oil_consumption=row["ae_oil_consumption"],
    )
