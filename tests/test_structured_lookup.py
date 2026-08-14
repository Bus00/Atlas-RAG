"""
tests/test_structured_lookup.py
----------------------------------
NOT: Bu dosya `database.db` üzerinden psycopg import ettiği için
`pip install -r requirements.txt` çalıştırılmadan (yani psycopg kurulu
değilken) import hatası verir. DB bağlantısı GEREKMEZ (get_cursor hiç
çağrılmıyor), sadece psycopg paketinin kurulu olması yeterli.
"""
from __future__ import annotations

from datetime import date

from retrieval.structured_lookup import _extract_date, looks_like_structured_query


def test_extract_iso_date():
    assert _extract_date("2021-05-14 tarihinde ne olmuştu?") == date(2021, 5, 14)


def test_extract_dotted_date():
    assert _extract_date("14.05.2021 tarihinde yakıt tüketimi ne kadardı?") == date(2021, 5, 14)


def test_extract_date_returns_none_when_missing():
    assert _extract_date("Genel olarak bakım geçmişi nasıl?") is None


def test_looks_like_structured_query_true_for_date_plus_keyword():
    assert looks_like_structured_query("2021-05-14 tarihinde yakıt tüketimi ne kadardı?") is True


def test_looks_like_structured_query_false_without_date():
    assert looks_like_structured_query("Yakıt tüketimi genel olarak nasıldı?") is False


def test_looks_like_structured_query_false_without_keyword():
    assert looks_like_structured_query("2021-05-14 tarihinde ne oldu?") is False
