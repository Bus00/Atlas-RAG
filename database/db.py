"""
database/db.py
---------------
PostgreSQL bağlantı yönetimi. Tek bir yerden bağlantı alınır, böylece
her modülün kendi connection mantığını yazmasına gerek kalmaz.

psycopg (v3) kullanıyoruz — modern, tip güvenli, context manager desteği iyi.
"""
from __future__ import annotations

import contextlib
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from config.settings import settings


def get_connection() -> psycopg.Connection:
    """Yeni bir PostgreSQL bağlantısı açar. Satırları dict olarak döner."""
    return psycopg.connect(settings.pg_dsn, row_factory=dict_row)


@contextlib.contextmanager
def get_cursor(commit: bool = False) -> Iterator[psycopg.Cursor]:
    """
    Kullanım:
        with get_cursor(commit=True) as cur:
            cur.execute("INSERT INTO ...", (...))

    commit=True verilirse işlem sonunda otomatik commit edilir, hata olursa
    rollback yapılır. Sadece okuma yapan sorgularda commit=False (varsayılan)
    yeterlidir.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_connection() -> bool:
    """Bağlantının çalışıp çalışmadığını test eder. README'deki 'test' adımı için."""
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1 AS ok;")
            row = cur.fetchone()
            return row is not None and row["ok"] == 1
    except Exception as exc:  # noqa: BLE001
        print(f"[DB HATASI] PostgreSQL bağlantısı kurulamadı: {exc}")
        return False


def run_migration_file(path: str) -> None:
    """Bir .sql migration dosyasını olduğu gibi çalıştırır."""
    with open(path, "r", encoding="utf-8") as f:
        sql = f.read()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print(f"[OK] Migration çalıştırıldı: {path}")
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        print(f"[HATA] Migration başarısız: {path}\n  -> {exc}")
        raise
    finally:
        conn.close()
