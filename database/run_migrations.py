"""
database/run_migrations.py
---------------------------
database/migrations/ klasöründeki .sql dosyalarını isim sırasına göre
(01_, 02_, 03_...) çalıştırır.

Çalıştırma:
    python -m database.run_migrations
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import check_connection, run_migration_file  # noqa: E402

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def main() -> None:
    print("PostgreSQL bağlantısı kontrol ediliyor...")
    if not check_connection():
        print(
            "Bağlantı kurulamadı. .env dosyasındaki PG_HOST/PG_PORT/PG_DB/"
            "PG_USER/PG_PASSWORD değerlerini ve PostgreSQL'in çalıştığını kontrol edin."
        )
        sys.exit(1)
    print("[OK] Bağlantı başarılı.\n")

    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        print(f"'{MIGRATIONS_DIR}' içinde .sql dosyası bulunamadı.")
        sys.exit(1)

    for f in files:
        run_migration_file(str(f))

    print("\nTüm migration'lar tamamlandı.")


if __name__ == "__main__":
    main()
