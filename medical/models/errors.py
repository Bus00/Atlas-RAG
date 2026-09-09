"""
medical/models/errors.py
---------------------------
Medikal veri modelleri için ortak, saf (dış servise bağımlı olmayan)
validasyon hata sınıfı.

Repodaki mevcut desenle tutarlı: bkz. ingestion/csv_loader.py::CsvValidationError
-- ValueError alt sınıfı, hiçbir DB/network bağımlılığı yok, testlerde
doğrudan yakalanabilir.
"""
from __future__ import annotations


class MedicalDataValidationError(ValueError):
    """
    Medikal veri modellerinden biri (Patient, Observation, VitalSign,
    LaboratoryResult) yapısal olarak geçersiz bir değerle oluşturulmaya
    çalışıldığında fırlatılır.

    ÖNEMLİ: Bu SADECE yapısal/format doğrulamasıdır (örn. boş id, yanlış
    tip, eksik zorunlu alan). Klinik anlamda "anormal" bir değer (örn.
    ateş = 41°C) burada asla reddedilmez -- bu, ileri faz(lar)daki klinik
    değerlendirme katmanının işidir, veri modelinin değil.
    """
