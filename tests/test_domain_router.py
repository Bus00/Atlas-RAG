"""
tests/test_domain_router.py
------------------------------
routing/domain_router.py için testler. Dış servise (Postgres/Chroma/Ollama)
bağımlı değil -- tamamen saf, deterministik fonksiyon testleri.
"""
from __future__ import annotations

from routing.domain_router import Domain, classify_domain


def test_maritime_question_from_brief_example():
    assert classify_domain("MT IVANI yakıt tüketimi nasıl değişti?") == Domain.MARITIME


def test_medical_question_from_brief_example():
    assert classify_domain("Hastanın ateşi 39.1 ve nabzı 120.") == Domain.MEDICAL


def test_both_question_from_brief_example():
    assert classify_domain("Gemide hasta var ve tahliye gerekiyor.") == Domain.BOTH


def test_general_question_has_no_domain_keywords():
    assert classify_domain("Bugün hava nasıl olacak?") == Domain.GENERAL


def test_maritime_english_keywords():
    assert classify_domain("What was the fuel consumption on the vessel last month?") == Domain.MARITIME


def test_medical_english_keywords():
    assert classify_domain("The patient has a fever and low blood pressure.") == Domain.MEDICAL


def test_classification_is_case_insensitive():
    assert classify_domain("GEMİDE BUNKER İKMALİ YAPILDI") == Domain.MARITIME


def test_existing_v1_test_questions_stay_maritime_or_general():
    """
    Regresyon güvencesi: mevcut test_pipeline_safety.py'deki soru metinleri
    domain routing eklendikten sonra da MARITIME/GENERAL kalmalı, yanlışlıkla
    MEDICAL/BOTH'a düşüp maritime akışı engellememeli.
    """
    maritime_or_general_questions = [
        "Bu soru veride kesinlikle olmayan bir konu hakkında",
        "AE yağı değişimiyle ilgili ne olmuş?",
        "çok alakasız bir soru",
        "MT IVANI 2021-02-20 tarihinde BBM ROB değeri nedir?",
        "2021-02-20 tarihinde bunker ile ilgili ne oldu?",
    ]
    for q in maritime_or_general_questions:
        assert classify_domain(q) in (Domain.MARITIME, Domain.GENERAL), q
