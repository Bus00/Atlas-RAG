"""
tests/test_pipeline_domain_routing.py
-----------------------------------------
V2 Faz 2: RagPipeline.answer()'ın domain routing kapısını doğrular.

- MEDICAL/BOTH sorularında: retriever ve LLM'e HİÇ gidilmemeli (henüz var
  olmayan bir modül için kaynak harcamamak / yanlış cevap üretmemek için),
  açık "henüz kullanılamıyor" mesajı dönmeli.
- MARITIME/GENERAL sorularında: V1 davranışı DEĞİŞMEDEN devam etmeli
  (test_pipeline_safety.py zaten bunu kapsıyor; burada ayrıca domain
  etiketinin doğru işlendiğini doğruluyoruz).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from rag.pipeline import MEDICAL_NOT_YET_AVAILABLE_MESSAGE, RagPipeline
from routing.domain_router import Domain


def _make_pipeline():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    llm = MagicMock()
    return RagPipeline(retriever=retriever, llm_client=llm), retriever, llm


def test_medical_question_is_blocked_without_calling_retriever_or_llm():
    pipeline, retriever, llm = _make_pipeline()

    result = pipeline.answer("Hastanın ateşi 39.1 ve nabzı 120.")

    assert result.answer == MEDICAL_NOT_YET_AVAILABLE_MESSAGE
    assert result.domain == Domain.MEDICAL
    assert result.llm_was_called is False
    retriever.retrieve.assert_not_called()
    llm.generate.assert_not_called()


def test_both_domain_question_is_also_blocked():
    pipeline, retriever, llm = _make_pipeline()

    result = pipeline.answer("Gemide hasta var ve tahliye gerekiyor.")

    assert result.answer == MEDICAL_NOT_YET_AVAILABLE_MESSAGE
    assert result.domain == Domain.BOTH
    retriever.retrieve.assert_not_called()
    llm.generate.assert_not_called()


def test_maritime_question_still_reaches_retriever_unaffected():
    """Domain routing eklenmesi, mevcut maritime akışını (V1) bozmamalı."""
    from retrieval.retriever import RetrievedChunk

    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_text="MT IVANI - 2021-01-06 ... AE 2 yağ değişimi ...",
            vessel="MT IVANI",
            report_date="2021-01-06",
            chunk_type="event",
            event_type="maintenance",
            equipment_hint="AE 2",
            distance=0.1,
        )
    ]
    llm = MagicMock()
    llm.generate.return_value = "AE 2 için 56 litre yağ değişimi yapılmış."

    pipeline = RagPipeline(retriever=retriever, llm_client=llm)
    result = pipeline.answer("AE yağı değişimiyle ilgili ne olmuş?")

    assert result.domain == Domain.MARITIME
    assert result.llm_was_called is True
    assert "56 litre" in result.answer
    llm.generate.assert_called_once()


def test_general_question_still_returns_no_info_message_unaffected():
    pipeline, retriever, llm = _make_pipeline()

    result = pipeline.answer("Bu soru veride kesinlikle olmayan bir konu hakkında")

    from rag.pipeline import NO_INFO_MESSAGE

    assert result.answer == NO_INFO_MESSAGE
    assert result.domain == Domain.GENERAL
    llm.generate.assert_not_called()
