"""
tests/test_pipeline_safety.py
--------------------------------
En kritik davranışı test eder: ilgili context bulunamadığında sistem
LLM'i hiç çağırmadan güvenli "bulunamadı" cevabını döndürmeli (hallucination
riski sıfıra iner). Retriever ve LLM mock'lanıyor — gerçek ChromaDB/Ollama
GEREKMEZ.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from rag.pipeline import NO_INFO_MESSAGE, SYSTEM_PROMPT, RagPipeline
from retrieval.structured_lookup import StructuredFact


def _make_pipeline_with_empty_retrieval():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    llm = MagicMock()
    return RagPipeline(retriever=retriever, llm_client=llm), retriever, llm


def test_no_context_returns_safe_message_without_calling_llm():
    pipeline, retriever, llm = _make_pipeline_with_empty_retrieval()

    result = pipeline.answer("Bu soru veride kesinlikle olmayan bir konu hakkında")

    assert result.answer == NO_INFO_MESSAGE
    assert result.llm_was_called is False
    llm.generate.assert_not_called()


def test_relevant_chunks_trigger_llm_call():
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
            distance=0.1,  # eşik altında -> alakalı kabul edilmeli
        )
    ]
    llm = MagicMock()
    llm.generate.return_value = "AE 2 için 56 litre yağ değişimi yapılmış."

    pipeline = RagPipeline(retriever=retriever, llm_client=llm)
    result = pipeline.answer("AE yağı değişimiyle ilgili ne olmuş?")

    assert result.llm_was_called is True
    assert "56 litre" in result.answer
    llm.generate.assert_called_once()


def test_low_relevance_chunks_are_filtered_out():
    from retrieval.retriever import RetrievedChunk

    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_text="alakasız bir kayıt",
            vessel="MT IVANI",
            report_date="2021-01-06",
            chunk_type="event",
            event_type="other",
            equipment_hint="",
            distance=0.9,  # eşik üstünde -> alakasız, filtrelenmeli
        )
    ]
    llm = MagicMock()

    pipeline = RagPipeline(retriever=retriever, llm_client=llm)
    result = pipeline.answer("çok alakasız bir soru")

    assert result.answer == NO_INFO_MESSAGE
    llm.generate.assert_not_called()


def test_structured_fact_answers_directly_without_retrieval_or_llm():
    """
    Structured bir soru (tarih + ROB/consumption) için PostgreSQL'de kayıt
    bulunduğunda: cevap doğrudan kayıttan üretilmeli, ChromaDB retrieval
    ve LLM çağrısı HİÇ yapılmamalı. Bkz. rag/pipeline.py tasarım notu.
    """
    fact = StructuredFact(
        report_date=date(2021, 2, 20),
        vessel="MT IVANI",
        fw_rob=22.00,
        bbm_rob=4385.00,
        me_oil_rob=152.00,
        ae_oil_rob=252.00,
        fw_consumption=2.00,
        bbm_consumption=216.00,
        me_oil_consumption=0.00,
        ae_oil_consumption=0.00,
    )

    retriever = MagicMock()
    llm = MagicMock()
    pipeline = RagPipeline(retriever=retriever, llm_client=llm)

    with patch("rag.pipeline.lookup_noon_report_by_date", return_value=fact):
        result = pipeline.answer("MT IVANI 2021-02-20 tarihinde BBM ROB değeri nedir?")

    assert result.used_structured_lookup is True
    assert result.llm_was_called is False
    assert "4385.0" in result.answer or "4385" in result.answer
    retriever.retrieve.assert_not_called()
    llm.generate.assert_not_called()


def test_system_prompt_instructs_partial_answers():
    """
    Regresyon testi: SYSTEM_PROMPT, context'te sorunun sadece bir kısmı
    varken tüm cevabı reddetmemesi (önceki hatalı davranış) için gereken
    talimatı içermeli. Bu, LLM'in gerçek cevabına bağımlı olmayan, prompt
    içeriği üzerinde çalışan bir kontrol.
    """
    assert "MUTLAKA cevapla" in SYSTEM_PROMPT
    assert "sadece o eksik kısmı" in SYSTEM_PROMPT
    assert NO_INFO_MESSAGE in SYSTEM_PROMPT


def test_system_prompt_prevents_negative_inference_from_partial_bunker_data():
    """
    Regresyon testi (gerçek bug): event_type='bunker' olarak etiketlenmiş,
    "bunker öncesi" değeri olan ama "bunker sonrası" değeri eksik/kesik olan
    bir kayıtta, model önceden "bunker yapılmadı" gibi yanlış bir olumsuz
    sonuç çıkarıyordu. SYSTEM_PROMPT'un bunu engelleyen açık talimatları
    içerdiğini doğrular — LLM'in gerçek cevabına bağımlı değil.
    """
    assert "bunker/ikmal olayı" in SYSTEM_PROMPT or "bunker/ikmal" in SYSTEM_PROMPT
    assert "yok saymanı GEREKTİRMEZ" in SYSTEM_PROMPT
    assert "birbirine karıştırma" in SYSTEM_PROMPT


def test_llm_is_called_with_current_system_prompt():
    """
    Semantic akışta LLM'e giden system prompt'un gerçekten SYSTEM_PROMPT
    sabiti olduğunu doğrular — biri prompt'u günceller ama çağrıya
    yansıtmayı unutursa bu test yakalar.
    """
    from retrieval.retriever import RetrievedChunk

    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_text="MT IVANI - 2021-02-20 tarihli kayıt: bunker öncesi 4.035 L",
            vessel="MT IVANI",
            report_date="2021-02-20",
            chunk_type="event",
            event_type="bunker",
            equipment_hint="",
            distance=0.1,
        )
    ]
    llm = MagicMock()
    llm.generate.return_value = "Bunker öncesi değer 4.035 L olarak kaydedilmiş."

    pipeline = RagPipeline(retriever=retriever, llm_client=llm)
    pipeline.answer("2021-02-20 tarihinde bunker ile ilgili ne oldu?")

    _, kwargs = llm.generate.call_args
    assert kwargs["system"] == SYSTEM_PROMPT


def test_non_empty_llm_answer_is_returned_unmodified():
    """
    Pipeline, LLM'den dolu bir cevap geldiğinde bunu değiştirmemeli /
    NO_INFO_MESSAGE ile ezmemeli — cevap kısmi bilgi içerse (örn. "eksik"
    kelimesi geçse) bile olduğu gibi döndürülmeli.
    """
    from retrieval.retriever import RetrievedChunk

    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_text="MT IVANI - 2021-02-20 tarihli kayıt: bunker öncesi 4.035 L, sonrası kesik",
            vessel="MT IVANI",
            report_date="2021-02-20",
            chunk_type="event",
            event_type="bunker",
            equipment_hint="",
            distance=0.1,
        )
    ]
    llm = MagicMock()
    partial_answer = (
        "2021-02-20 tarihinde bunker öncesi değer 4.035 L olarak kaydedilmiş. "
        "Bunker sonrası değer ise görüntüde kesik olduğu için mevcut veride yok."
    )
    llm.generate.return_value = partial_answer

    pipeline = RagPipeline(retriever=retriever, llm_client=llm)
    result = pipeline.answer("2021-02-20 tarihinde bunker ile ilgili ne oldu?")

    assert result.answer == partial_answer
    assert result.answer != NO_INFO_MESSAGE
