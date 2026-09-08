"""
tests/test_streamlit_sources_rendering.py
--------------------------------------------
Regresyon testi: app/streamlit_app.py::render_sources() içindeki girinti
hatası (else, iç `if structured_source:` yerine dış `if used_structured:`e
bağlanıyordu) için.

Bu dosya olmadan bug fark edilmiyordu çünkü mevcut test paketinde UI
katmanına (app/streamlit_app.py) dokunan hiçbir test yoktu -- diğer 4 test
dosyasının hepsi backend modüllerini (rag/, retrieval/, ingestion/) test
ediyor.

İzolasyon yöntemi: gerçek Streamlit çalışan bir uygulama (script run context)
olmadan çağrıldığında güvenilir davranmayabilir / gürültülü uyarılar
basabilir. Bunu önlemek ve st.markdown()'a giden metni yakalamak için
`unittest.mock.patch` (stdlib) ile yalnızca `streamlit.markdown` ve
`streamlit.expander` fonksiyonları test süresince mock'lanıyor. Bu, hem
gerçekten kurulu Streamlit paketiyle hem de Streamlit kurulu olmayan bir
ortamdaki minimal bir stub ile aynı şekilde çalışır -- teste özgü hiçbir
sahte API'ye (örn. bir önceki sürümdeki `st.markdown_calls`) bağımlı değil.
"""
from __future__ import annotations

from unittest.mock import patch

from app.streamlit_app import render_sources


class _FakeSource:
    """retrieval.retriever.RetrievedChunk ile aynı arayüz (render_sources sadece attribute erişiyor)."""

    def __init__(self):
        self.vessel = "MT IVANI"
        self.report_date = "2021-01-06"
        self.chunk_type = "event"
        self.event_type = "maintenance"
        self.equipment_hint = "AE 2"
        self.chunk_text = "AE 2 yağ değişimi yapıldı."
        self.distance = 0.12


def _render_and_capture(**kwargs) -> str:
    """render_sources()'ı çağırır ve st.markdown()'a giden tüm metinleri birleştirip döner."""
    with patch("streamlit.expander"), patch("streamlit.markdown") as mock_markdown:
        render_sources(**kwargs)
    return "\n".join(call.args[0] for call in mock_markdown.call_args_list)


def test_daily_structured_lookup_without_detail_shows_generic_fallback():
    """
    Bug senaryosu 1: günlük tarih sorgusu (structured_source=None,
    sources=[]) -- rag/pipeline.py'de bu hep böyledir (structured_source
    yalnızca aylık sorguda dolduruluyor). Düzeltmeden önce: hiçbir mesaj
    basılmıyordu (expander boş açılıyordu).
    """
    combined = _render_and_capture(sources=[], used_structured=True, structured_source=None)
    assert "Matching Noon Report ROB/Consumption record" in combined


def test_monthly_structured_lookup_with_detail_shows_detail_block():
    """
    Aylık sorgu: structured_source dolu geliyor -> detay bloğu (yıl/ay/BBM)
    basılmalı, generic fallback DEĞİL.
    """
    structured_source = {"year": 2021, "month": 3, "bbm_consumption": 12345}
    combined = _render_and_capture(sources=[], used_structured=True, structured_source=structured_source)
    assert "2021" in combined and "3" in combined
    assert "Matching Noon Report ROB/Consumption record" not in combined


def test_plain_rag_answer_does_not_show_structured_lookup_label():
    """
    Bug senaryosu 2: used_structured=False ama gerçek kaynaklar var (normal
    semantik RAG cevabı). Düzeltmeden önce: dış `else` her zaman çalıştığı
    için "[Structured lookup] ... Matching Noon Report ..." metni YANLIŞ
    şekilde basılıyordu.
    """
    combined = _render_and_capture(sources=[_FakeSource()], used_structured=False, structured_source=None)
    assert "Matching Noon Report ROB/Consumption record" not in combined
    assert "[Structured lookup]" not in combined
    # Gerçek kaynak yine de gösterilmeli:
    assert "MT IVANI" in combined
    assert "AE 2 yağ değişimi yapıldı." in combined


def test_no_sources_and_not_structured_renders_nothing():
    """Erken return -- st.expander hiç açılmamalı, hiçbir markdown çağrısı olmamalı."""
    with patch("streamlit.expander") as mock_expander, patch("streamlit.markdown") as mock_markdown:
        render_sources(sources=[], used_structured=False, structured_source=None)
    mock_expander.assert_not_called()
    mock_markdown.assert_not_called()
