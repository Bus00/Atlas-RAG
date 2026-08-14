"""
app.py
--------
Maritime RAG Assistant — Streamlit chatbot arayüzü.

Bu dosya YALNIZCA UI'dır: retrieval/embedding/LLM mantığının hiçbiri burada
tekrar yazılmadı. Tüm iş mevcut `rag.pipeline.RagPipeline.answer()` üzerinden
yapılıyor; backend dosyalarına (rag/, retrieval/, embeddings/, config/, vb.)
dokunulmadı.

Çalıştırma:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from config.settings import settings  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402

st.set_page_config(page_title="Maritime RAG Assistant", page_icon="🚢", layout="wide")


@st.cache_resource(show_spinner="Sistem hazırlanıyor...")
def get_pipeline() -> RagPipeline:
    return RagPipeline()


def render_sources(sources) -> None:
    if not sources:
        return
    st.markdown("**Kaynaklar**")
    for i, s in enumerate(sources, start=1):
        label = f"{s.vessel} — {s.report_date} — {s.event_type or s.chunk_type}"
        with st.expander(label):
            st.write(s.chunk_text)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🚢 Maritime RAG Assistant")
        st.caption("Maritime operasyon raporları için RAG tabanlı doküman asistanı.")
        st.divider()
        st.markdown(f"**LLM Modeli:** `{settings.ollama_model}`")
        st.markdown("**Vector Store:** ChromaDB")
        st.markdown("**RAG Pipeline:** 🟢 Connected")
        st.divider()
        if st.button("🗑️ Yeni Sohbet", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


def main() -> None:
    render_sidebar()

    st.title("Maritime RAG Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    question = st.chat_input("Sorunuzu yazın (Örn: MT IVANI gemisinde bunker ile ilgili ne oldu?)")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Kayıtlar aranıyor ve cevap oluşturuluyor..."):
            try:
                pipeline = get_pipeline()
                result = pipeline.answer(question)
            except Exception:  # noqa: BLE001
                error_text = "Yanıt oluşturulamadı. Lütfen sistem bağlantısını (PostgreSQL / ChromaDB / Ollama) kontrol edin."
                st.error(error_text)
                st.session_state.chat_history.append({"role": "assistant", "content": error_text, "sources": []})
                return

        st.write(result.answer)
        render_sources(result.sources)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )


if __name__ == "__main__":
    main()
