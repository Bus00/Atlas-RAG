from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from database.db import check_connection  # noqa: E402
from llm.ollama_client import OllamaClient  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402
from vectorstore.chroma_store import ChromaStore  # noqa: E402


# Sayfa yapılandırması

st.set_page_config(page_title="ATLAS — Your Assistant", page_icon="◈", layout="centered")

ASSISTANT_SYMBOL = "◈"
USER_SYMBOL = "●"

EXAMPLE_QUESTIONS = [
    "Bunker operasyonlarıyla ilgili neler oldu?",
    "Mart ayında yakıt tüketimi ne kadardı?",
    "AE lube oil ile ilgili neler oldu?",
    "MT IVANI ile ilgili olayları göster.",
]



# Tema

def inject_css() -> None:
    st.markdown(
        """
        <style>
            #MainMenu, footer, header {visibility: hidden;}

            .stApp {
                background-color: #0e0f11;
                color: #e6e6e6;
            }

            section[data-testid="stSidebar"] {
                background-color: #131416;
                border-right: 1px solid #22242a;
            }

            .block-container {
                padding-top: 2.5rem;
                max-width: 780px;
            }

            .atlas-sidebar-title {
                font-size: 1.1rem;
                font-weight: 600;
                letter-spacing: 0.02em;
                margin-bottom: 0;
                color: #f2f2f2;
            }
            .atlas-sidebar-subtitle {
                font-size: 0.8rem;
                color: #8a8d94;
                margin-top: 0;
                margin-bottom: 1.1rem;
            }
            .atlas-section-label {
                font-size: 0.7rem;
                letter-spacing: 0.08em;
                color: #6b6e76;
                margin-top: 1.4rem;
                margin-bottom: 0.4rem;
                text-transform: uppercase;
            }
            .atlas-recent-item {
                font-size: 0.82rem;
                color: #b7b9bf;
                padding: 0.25rem 0;
                border-bottom: 1px solid #1c1e22;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .atlas-status-row {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 0.82rem;
                color: #cfd1d6;
                padding: 0.2rem 0;
            }
            .atlas-dot-ok { color: #4ade80; }
            .atlas-dot-fail { color: #f87171; }

            .stButton > button {
                background-color: #1a1c20;
                color: #e6e6e6;
                border: 1px solid #2a2c32;
                border-radius: 8px;
                font-size: 0.85rem;
                padding: 0.45rem 0.9rem;
            }
            .stButton > button:hover {
                border-color: #3d4046;
                background-color: #202226;
                color: #ffffff;
            }

            .atlas-welcome {
                text-align: center;
                padding: 3.5rem 0 1.5rem 0;
            }
            .atlas-welcome-symbol {
                font-size: 2.6rem;
                color: #e6e6e6;
            }
            .atlas-welcome-title {
                font-size: 1.6rem;
                font-weight: 600;
                margin-top: 0.3rem;
                color: #f2f2f2;
            }
            .atlas-welcome-subtitle {
                font-size: 0.95rem;
                color: #9a9da4;
                margin-top: 0.3rem;
                margin-bottom: 1.6rem;
            }

            .atlas-msg {
                margin-bottom: 1.4rem;
            }
            .atlas-msg-label {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 0.85rem;
                font-weight: 600;
                margin-bottom: 0.35rem;
                color: #d6d8dc;
            }
            .atlas-msg-label .symbol {
                font-size: 0.95rem;
            }
            .atlas-msg-user .atlas-msg-label { color: #a9acb3; }
            .atlas-msg-content {
                font-size: 0.95rem;
                line-height: 1.55;
                color: #e6e6e6;
                white-space: pre-wrap;
            }

            .atlas-thinking-label {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.9rem;
                color: #9a9da4;
            }
            .atlas-pulse {
                display: inline-block;
                animation: atlas-pulse 1.3s ease-in-out infinite;
            }
            @keyframes atlas-pulse {
                0%, 100% { opacity: 0.35; transform: scale(0.92); }
                50% { opacity: 1; transform: scale(1.18); }
            }

            .atlas-source-item {
                font-size: 0.82rem;
                color: #c2c4c9;
                padding: 0.35rem 0;
                border-bottom: 1px solid #1c1e22;
            }
            .atlas-source-meta {
                color: #8a8d94;
                font-size: 0.76rem;
            }
            .atlas-source-text {
                color: #9a9da4;
                font-size: 0.78rem;
                margin-top: 0.15rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )



# Backend erişimi

@st.cache_resource(show_spinner="ATLAS hazırlanıyor (embedding modeli yükleniyor)...")
def get_pipeline() -> RagPipeline:
    return RagPipeline()


def get_system_status() -> dict:
    status = {"postgres": False, "chroma": False, "ollama": False, "chroma_count": None}

    try:
        status["postgres"] = bool(check_connection())
    except Exception:  # noqa: BLE001
        status["postgres"] = False

    try:
        status["chroma_count"] = ChromaStore().count()
        status["chroma"] = True
    except Exception:  # noqa: BLE001
        status["chroma"] = False

    try:
        status["ollama"] = bool(OllamaClient().is_available())
    except Exception:  # noqa: BLE001
        status["ollama"] = False

    return status



# Sidebar

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f'<p class="atlas-sidebar-title">{ASSISTANT_SYMBOL} ATLAS</p>', unsafe_allow_html=True)
        st.markdown('<p class="atlas-sidebar-subtitle">Your Assistant</p>', unsafe_allow_html=True)

        if st.button("＋ New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown('<p class="atlas-section-label">Recent</p>', unsafe_allow_html=True)
        user_questions = [
            msg["content"] for msg in st.session_state.get("chat_history", []) if msg["role"] == "user"
        ]
        if not user_questions:
            st.markdown('<p class="atlas-recent-item">—</p>', unsafe_allow_html=True)
        else:
            for question in reversed(user_questions[-8:]):
                label = question if len(question) <= 42 else question[:39] + "..."
                st.markdown(f'<div class="atlas-recent-item">{label}</div>', unsafe_allow_html=True)

        st.markdown('<p class="atlas-section-label">System</p>', unsafe_allow_html=True)
        status = get_system_status()

        def status_row(label: str, ok: bool) -> str:
            dot_class = "atlas-dot-ok" if ok else "atlas-dot-fail"
            return f'<div class="atlas-status-row"><span class="{dot_class}">●</span>{label}</div>'

        st.markdown(status_row("PostgreSQL", status["postgres"]), unsafe_allow_html=True)
        chroma_label = "ChromaDB"
        if status["chroma"] and status["chroma_count"] is not None:
            chroma_label = f"ChromaDB ({status['chroma_count']})"
        st.markdown(status_row(chroma_label, status["chroma"]), unsafe_allow_html=True)
        st.markdown(status_row("Ollama", status["ollama"]), unsafe_allow_html=True)
        pipeline_ok = status["postgres"] and status["chroma"] and status["ollama"]
        st.markdown(status_row("RAG Pipeline", pipeline_ok), unsafe_allow_html=True)



# Welcome screen

def render_welcome_screen() -> str | None:
    st.markdown(
        f"""
        <div class="atlas-welcome">
            <div class="atlas-welcome-symbol">{ASSISTANT_SYMBOL}</div>
            <div class="atlas-welcome-title">ATLAS</div>
            <div class="atlas-welcome-subtitle">How can I help you? Query your available records.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    clicked_question = None
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                clicked_question = example
    return clicked_question



# Mesaj - kaynak render

def render_message(role: str, content: str) -> None:
    symbol = ASSISTANT_SYMBOL if role == "assistant" else USER_SYMBOL
    name = "ATLAS" if role == "assistant" else "USER"
    row_class = "atlas-msg-assistant" if role == "assistant" else "atlas-msg-user"
    st.markdown(
        f"""
        <div class="atlas-msg {row_class}">
            <div class="atlas-msg-label"><span class="symbol">{symbol}</span>{name}</div>
            <div class="atlas-msg-content">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources, used_structured: bool, structured_source=None) -> None:
    if not sources and not used_structured:
        return
    with st.expander(f"Sources ({len(sources)}{' + structured lookup' if used_structured else ''})"):
        if used_structured:
         if structured_source:
          st.markdown(
            f"""
            <div class="atlas-source-item">
                <strong>[Structured lookup]</strong>
                <span class="atlas-source-meta"> · PostgreSQL · Noon Report</span>
                <div class="atlas-source-text">
                    MT IVANI — {structured_source.get("year", "")}/{structured_source.get("month", "")}
                    · BBM Consumption: {structured_source.get("bbm_consumption", "")} L
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        else:
         st.markdown(
            '<div class="atlas-source-item"><strong>[Structured lookup]</strong> '
            "Matching Noon Report ROB/Consumption record (PostgreSQL)</div>",
            unsafe_allow_html=True,
        )
        for s in sources:
            equipment = f" · equipment: {s.equipment_hint}" if s.equipment_hint else ""
            st.markdown(
                f"""
                <div class="atlas-source-item">
                    <strong>{s.vessel}</strong> — {s.report_date}
                    <span class="atlas-source-meta"> · {s.event_type or s.chunk_type}{equipment}
                    · distance: {s.distance:.3f}</span>
                    <div class="atlas-source-text">{s.chunk_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_thinking_placeholder(placeholder) -> None:
    placeholder.markdown(
        f"""
        <div class="atlas-thinking-label">
            <span class="atlas-pulse">{ASSISTANT_SYMBOL}</span> thinking...
        </div>
        """,
        unsafe_allow_html=True,
    )



# Soru işleme

def handle_question(question: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": question})

    placeholder = st.empty()
    render_thinking_placeholder(placeholder)

    try:
        pipeline = get_pipeline()
        result = pipeline.answer(question)
        placeholder.empty()
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
                "used_structured": result.used_structured_lookup,
                "structured_source": result.structured_source,
            }
        )
    except Exception as exc:  # noqa: BLE001
        placeholder.empty()
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"Beklenmeyen bir hata oluştu: {exc}",
                "sources": [],
                "used_structured": False,
            }
        )

    st.rerun()


# Main

def main() -> None:
    inject_css()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    render_sidebar()

    example_question = None
    if not st.session_state.chat_history:
        example_question = render_welcome_screen()
    else:
        for msg in st.session_state.chat_history:
            render_message(msg["role"], msg["content"])
            if msg["role"] == "assistant" and msg.get("sources") is not None:
                render_sources(
    msg["sources"],
    msg.get("used_structured", False),
    msg.get("structured_source"),
)

    typed_question = st.chat_input("Ask ATLAS about your records...")

    question = typed_question or example_question
    if question:
        handle_question(question)


if __name__ == "__main__":
    main()
