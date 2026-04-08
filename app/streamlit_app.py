import streamlit as st
from dotenv import load_dotenv
import os
import sys
from datetime import datetime

# Ensure project root is on sys.path so `src` package imports work when Streamlit changes cwd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()
from src.rag import query_rag

st.set_page_config(page_title="Seismic Reporter", layout="wide", initial_sidebar_state="collapsed")

K_DEFAULT = int(os.getenv("RAG_RETRIEVAL_K", "3"))
MAX_TOKENS_DEFAULT = int(os.getenv("RAG_MAX_TOKENS", "180"))
AUTHOR_NAME = os.getenv("AUTHOR_NAME", "Anami James A")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_about" not in st.session_state:
    st.session_state.show_about = False

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --bg-1: #f4f8ff;
            --bg-2: #eaf8f1;
            --ink: #11263a;
            --muted: #46627b;
            --card: rgba(255, 255, 255, 0.88);
            --border: rgba(17, 38, 58, 0.14);
            --footer-bg: rgba(255, 255, 255, 0.75);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg-1: #0d1726;
                --bg-2: #142433;
                --ink: #e6eef8;
                --muted: #adc0d4;
                --card: rgba(12, 24, 39, 0.82);
                --border: rgba(171, 196, 224, 0.22);
                --footer-bg: rgba(8, 17, 28, 0.68);
            }
        }

        [data-theme="dark"] {
            --bg-1: #0d1726;
            --bg-2: #142433;
            --ink: #e6eef8;
            --muted: #adc0d4;
            --card: rgba(12, 24, 39, 0.82);
            --border: rgba(171, 196, 224, 0.22);
            --footer-bg: rgba(8, 17, 28, 0.68);
        }

        .stApp {
            background: radial-gradient(circle at 15% 10%, var(--bg-2), transparent 35%),
                        radial-gradient(circle at 85% 20%, color-mix(in srgb, var(--bg-1) 78%, #4ea3ff 22%), transparent 32%),
                        linear-gradient(180deg, var(--bg-1), color-mix(in srgb, var(--bg-1) 72%, #ffffff 28%) 45%);
            color: var(--ink);
        }

        section.main > div {
            max-width: 1020px;
            margin: 0 auto;
            padding-bottom: 72px;
        }

        [data-testid="collapsedControl"] {
            opacity: 0.7;
        }

        .hero {
            font-family: 'Space Grotesk', sans-serif;
            border: 1px solid var(--border);
            background: var(--card);
            backdrop-filter: blur(8px);
            border-radius: 18px;
            padding: 1.1rem 1.2rem 1rem;
            margin-bottom: 1rem;
            animation: rise 550ms ease-out;
            box-shadow: 0 10px 30px rgba(16, 30, 48, 0.08);
        }

        .hero h1 {
            font-size: clamp(1.5rem, 2.6vw, 2.5rem);
            margin: 0;
            letter-spacing: -0.02em;
        }

        .hero p {
            margin: 0.4rem 0 0;
            color: var(--muted);
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .small-note {
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 0.92rem;
            color: var(--muted);
        }

        .top-actions {
            display: flex;
            justify-content: flex-end;
            margin: 0 0 0.8rem;
        }

        .footer {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 999;
            padding: 10px 18px;
            background: var(--footer-bg);
            border-top: 1px solid var(--border);
            backdrop-filter: blur(8px);
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--muted);
            text-align: center;
        }

        .footer a {
            color: inherit;
            font-weight: 600;
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        @keyframes rise {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 900px) {
            .hero { padding: 0.9rem 0.9rem; }
            .footer { font-size: 0.86rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <h1>Seismic Reporter</h1>
        <p>Ask about recent earthquake activity with retrieval-grounded answers from USGS data.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([0.82, 0.18], vertical_alignment="center")
with header_left:
    st.markdown("<p class='small-note'>Built for real-world data exploration, retrieval, and concise reporting.</p>", unsafe_allow_html=True)
with header_right:
    if st.button("About Seismic Reporter", use_container_width=True):
        st.session_state.show_about = not st.session_state.show_about

if st.session_state.show_about:
    st.markdown(
        """
        <section class="hero">
            <h3 style="margin:0 0 0.6rem 0; font-family:'Space Grotesk', sans-serif;">About this project</h3>
            <p style="margin:0 0 0.75rem 0;">Seismic Reporter is a compact AI reporting app that demonstrates how a fresher-level AI project can combine public data, retrieval, and LLM generation into a useful user experience.</p>
            <ul style="margin:0; padding-left: 1.1rem; color: var(--muted); font-family:'IBM Plex Sans', sans-serif; line-height:1.7;">
                <li><strong>Public data ingestion:</strong> pulls raw earthquake records from USGS feeds and turns them into structured documents.</li>
                <li><strong>RAG pipeline:</strong> embeds the raw records, retrieves the most relevant context, and answers questions using that context.</li>
                <li><strong>Groq API integration:</strong> sends the prompt and retrieved evidence to Groq for fast, natural-language answers.</li>
                <li><strong>Readable output:</strong> returns a short report with cited source events so the result is easy to verify.</li>
                <li><strong>Production-style UI:</strong> includes responsive layout, dark/light adaptive styling, persistent chat history, and a clean footer.</li>
            </ul>
            <p style="margin:0.8rem 0 0; color: var(--muted);">This layout is intended to show practical AI project skill: data handling, retrieval, LLM orchestration, and user-facing presentation.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="top-actions">', unsafe_allow_html=True)
if st.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    f"<p class='small-note'>Conversation turns: {len(st.session_state.chat_history)}</p>",
    unsafe_allow_html=True,
)

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.markdown(f"- {s.get('id')} - {s.get('meta', {}).get('place')}")
        st.caption(f"Answered at {turn['time']}")

q = st.chat_input("Ask about recent seismic activity...")
if q and q.strip():
    with st.chat_message("user"):
        st.write(q)
    with st.spinner("Retrieving and generating answer..."):
        try:
            res = query_rag(q, k=K_DEFAULT, max_tokens=MAX_TOKENS_DEFAULT)
        except Exception as e:
            st.error("Generation failed: see details below")
            st.exception(e)
        else:
            answer = res.get("answer", "")
            sources = res.get("sources", [])
            with st.chat_message("assistant"):
                st.write(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- {s.get('id')} - {s.get('meta', {}).get('place')}")
            st.session_state.chat_history.append(
                {
                    "question": q,
                    "answer": answer,
                    "sources": sources,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
            )

st.markdown(
    f"""
    <div class="footer">
        &copy; {datetime.now().year} {AUTHOR_NAME} 
    </div>
    """,
    unsafe_allow_html=True,
)
