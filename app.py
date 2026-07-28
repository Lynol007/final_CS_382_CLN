"""
📚 Book Search Assistant — RAG-based AI search over a self-improvement library.

Run with:
    streamlit run app.py

Pipeline: ingest & chunk (rag/ingest.py) → embed & index (rag/embed_store.py)
→ retrieve top-k → generate a grounded, cited answer (rag/generate.py).
"""

import time

import streamlit as st

from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer

DATA_FOLDER = "data/books"

EXAMPLE_QUERIES = [
    "How do I break a bad habit?",
    "What's the smartest way to start investing?",
    "How do I stop caring what people think?",
    "Which book should I read first if I procrastinate?",
]

st.set_page_config(page_title="Book Search Assistant", page_icon="📚", layout="centered")

# ---------- styling ----------

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      html, body, [class*="css"] {font-family: 'Inter', -apple-system, sans-serif;}
      #MainMenu, footer {visibility: hidden;}
      header[data-testid="stHeader"] {background: transparent;}
      [data-testid="stSidebarCollapsedControl"] {visibility: visible;}

      /* Full-page glossy gradient backdrop */
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(1200px 600px at 12% -8%, #ffe9d6 0%, rgba(255,233,214,0) 55%),
          radial-gradient(1000px 700px at 100% 0%, #fff0e0 0%, rgba(255,240,224,0) 50%),
          linear-gradient(180deg, #FFFFFF 0%, #FFF6EF 100%);
        background-attachment: fixed;
      }
      .block-container {padding-top: 2.2rem; padding-bottom: 4rem; max-width: 52rem;}

      /* Animated glossy hero */
      .hero {
        position: relative; overflow: hidden;
        background: linear-gradient(120deg, #FF6A00 0%, #FF8A2B 45%, #FFB169 100%);
        background-size: 200% 200%; animation: shift 12s ease infinite;
        border-radius: 1.35rem; padding: 2.1rem 2.2rem; color: #fff;
        box-shadow: 0 20px 45px -12px rgba(255,106,0,.5), inset 0 1px 0 rgba(255,255,255,.4);
      }
      .hero::before {  /* glossy sheen */
        content:""; position:absolute; inset:0;
        background: linear-gradient(180deg, rgba(255,255,255,.32) 0%, rgba(255,255,255,0) 42%);
        pointer-events:none;
      }
      .hero::after {   /* soft light orb */
        content:""; position:absolute; top:-40%; right:-10%; width:340px; height:340px;
        background: radial-gradient(circle, rgba(255,255,255,.4), rgba(255,255,255,0) 70%);
        pointer-events:none;
      }
      @keyframes shift {0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}
      .hero h1 {font-size: 2.3rem; margin: 0 0 .4rem 0; font-weight: 800; letter-spacing: -.02em;
                text-shadow: 0 2px 10px rgba(0,0,0,.12);}
      .hero p  {margin: 0; font-size: 1.03rem; line-height: 1.6; color: rgba(255,255,255,.96); max-width: 40rem;}
      .hero .badge {display:inline-block; margin-bottom:.7rem; padding:.28rem .8rem; border-radius:99px;
                    background: rgba(255,255,255,.22); border:1px solid rgba(255,255,255,.45);
                    font-size:.76rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
                    backdrop-filter: blur(6px);}

      /* Glassmorphism answer card */
      .answer-card {
        background: rgba(255,255,255,.78); backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,.7); border-left: 4px solid #FF7A18; border-radius: 1.1rem;
        padding: 1.5rem 1.7rem; margin-top: .6rem;
        box-shadow: 0 18px 40px -18px rgba(200,90,0,.32), inset 0 1px 0 rgba(255,255,255,.8);
      }
      .answer-card h3 {margin: 0 0 .7rem 0; font-size: 1.02rem; font-weight: 700;
        background: linear-gradient(90deg,#FF6A00,#FFA24D); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;}

      .meta {color: #A38A78; font-size: .86rem; margin: .7rem .2rem 1.6rem;}

      /* Source similarity bar */
      .simwrap {display:flex; align-items:center; gap:.6rem; margin:-.2rem 0 .7rem;}
      .simbar {flex:1; height:8px; border-radius:99px; background:#FDE4CE; overflow:hidden;
               box-shadow: inset 0 1px 2px rgba(200,90,0,.14);}
      .simfill {height:100%; border-radius:99px;
                background:linear-gradient(90deg,#FFA24D,#FF6A00); box-shadow: 0 0 10px rgba(255,106,0,.5);}
      .simval {font-size:.8rem; color:#E8620A; font-weight:700; font-variant-numeric:tabular-nums;}

      /* Glossy buttons */
      .stButton button, .stFormSubmitButton button {
        border-radius:.8rem; font-weight:600; border:1px solid rgba(255,122,24,.3);
        transition: transform .12s ease, box-shadow .12s ease;
      }
      .stButton button:hover, .stFormSubmitButton button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 20px -6px rgba(255,122,24,.55);
      }
      .stFormSubmitButton button {
        background: linear-gradient(120deg,#FF6A00,#FF8A2B); color:#fff; border:none;
        box-shadow: 0 8px 22px -8px rgba(255,106,0,.75);
      }
      /* Example chips as glass pills */
      .stButton button {background: rgba(255,255,255,.75); backdrop-filter: blur(8px); color:#C85A00;}

      div[data-testid="stExpander"] details {
        border-radius:.85rem; border:1px solid rgba(255,255,255,.7);
        background: rgba(255,255,255,.6); backdrop-filter: blur(10px);
      }

      .chips-label {color:#A38A78; font-size:.85rem; margin:1.4rem 0 .5rem; font-weight:500;}

      /* Sidebar glass */
      section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,.9), rgba(255,244,236,.92));
        backdrop-filter: blur(12px); border-right: 1px solid rgba(255,255,255,.7);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading and indexing the library (first run downloads the embedding model)...")
def load_store():
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks


try:
    store, docs, chunks = load_store()
except Exception as exc:
    st.error(f"Failed to load the library: {exc}")
    st.stop()

n_books = sum(1 for d in docs if d["kind"] == "book_notes")

# ---------- sidebar ----------

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Passages to retrieve (top-k)", min_value=1, max_value=10, value=4)
    mode = st.radio(
        "Answer mode",
        ["llm", "extractive"],
        index=0,
        format_func=lambda m: "🤖 AI answer (cited)" if m == "llm" else "📄 Passages only",
        help="AI mode asks the configured LLM (Gemini); if it's unavailable, the app shows the retrieved passages instead.",
    )

    st.divider()
    st.markdown("### 📊 Library")
    c1, c2, c3 = st.columns(3)
    c1.metric("Books", n_books)
    c2.metric("Guides", len(docs) - n_books)
    c3.metric("Chunks", len(chunks))
    backend_label = "semantic embeddings" if store.backend == "embeddings" else "TF-IDF (fallback)"
    st.caption(f"Retrieval: **{backend_label}**")

    with st.expander("📖 Browse the library"):
        for d in docs:
            if d["kind"] == "book_notes":
                st.markdown(f"**{d['title']}**  \n<span style='color:#8A8FA3;font-size:.85rem'>{d['author']}</span>",
                            unsafe_allow_html=True)
        st.markdown("---\n*Cross-book guides:*")
        for d in docs:
            if d["kind"] == "guide":
                st.markdown(f"- {d['title'].removeprefix('Guide: ')}")

# ---------- hero ----------

st.markdown(
    f"""
    <div class="hero">
      <span class="badge">✦ RAG-powered · cited answers</span>
      <h1>📚 Book Search Assistant</h1>
      <p>Ask anything about {n_books} popular self-improvement books — habits, money, focus,
      mindset, relationships. Every answer is grounded in the library and cites its sources.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- query input ----------

def _use_example(text: str) -> None:
    st.session_state.q = text
    st.session_state.do_search = True


st.markdown('<div class="chips-label">Try one of these ↓</div>', unsafe_allow_html=True)
chip_cols = st.columns(2)
for idx, example in enumerate(EXAMPLE_QUERIES):
    chip_cols[idx % 2].button(example, key=f"ex{idx}", on_click=_use_example, args=(example,),
                              use_container_width=True)

with st.form("search", border=False):
    input_col, button_col = st.columns([5, 1], vertical_alignment="bottom")
    query = input_col.text_input(
        "Your question",
        key="q",
        placeholder="e.g. What does Atomic Habits say about breaking a bad habit?",
        label_visibility="collapsed",
    )
    submitted = button_col.form_submit_button("Search", type="primary", use_container_width=True)

run_search = submitted or st.session_state.pop("do_search", False)

# ---------- search + answer ----------

if run_search:
    if not query.strip():
        st.warning("Type a question first — or tap one of the examples above.")
        st.stop()

    t0 = time.perf_counter()
    try:
        with st.spinner("🔎 Searching the library..."):
            retrieved = store.query(query, top_k=top_k)
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        st.stop()
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Graceful failure: if nothing clears the relevance floor, say so — don't hallucinate.
    relevant = [(c, s) for c, s in retrieved if s >= store.min_relevance]
    if not relevant:
        st.warning(
            "😕 I couldn't find anything in the library that's relevant to that question. "
            "Try rephrasing, or ask about habits, productivity, money, mindset, or relationships."
        )
        st.stop()

    t1 = time.perf_counter()
    try:
        with st.spinner("✍️ Writing a grounded answer from the sources..."):
            answer, engine = generate_answer(query, relevant, mode=mode)
    except Exception as exc:
        st.error(f"Answer generation failed: {exc}")
        st.stop()
    generation_ms = (time.perf_counter() - t1) * 1000

    # Answer card
    st.markdown('<div class="answer-card"><h3>💡 Answer</h3>', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown("</div>", unsafe_allow_html=True)

    engine_names = {
        "gemini": "Google Gemini",
        "openai": "OpenAI (ChatGPT)",
        "extractive": "extractive (no LLM)",
        "extractive-fallback": "extractive fallback (no LLM available)",
    }
    st.markdown(
        f'<div class="meta">⏱️ retrieval {retrieval_ms:.0f} ms · generation {generation_ms:.0f} ms · '
        f'engine: {engine_names.get(engine, engine)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("##### 📑 Sources")
    # Show each book only once (its best-scoring passage). `relevant` is already
    # ordered by descending score, so the first time we see a title is its best.
    seen_titles = set()
    for chunk, score in relevant:
        if chunk.doc_title in seen_titles:
            continue
        seen_titles.add(chunk.doc_title)
        with st.expander(f"{chunk.doc_title} — {chunk.author}"):
            pct = int(min(1.0, max(0.0, score)) * 100)
            st.markdown(
                f'<div class="simwrap"><div class="simbar"><div class="simfill" style="width:{pct}%"></div></div>'
                f'<span class="simval">{score:.2f}</span></div>',
                unsafe_allow_html=True,
            )
            st.write(chunk.text)
else:
    st.info("💬 Type a question above (or tap an example) and hit **Search**.")
