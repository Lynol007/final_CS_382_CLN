# 📚 Book Search Assistant — RAG-Based AI Search System

**CS382 Final Project.** Ask a question about 20 popular self-improvement books
(Atomic Habits, Deep Work, The Psychology of Money, …) and get an AI answer
grounded in the library's study notes, with visible citations and similarity
scores back to the source material.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). The first run
downloads the embedding model (~90 MB) and builds the index; afterwards the
index loads from a disk cache in ~1 s.

**LLM configuration** lives in `.env` (not committed — contains the API key):

```
OPENAI_API_KEY=...            # any OpenAI-compatible key (this project uses Google Gemini)
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-flash-latest
OPENAI_FALLBACK_MODELS=gemini-3.5-flash,gemini-flash-lite-latest
```

Without a key the app still runs: it falls back to showing the retrieved
passages directly instead of an AI-written answer.

## Architecture

```
data/books/*.txt + catalog.json        25 documents (20 book study notes + 5 thematic guides)
        │
        ▼  rag/ingest.py               sentence-aware chunking, ~180 words/chunk,
        │                              2-sentence overlap → 119 Chunk records
        ▼  rag/embed_store.py          all-MiniLM-L6-v2 embeddings (local, free),
        │                              L2-normalised matrix, disk-cached by corpus hash;
        │                              cosine top-k search; TF-IDF fallback if not installed
        ▼  rag/generate.py             grounded prompt → Gemini → extractive fallback;
        │                              citations required; refuses when sources
        │                              don't answer
        ▼  app.py                      Streamlit UI: query form, answer panel, expandable
                                       sources with scores, top-k slider, answer-mode
                                       toggle, latency display
```

The four layers are independent modules with stable interfaces — each one was
upgraded from the TF-IDF starter without touching the others.

## Design decisions

- **Corpus: original study notes, not book text.** The 20 books are
  copyrighted, so the corpus is original ~900-word study-note documents written
  for this project (core thesis, key frameworks, practices, cross-references),
  plus 5 thematic guides that compare books. The guides measurably help:
  comparison and "which book should I read" queries retrieve them first (see
  evaluation).
- **Sentence-aware chunking** (~180-word budget, 2-sentence overlap) instead of
  the starter's fixed word windows: no sentence is ever split across a
  boundary, and the overlap keeps boundary-spanning ideas retrievable.
- **`all-MiniLM-L6-v2` over TF-IDF**: retrieval by meaning, not word overlap —
  "stop caring what people think" finds *The Courage to Be Disliked* despite
  zero shared keywords. Local and free, no API dependency for retrieval.
- **In-memory cosine search over a vector DB**: at 119 chunks a normalised
  matrix multiply is ~100 ms; FAISS/Chroma would add complexity with no gain
  at this scale. The `VectorStore.build/query` interface is DB-shaped so one
  could be swapped in for a larger corpus.
- **Disk-cached embeddings keyed by corpus hash**: re-embedding only happens
  when documents actually change — keeps demo startup fast.
- **Generation fallback** (Gemini → extractive passages) with retry + fallback
  models on transient 503s: the live demo cannot be broken by a missing key,
  a down API, or a provider load spike.
- **Graceful failure by relevance floor**: if no chunk scores ≥ 0.20 cosine,
  the app says it found nothing relevant and never calls the LLM — off-topic
  questions structurally cannot produce hallucinated answers.

## Evaluation

See [eval/EVALUATION.md](eval/EVALUATION.md): 8 test queries across 7 query
types. **Retrieval hit@4 = 7/7, citation accuracy = 7/7, graceful failure =
1/1**, with the tuning changes the evaluation motivated (relevance floor
0.30 → 0.20, generation token budget 1,000 → 3,000). Reproduce with
`.venv/bin/python -m eval.run_eval --llm`.

## Known limitations

- Answers reflect the study notes' coverage of each book, not the full text —
  a question about an obscure chapter detail may correctly return "not enough
  information".
- The 0.20 relevance floor was tuned on a single off-topic probe; a larger
  negative test set would tighten it.
- Single-turn only (no conversation memory), English only.
- Answer latency is ~10–18 s (LLM generation dominates); streaming would
  improve perceived speed.

## Project structure

```
├── app.py                   # Streamlit interface
├── requirements.txt
├── .env                     # API key + model config (not committed)
├── data/books/              # 25-document corpus + catalog.json metadata
├── rag/
│   ├── ingest.py            # load + sentence-aware chunking
│   ├── embed_store.py       # embeddings + cached cosine index
│   └── generate.py          # grounded generation with fallback chain
└── eval/
    ├── run_eval.py          # evaluation harness (8 queries)
    ├── EVALUATION.md        # results + discussion
    └── results.json         # raw per-query output
```
