# Evaluation — Book Search Assistant

Evaluated on 2026-07-22 with the full pipeline: sentence-aware chunking →
`all-MiniLM-L6-v2` embeddings → cosine top-4 retrieval → Gemini
(`gemini-flash-latest`) generation. Reproduce with:

```bash
.venv/bin/python -m eval.run_eval          # retrieval metrics only (free)
.venv/bin/python -m eval.run_eval --llm    # + LLM generation metrics
```

Raw per-query output (including full generated answers) is in `eval/results.json`.

## Test set design

Eight queries covering the query types a real user would ask, each with the
book(s) that *should* be retrieved, plus one deliberately off-topic query that
the system must refuse rather than hallucinate:

| # | Query type | Query | Expected source(s) |
|---|------------|-------|--------------------|
| 1 | Single-book factual | What does Atomic Habits say about habit stacking? | Atomic Habits |
| 2 | Concept lookup, no book named | How can I stop caring about other people's opinions? | Courage to Be Disliked / Four Agreements / Subtle Art |
| 3 | Cross-book comparison | Compare Rich Dad Poor Dad and The Psychology of Money on wealth | either book / money guide |
| 4 | Named-concept lookup | What is the 40% rule and who came up with it? | Can't Hurt Me |
| 5 | Multi-book synthesis | What morning routine do these books recommend? | Miracle Morning / Compound Effect / habits guide |
| 6 | Practical how-to | How do I do deep focused work without phone distraction? | Deep Work / focus guide |
| 7 | Recommendation | Which book should I read first if I procrastinate a lot? | Reader's Roadmap / Art of Laziness |
| 8 | Off-topic (graceful failure) | What is the recipe for chocolate cake? | — must be refused |

## Results

| # | Retrieval hit@4 | Top-1 document | Top score | Answer cites expected source | Retrieval | Generation |
|---|----------------|----------------|-----------|------------------------------|-----------|------------|
| 1 | ✅ | Atomic Habits | 0.674 | ✅ | 7,490 ms* | 9,545 ms |
| 2 | ✅ | The Four Agreements | 0.479 | ✅ | 81 ms | 17,737 ms |
| 3 | ✅ | Guide: Money & Wealth Mindsets | 0.764 | ✅ | 60 ms | 10,032 ms |
| 4 | ✅ | Can't Hurt Me | 0.232 | ✅ | 122 ms | 13,664 ms |
| 5 | ✅ | The Compound Effect | 0.542 | ✅ | 298 ms | 9,798 ms |
| 6 | ✅ | Deep Work | 0.610 | ✅ | 223 ms | 13,774 ms |
| 7 | ✅ | Guide: A Reader's Roadmap | 0.578 | ✅ | 163 ms | 10,812 ms |
| 8 | ✅ refused (0.166 < 0.20 floor) | — | 0.166 | — (no LLM call made) | 70 ms | — |

\* Query 1 includes one-time embedding-model load; warm retrieval is 60–300 ms.

**Summary: retrieval hit@4 = 7/7 · citation accuracy = 7/7 · graceful failure = 1/1.**

## Discussion — what worked

- **Semantic retrieval genuinely beats the TF-IDF baseline.** Query 2 never
  mentions any book, yet the embedding model maps "stop caring about other
  people's opinions" onto the right passages in three different books. Under
  TF-IDF this query mostly matches stop-words.
- **The thematic guide documents earn their place.** For comparison and
  recommendation queries (3, 7), a purpose-written cross-book guide outranks
  the individual books and gives the LLM a ready-made comparison to ground on.
- **Grounding held.** Every generated answer cited only books present in the
  retrieved context; the off-topic query never reached the LLM at all, so
  hallucination was structurally impossible for it.

## Discussion — what didn't work, and fixes applied during evaluation

- **Relevance floor was initially too strict.** With the floor at 0.30, query 4
  ("the 40% rule", score 0.232) was wrongly refused: short named-concept
  queries produce low cosine scores because the query has almost no context.
  Tuned the floor to 0.20, which keeps a clear margin above the off-topic
  score (0.166). With only one off-topic probe this margin is thin —
  a larger negative test set would be needed to tune it properly.
- **Answers were truncated mid-sentence.** The newer Gemini models spend part
  of the `max_tokens` budget on internal reasoning; a 1,000-token cap cut
  answers off (query 6 originally ended "…boredom [Deep"). Raised to 3,000.
- **First-run latency is dominated by model load** (~7 s) and each answer takes
  ~10–18 s of LLM time. Mitigations already in place: the embedding matrix is
  cached on disk and the Streamlit app caches the store, so only generation
  cost remains per query. A streaming response would improve perceived latency.

## Known limitations

- The corpus is original *study notes about* the books, not the books' full
  text — answers reflect the notes' coverage, not every page of every book.
- Similarity scores are not calibrated probabilities; the 0.20 floor is tuned
  on this corpus and would need re-tuning for a different document set.
- Single-language (English), single-turn: no conversation memory.
