"""
Evaluation harness for the Book Search Assistant.

Runs a fixed set of test queries through the full pipeline and reports:
- Retrieval: was the expected book among the top-k retrieved chunks (hit@k)?
  What was the top-1 document and the top similarity score?
- Generation: which engine answered, and does the answer cite the expected book?
- Graceful failure: the off-topic query must fall below the relevance floor.

Run:
    .venv/bin/python -m eval.run_eval          # retrieval only (free, fast)
    .venv/bin/python -m eval.run_eval --llm    # also test LLM generation
"""

import json
import sys
import time

from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer

TOP_K = 4

# Each case: the query, the book(s) that should be retrieved, and whether the
# query is answerable from the library at all.
CASES = [
    {
        "id": 1,
        "query": "What does Atomic Habits say about habit stacking?",
        "expected_any": ["Atomic Habits"],
        "answerable": True,
        "kind": "single-book factual",
    },
    {
        "id": 2,
        "query": "How can I stop caring so much about other people's opinions of me?",
        "expected_any": ["The Courage to Be Disliked", "The Four Agreements",
                          "The Subtle Art of Not Giving a F*ck"],
        "answerable": True,
        "kind": "concept lookup (no book named)",
    },
    {
        "id": 3,
        "query": "Compare what Rich Dad Poor Dad and The Psychology of Money say about building wealth.",
        "expected_any": ["Rich Dad Poor Dad", "The Psychology of Money", "Guide: Money & Wealth Mindsets"],
        "answerable": True,
        "kind": "cross-book comparison",
    },
    {
        "id": 4,
        "query": "What is the 40% rule and who came up with it?",
        "expected_any": ["Can't Hurt Me"],
        "answerable": True,
        "kind": "named-concept lookup",
    },
    {
        "id": 5,
        "query": "What morning routine do these books recommend?",
        "expected_any": ["The Miracle Morning", "The Compound Effect",
                          "Guide: Habit Formation Across the Library"],
        "answerable": True,
        "kind": "multi-book synthesis",
    },
    {
        "id": 6,
        "query": "How do I do deep focused work without getting distracted by my phone?",
        "expected_any": ["Deep Work", "Guide: Focus & Productivity Systems"],
        "answerable": True,
        "kind": "practical how-to",
    },
    {
        "id": 7,
        "query": "Which book should I read first if I procrastinate a lot?",
        "expected_any": ["Guide: A Reader's Roadmap", "The Art of Laziness",
                          "Guide: Focus & Productivity Systems"],
        "answerable": True,
        "kind": "recommendation",
    },
    {
        "id": 8,
        "query": "What is the recipe for chocolate cake?",
        "expected_any": [],
        "answerable": False,
        "kind": "off-topic (graceful failure)",
    },
]


def main() -> None:
    use_llm = "--llm" in sys.argv

    docs = load_documents("data/books")
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)
    print(f"Index: {len(docs)} docs, {len(chunks)} chunks, backend={store.backend}\n")

    results = []
    hits = 0
    graceful_ok = 0
    for case in CASES:
        t0 = time.perf_counter()
        retrieved = store.query(case["query"], top_k=TOP_K)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        top_titles = [c.doc_title for c, _ in retrieved]
        top_score = retrieved[0][1] if retrieved else 0.0
        relevant = [(c, s) for c, s in retrieved if s >= store.min_relevance]

        row = {
            "id": case["id"],
            "kind": case["kind"],
            "query": case["query"],
            "top1": top_titles[0] if top_titles else None,
            "top_score": round(top_score, 3),
            "retrieved_titles": top_titles,
            "retrieval_ms": round(retrieval_ms),
        }

        if case["answerable"]:
            hit = any(exp in top_titles for exp in case["expected_any"])
            hits += hit
            row["retrieval_hit"] = hit
        else:
            row["graceful_failure"] = not relevant
            graceful_ok += int(not relevant)

        if use_llm and relevant:
            t1 = time.perf_counter()
            answer, engine = generate_answer(case["query"], relevant, mode="llm")
            row["generation_ms"] = round((time.perf_counter() - t1) * 1000)
            row["engine"] = engine
            row["answer"] = answer
            if case["answerable"]:
                row["cites_expected"] = any(exp in answer for exp in case["expected_any"])

        results.append(row)
        status = "HIT " if row.get("retrieval_hit") else ("OK  " if row.get("graceful_failure") else "MISS")
        print(f"[{status}] Q{case['id']} ({case['kind']}) top1={row['top1']} score={row['top_score']}")

    answerable = [c for c in CASES if c["answerable"]]
    print(f"\nRetrieval hit@{TOP_K}: {hits}/{len(answerable)}")
    print(f"Graceful failures handled: {graceful_ok}/{len(CASES) - len(answerable)}")
    if use_llm:
        cited = sum(1 for r in results if r.get("cites_expected"))
        print(f"Answers citing an expected source: {cited}/{len(answerable)}")

    with open("eval/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nDetailed results written to eval/results.json")


if __name__ == "__main__":
    main()
