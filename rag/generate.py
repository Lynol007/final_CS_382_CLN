"""
Generation: turn retrieved chunks + a query into a final, grounded answer.

Fallback chain (design decision — the demo must never break):
1. LLM via any OpenAI-compatible API (this project uses Google Gemini),
   configured in .env. Includes retry + fallback models for transient errors.
2. Extractive — always works: shows the retrieved passages directly.

Grounding rules enforced in the prompt:
- Answer ONLY from the retrieved sources; cite book titles in [brackets].
- If the sources don't answer the question, say so instead of guessing.
"""

import os
import time
from typing import List, Optional, Tuple

from .ingest import Chunk

_SYSTEM_PROMPT = (
    "You are a Book Search Assistant answering questions about a library of "
    "self-improvement book study notes. Answer the user's question using ONLY "
    "the sources provided. Write a thorough, helpful answer of two to four "
    "short paragraphs (or a bulleted list if the question asks for steps): "
    "explain the idea, how it works in practice, and any nuances the sources "
    "mention. Cite every claim with the BOOK TITLE in square brackets, e.g. "
    "[Atomic Habits] or [Rich Dad Poor Dad]. Never cite by number — do not "
    "write [Source 1] or [1]; always use the book's title. If several books "
    "are relevant, compare their views. If the sources do not contain enough "
    "information to answer, say exactly that — do not use outside knowledge "
    "or guess."
)


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) — avoids an extra dependency."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _format_context(retrieved: List[Tuple[Chunk, float]]) -> str:
    # Sources are labelled by book title only (no numbering) so the model
    # cannot fall back to citing "[Source 1]"-style references.
    lines = []
    for chunk, _ in retrieved:
        lines.append(f'From the book "{chunk.doc_title}" by {chunk.author}:\n{chunk.text}')
    return "\n\n".join(lines)


def _build_user_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    return f"Sources:\n\n{_format_context(retrieved)}\n\nQuestion: {query}"


# ---------- answer modes ----------

def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """No-LLM fallback: present the retrieved passages directly."""
    if not retrieved:
        return "No relevant passages were found for that query."
    lines = [f"Top passages related to: “{query}”\n"]
    for chunk, score in retrieved:
        lines.append(f"**[{chunk.doc_title}]** (similarity {score:.2f}) {chunk.text}\n")
    return "\n".join(lines)


def openai_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> Optional[str]:
    """Grounded answer via any OpenAI-compatible API (OpenAI, Gemini, etc.).

    Provider is configured entirely in .env: OPENAI_API_KEY, plus optional
    OPENAI_BASE_URL (e.g. Gemini's compatibility endpoint) and OPENAI_MODEL.
    Returns None if no API key is set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL") or None)
    # Primary model plus fallbacks: transient 503s ("model overloaded") on one
    # model usually don't affect its siblings, so trying a second model keeps
    # the live demo running through provider load spikes.
    models = [os.environ.get("OPENAI_MODEL", "gpt-4o-mini")]
    fallbacks = os.environ.get("OPENAI_FALLBACK_MODELS", "")
    models += [m.strip() for m in fallbacks.split(",") if m.strip()]

    last_error: Exception = RuntimeError("no model configured")
    for attempt, model in enumerate(m for m in models for _ in range(2)):
        try:
            response = client.chat.completions.create(
                model=model,
                # Generous budget: newer Gemini models spend tokens on internal
                # reasoning before the visible answer, so a tight cap truncates.
                max_tokens=3000,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(query, retrieved)},
                ],
            )
            return response.choices[0].message.content
        except Exception as exc:  # retry transient overload/rate-limit errors
            last_error = exc
            if not any(code in str(exc) for code in ("503", "429", "500", "overloaded", "UNAVAILABLE")):
                raise
            time.sleep(1.5 * (attempt + 1))
    raise last_error


def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "llm") -> Tuple[str, str]:
    """Produce (answer_text, engine_used).

    mode="llm": try the configured LLM, then fall back to extractive.
    mode="extractive": skip the LLM entirely.
    """
    if not retrieved:
        return ("No relevant passages were found for that query.", "none")

    if mode == "llm":
        try:
            answer = openai_answer(query, retrieved)
            if answer:
                base_url = os.environ.get("OPENAI_BASE_URL", "")
                return (answer, "gemini" if "googleapis" in base_url else "openai")
        except Exception as exc:  # API failure must not crash the demo
            print(f"[generate] LLM failed, falling back to passages: {exc}")

        return (
            "_The AI answer service is unavailable right now (check OPENAI_API_KEY in .env "
            "or your connection) — showing retrieved passages instead._\n\n"
            + extractive_answer(query, retrieved),
            "extractive-fallback",
        )

    return (extractive_answer(query, retrieved), "extractive")
