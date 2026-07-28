"""
Ingestion: load the book-notes corpus from disk and split it into
retrievable chunks.

Design decisions (see README):
- Documents are plain .txt files; a catalog.json beside them supplies
  title/author/topic metadata so citations can show real book names.
- Chunking is sentence-aware: sentences are packed into chunks up to a
  word budget, with a 2-sentence overlap between consecutive chunks so
  no idea is cut mid-thought at a chunk boundary. This is more defensible
  than fixed word windows, which split sentences arbitrarily.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List

# Sentences end with . ! or ? followed by whitespace and an uppercase/quote/digit.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])")


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    author: str
    text: str


def load_documents(folder: str) -> List[dict]:
    """Load every .txt file in `folder`, enriched with catalog.json metadata."""
    catalog_path = os.path.join(folder, "catalog.json")
    catalog = {}
    if os.path.exists(catalog_path):
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

    docs = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        meta = catalog.get(filename, {})
        title = meta.get("title") or os.path.splitext(filename)[0].replace("_", " ").title()
        docs.append(
            {
                "filename": filename,
                "title": title,
                "author": meta.get("author", "Unknown"),
                "topic": meta.get("topic", ""),
                "kind": meta.get("kind", "document"),
                "text": text,
            }
        )
    return docs


def split_sentences(text: str) -> List[str]:
    """Split text into sentences (newline blocks count as boundaries too)."""
    sentences = []
    for block in re.split(r"\n{2,}", text):
        block = " ".join(block.split())  # collapse internal whitespace
        if not block:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_RE.split(block) if s.strip())
    return sentences


def chunk_text(text: str, max_words: int = 180, overlap_sentences: int = 2) -> List[str]:
    """Pack whole sentences into chunks of up to `max_words` words.

    Consecutive chunks share `overlap_sentences` sentences so that a thought
    spanning a boundary is still retrievable from either side.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current: List[str] = []
    current_words = 0
    for sentence in sentences:
        n_words = len(sentence.split())
        if current and current_words + n_words > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(s.split()) for s in current)
        current.append(sentence)
        current_words += n_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunk_records(docs: List[dict], max_words: int = 180, overlap_sentences: int = 2) -> List[Chunk]:
    """Turn loaded documents into a flat list of Chunk records ready for embedding."""
    records = []
    for doc in docs:
        pieces = chunk_text(doc["text"], max_words=max_words, overlap_sentences=overlap_sentences)
        for i, piece in enumerate(pieces):
            records.append(
                Chunk(
                    chunk_id=f"{doc['filename']}::{i}",
                    doc_title=doc["title"],
                    author=doc["author"],
                    text=piece,
                )
            )
    return records
