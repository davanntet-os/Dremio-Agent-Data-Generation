"""Embedding-based semantic near-duplicate detection (optional layer).

Uses a local persistent vector index (ChromaDB when installed; graceful
no-op otherwise). Complements — never replaces — the deterministic
lexical/SQL fingerprinting:

  layer 1: exact/normalized SQL fingerprint (deterministic)
  layer 2: lexical question fingerprint (deterministic)
  layer 3: embedding cosine similarity (this module) — catches paraphrases
           like "What was our revenue last month?" vs
           "How much did we make in the previous month?"

Also used for diversity-guided generation: the question engine retrieves
recent existing questions and is instructed to produce questions on
different angles.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataagent.config import DATA_DIR
from dataagent.logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.90


@dataclass
class SemanticMatch:
    similar: bool
    score: float
    matched_example_id: str | None = None


class SemanticDedup:
    """Persistent embedding index of accepted questions."""

    def __init__(self, threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 persist_dir=None) -> None:
        self._threshold = threshold
        self._available = False
        try:
            import chromadb
        except ImportError:
            log.info("chromadb not installed; semantic near-dup detection disabled "
                     "(pip install 'dataagent[semantic]')")
            self._collection = None
            return
        client = chromadb.PersistentClient(path=str(persist_dir or DATA_DIR / "semantic_index"))
        self._collection = client.get_or_create_collection("questions")
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def count(self) -> int:
        return self._collection.count() if self._available else 0

    def add(self, question: str, example_id: str) -> None:
        if not self._available or not question:
            return
        try:
            self._collection.add(
                ids=[example_id],
                documents=[question],
            )
        except Exception as exc:  # duplicate id etc.
            log.debug("semantic index add skipped: %s", exc)

    def check(self, question: str, n: int = 1) -> SemanticMatch:
        """Return the closest stored question and whether it is too similar."""
        if not self._available or self.count() == 0:
            return SemanticMatch(False, 0.0)
        try:
            res = self._collection.query(query_texts=[question], n_results=min(n, self.count()))
        except Exception as exc:
            log.warning("semantic query failed: %s", exc)
            return SemanticMatch(False, 0.0)
        docs = (res.get("documents") or [[]])[0]
        scores = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        if not docs or not scores:
            return SemanticMatch(False, 0.0)
        # Chroma returns distances; cosine similarity = 1 - distance
        similarity = 1.0 - float(scores[0])
        return SemanticMatch(similar=similarity >= self._threshold,
                             score=similarity,
                             matched_example_id=ids[0] if ids else None)

    def most_similar_questions(self, question: str, k: int = 5) -> list[str]:
        """k nearest stored questions (for diversity-aware prompting)."""
        if not self._available or self.count() == 0:
            return []
        try:
            res = self._collection.query(
                query_texts=[question], n_results=min(k, self.count()))
        except Exception:
            return []
        return list((res.get("documents") or [[]])[0])
