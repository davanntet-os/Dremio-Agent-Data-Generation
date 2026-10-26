# Vector DB / embeddings calibration findings

**Question:** would ChromaDB (or any vector DB) improve the data-generation
pipeline?

**What we tested (2026-10-26):**

| Model | Paraphrase ("revenue last month" → "made in previous month") | Different metric ("order amount" vs "shipping cost") | Same SQL, reworded | Verdict |
|---|---|---|---|---|
| ChromaDB default (onnx MiniLM) | 0.498 | 0.439 | 0.777 | gap too small |
| model2vec potion-base-8M | 0.474 | 0.816 (inverted!) | 0.861 | unusable |
| sentence-transformers all-MiniLM-L6-v2 | 0.749 | 0.719 | 0.888 | gap ~0.03 |

**Conclusion:** short analytical questions share so much surface structure
that embedding models cannot reliably separate "same question, reworded"
from "same shape, different metric". No threshold works (3 embedding models,
same result).

**What the pipeline keeps instead (deterministic, strictly better here):**

1. Normalized SQL fingerprints — catch reworded questions that compile to the
   same query exactly.
2. Lexical question fingerprints — catch most near-duplicate questions.
3. The GLM semantic reviewer — reads meaning, catches the paraphrase→slightly
   different SQL case.

**What we adopted from this work anyway:** diversity-guided generation —
each generation batch now receives the list of questions already in the
library and is instructed to produce different wording, metrics and filters.
This requires no embeddings and directly improves multi-day runs.

**The embedding layer remains available** as an experiment:
`pip install 'dataagent[semantic]'` and set `SEMANTIC_DEDUP_ENABLED=true`.
