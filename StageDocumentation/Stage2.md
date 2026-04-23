# Stage 2: Lexical and Dense Retrieval
## Development Log

---

## 📌 Objective

Build retrieval engines over the structured NCERT corpus from Stage 1. 
- **Stage 2a:** Lexical retrieval using BM25
- **Stage 2b:** Dense retrieval using `sentence-transformers/all-MiniLM-L6-v2` (Bi-encoder)

---

## 🔁 Stage 2a — Lexical Retrieval (BM25)

*Iterations 1, 2, and 3 implemented `retrieval.py`.*

| Iter | Key Feature | Output |
|------|-------------|--------|
| **1** | Built `BM25Okapi` index over 92 corpus blocks. | `results_v1.txt` |
| **2** | Added `content_type` filtering to target specific sections. Dropped duplicates (first 80 chars). | `results_v2.txt` |
| **3** | Added interactive terminal REPL with color highlighting and inline type flags. JSON export added. | `results_v3.json` |

**Pros:** Fast, exact keyword matching (especially good for specific terms like "Panch Tatva").
**Cons:** Misses semantic overlap (e.g. "boiling" vs "vaporisation").

---

## 🔁 Stage 2b — Dense Retrieval (MiniLM)

*Iterations 4, 5, and 6 implemented `dense_retrieval.py` to overcome lexical limitations.*

### Iteration 1 — MiniLM Base (`dense_results_v1.txt`)
- Loaded `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings).
- Normalised vectors and used numpy matrix multiplication for exact cosine similarity search.
- Successfully matched semantic queries: "evaporation cooling" mathematically aligned with the text block *"In an open vessel, the liquid keeps on evaporating. The particles ... absorb energy ... make the surroundings cold."* at 0.74 cosine sim. 

### Iteration 2 — Embedding Cache (`dense_results_v2.txt`)
- Model encoding on a CPU takes a few seconds. Added an `np.save` embedding cache at `retrieval/cache/corpus_embeddings.npy`.
- Encoding now only happens on the first run; subsequent runs load the numeric cache instantly.
- *(Note: Initially attempted FAISS `IndexFlatIP`, but the tiny corpus size triggered a Windows paging allocation error. Reverted to pure numpy which is entirely sufficient for <1000 blocks).*

### Iteration 3 — Interactive REPL & Export (`dense_results_v3.json`)
- Brought dense retrieval to feature-parity with BM25.
- Added color output, structured JSON generation, and a terminal search loop.

---

## 📊 Lexical vs Dense Performance (Observation)

| Query | BM25 Top Match | MiniLM Top Match |
|-------|----------------|------------------|
| *"evaporation causes cooling"* | Score: 7.32 (Hits exact words) | Score: 0.74 (Hits semantic intent) |
| *"latent heat"* | Score: 6.50 (Finds definition chunk) | Score: 0.65 (Finds definition chunk) |

**Conclusion:** Both are viable. Dense retrieval is more resilient to paraphrasing, but BM25 is better at exact terminology lookup. A hybrid approach (Reciprocal Rank Fusion) would be ideal if precision requirements increase.

---

## 🚀 Next: Stage 3 — Grounded QA

Pipe `results_v3.json` or `dense_results_v3.json` top-k results as context into a generative LLM (like T5) to provide grounded answers with source attribution.
