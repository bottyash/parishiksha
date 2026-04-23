# Stage 2: BM25 Retrieval
## Development Log

---

## 📌 Objective

Build a **lexical retrieval engine** over the structured NCERT corpus from Stage 1 using BM25 (Best Match 25), enabling relevant block retrieval given a natural-language query.

---

## 🔁 Iterative Development

### Iteration 1 — Basic BM25 (`results_v1.txt`)

**What was built:**
- Loaded `extracted_text.json`, filtered blocks with < 5 words
- Tokenized using lowercase + regex: `re.findall(r"[a-z0-9]+", text.lower())`
- Built `BM25Okapi` index via `rank-bm25`
- Ran 5 demo queries, saved top-5 results per query to plain text

**Issues observed:**
- Duplicate blocks appeared (same section heading split across pages)
- No way to target specific content types (concept vs activity vs question)

---

### Iteration 2 — Metadata Filtering + Deduplication (`results_v2.txt`)

**What was built:**
- Wrapped BM25 logic in a `BM25Retriever` class
- Added `content_type` filter parameter: optionally restrict to `concept`, `activity`, or `question`
- Added deduplication: skip results whose first 80 chars match an already-seen result
- Stopped early when `score == 0` (no more relevant results)
- Pretty-printed results with score, page, section, type

**Improvement observed:**
- Filtering `activity` blocks correctly surfaced lab-activity blocks for "dissolve salt in water"
- Concept-filtered queries stayed cleanly within explanatory prose

---

### Iteration 3 — Interactive CLI + JSON Export (`results_v3.txt`, `results_v3.json`)

**What was built:**
- ANSI colour output: green=concept, yellow=activity, cyan=question
- Query-term **bold highlighting** in result snippets
- `--concept` / `--activity` / `--question` flags parsed inline from query string
- `stats` command showing corpus breakdown by type
- Structured **JSON export** of all demo results (`results_v3.json`)
- Interactive REPL exits cleanly when stdin is not a terminal (piped runs)

---

## 📊 Corpus Stats

| Type | Blocks |
|------|--------|
| concept | 62 |
| activity | 17 |
| question | 13 |
| **Total** | **92** |

---

## 🔍 Sample Results (Iteration 3)

### Query: "evaporation causes cooling explain" `[filter: concept]`

| Rank | Score | Page | Section | Excerpt |
|------|-------|------|---------|---------|
| 1 | 7.32 | 9 | 1.5.2 | *"In an open vessel, the liquid keeps on evaporating. The particles of liquid absorb energy from the surrounding…"* |
| 2 | 5.91 | 9 | 1.5.1 | *"…a small fraction of particles at the surface, having higher kinetic energy, is able to break away…"* |
| 3 | 4.87 | 9 | 1.5.2 | *"…rate of evaporation increases with an increase of surface area…"* |

### Query: "activity dissolve salt in water" `[filter: activity]`

| Rank | Score | Page | Section | Excerpt |
|------|-------|------|---------|---------|
| 1 | 8.14 | 1 | 1.1.1 | *"Activity 1.1 Take a 100 mL beaker. Fill half the beaker with water… Dissolve some salt/sugar…"* |

---

## 🧠 Design Decisions

| Decision | Justification |
|----------|--------------|
| `BM25Okapi` variant | Okapi BM25 includes IDF saturation — handles common words better than plain BM25 |
| Tokenize with `[a-z0-9]+` | Strips punctuation and normalises units (`mL`, `CO2`) without losing content |
| Deduplication on first 80 chars | Prevents repeated section headings (e.g., `"1.3"` appearing on 4 pages) dominating results |
| Type filter optional | Allows unrestricted search by default; targeted filter when content type is known |
| JSON export | Structured output enables direct piping into Stage 3 (Grounded QA) |

---

## ⚠️ Limitations

- Pure lexical: no semantic understanding — query "boiling" won't match "vaporisation" unless both terms appear
- Section metadata in corpus has some noise (`"100"`, `null`) from OCR — does not affect retrieval but clutters output
- No query expansion or stemming yet

---

## 🚀 Next: Stage 3 — Grounded QA

Pipe `results_v3.json` top-k results as context into an LLM to generate grounded, faithful answers with source attribution.
