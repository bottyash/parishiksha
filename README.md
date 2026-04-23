# PariShiksha — NCERT Textbook Corpus Pipeline

> **Week 9 Assignment** — PDF ingestion, structured extraction, tokenizer comparison, and chunking design for an NCERT Science textbook.

---

## 📚 Source Material

| Item | Detail |
|------|--------|
| **Textbook** | NCERT Class 9 Science — *Matter in Our Surroundings* |
| **Chapters used** | Chapter 1 (`ch1.pdf`) |
| **Download** | [https://ncert.nic.in/textbook.php?iesc1=0-11](https://ncert.nic.in/textbook.php?iesc1=0-11) |
| **Local path** | `./pdfs/` (place downloaded PDFs here) |

---

## 🗂️ Project Structure

```
PariShiksha/
├── pdfs/                        # Input PDFs (ch1.pdf … ch12.pdf)
├── extracted_text/
│   ├── extracted_text.json      # Structured extraction output (v4)
│   ├── extracted_text.txt       # Plain-text extraction output
│   └── versions/                # v1–v4 incremental extraction outputs
├── chunk/
│   └── chunked_text.json        # Chunked output ready for retrieval
├── tokening/
│   └── tokenizer_analysis.txt   # GPT-2 / BERT / T5 comparison report
├── retrieval/
│   ├── results_v1.txt           # Basic BM25 results
│   ├── results_v2.txt           # Filtered + deduplicated results
│   ├── results_v3.txt           # Final plain-text results
│   └── results_v3.json          # Structured JSON results
├── StageDocumentation/
│   ├── Stage1.md                # Extraction dev log
│   └── Stage2.md                # Retrieval dev log
├── textextract.py               # Stage 1 — PDF extraction & structuring
├── tokening.py                  # Tokenizer comparison (GPT-2 / BERT / T5)
├── chunking.py                  # Overlapping chunking
├── retrieval.py                 # Stage 2 — BM25 retrieval
├── hld.txt                      # High-level pipeline diagram
└── README.md
```

---

## 🔄 Pipeline Overview

```
NCERT PDF
   ↓
[Ingestion + OCR fallback]       →  textextract.py
   ↓
[Cleaning + Structuring]         →  extracted_text/extracted_text.json
   ↓
[Tokenizer Comparison]           →  tokening.py
   ↓
[Chunking + Metadata]            →  chunking.py
   ↓
[Lexical Retrieval (BM25)]  ✅   →  retrieval.py
   ↓
[LLM (Grounded Generation)]      →  (Stage 3)
   ↓
[Evaluation Engine]              →  (Stage 4)
```

---

## 🚀 Quick Start

### 1. Prerequisites

Python 3.10+ and `pip`. A virtual environment is recommended.

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install pymupdf transformers tokenizers sentencepiece rank-bm25
```

> `rank-bm25` is required for Stage 2 retrieval. HuggingFace will auto-download tokenizer weights on first run.

### 3. Download PDFs

Download the NCERT textbook chapters from:
[https://ncert.nic.in/textbook.php?iesc1=0-11](https://ncert.nic.in/textbook.php?iesc1=0-11)

Place the downloaded files in the `./pdfs/` directory:
```
pdfs/ch1.pdf
pdfs/ch2.pdf
```

### 4. Run the Pipeline

Run each script **in order**:

#### Step 1 — Extract & Structure PDF Text
```bash
python textextract.py
```
**Output:** `extracted_text/extracted_text.json`  
**Current version:** v4 (JSON, section-aware) — see version history below.

Each entry in the JSON has the form:
```json
{
  "page": 3,
  "section": "1.1",
  "type": "concept",
  "text": "Matter is made up of particles..."
}
```
Content types: `concept` | `activity` | `question`

#### Extraction Version History

| Version | File | Format | Key Change |
|---------|------|--------|------------|
| v1 | `extracted_text/versions/v1.txt` | Plain text | Raw `page.get_text()` output — noisy, no structure |
| v2 | `extracted_text/versions/v2.txt` | Plain text | Added regex cleaning (headers, footers, page numbers) |
| v3 | `extracted_text/versions/v3.txt` | Plain text | Layout-aware `get_text("blocks")` + sorted reading order |
| **v4** ✅ | `extracted_text/extracted_text.json` | **JSON** | Anchor-based segmentation, section tracking, content-type labelling |

> `textextract.py` currently produces **v4** — the final structured JSON with `page`, `section`, `type`, and `text` fields per block.



---

#### Step 2 — Tokenizer Comparison
```bash
python tokening.py
```
**Output:** `tokening/tokenizer_analysis.txt`

Compares **GPT-2 BPE**, **BERT WordPiece**, and **T5 SentencePiece** on 5 representative passages drawn from the extracted corpus.

---

#### Step 3 — Chunking
```bash
python chunking.py
```
**Output:** `chunk/chunked_text.json`

Each chunk has the form:
```json
{
  "chunk_id": 12,
  "page": 4,
  "text": "..."
}
```

---

#### Step 4 — BM25 Retrieval ✅
```bash
python retrieval.py
```
**Output:** `retrieval/results_v3.txt`, `retrieval/results_v3.json`

Runs 5 demo queries against the BM25 index and saves results. If run from a terminal, also starts an **interactive search REPL**.

**Inline type filters:**
```
🔍 Query > evaporation cooling --concept
🔍 Query > dissolve salt beaker --activity
🔍 Query > stats
```

**Retrieval version history:**

| Version | Output | Key Addition |
|---------|--------|--------------|
| v1 | `results_v1.txt` | Basic BM25 index + 5 demo queries |
| v2 | `results_v2.txt` | Metadata type filter + deduplication |
| **v3** ✅ | `results_v3.txt` + `results_v3.json` | Color output, term highlighting, REPL, JSON export |

Full dev log: [`StageDocumentation/Stage2.md`](StageDocumentation/Stage2.md)

---

## 📐 Content-Type Classification

`textextract.py` classifies every extracted block into one of three types:

| Type | Detection Rule | Example |
|------|---------------|---------|
| `activity` | Line matches `Activity \d+\.\d+` | *"Activity 1.1 Take a 100 mL beaker…"* |
| `question` | Line contains `Questions` / `uestions` | *"Questions 1. Convert 300 K to…"* |
| `concept` | All other prose | *"Matter is made up of particles that are very small…"* |

**Anchor-based segmentation:** a new block is started on every section heading (`1.1`, `1.2.3`), every `Activity`, and every `Questions` anchor — preventing over-merging and over-fragmentation.

---

## 🔤 Tokenizer Analysis

Five passages (at least one from each content type) were tokenised with all three tokenizers. Results from `StageDocumentation/Stage1.md`:

| Sample | GPT-2 BPE | BERT WordPiece | T5 SentencePiece |
|--------|-----------|----------------|-----------------|
| Concept text | 234 | **227** | 239 |
| Section heading | 79 | **73** | 82 |
| Activity block | 72 | 81 | 79 |
| Question | 50 | **47** | **47** |
| Mixed / figure content | 112 | **110** | 128 |

### Where Boundaries Disagree

| Tokenizer | Sub-word marker | Behaviour |
|-----------|----------------|-----------|
| GPT-2 BPE | `Ġ` (space prefix) | Aggressive splitting; numeric sections (`1.1.1`) fragmented into `1`, `.`, `1`, `.`, `1` |
| BERT WordPiece | `##` (continuation) | Lowercases input; compact on section headings; underscore tokens expanded individually |
| T5 SentencePiece | `▁` (word-start) | More natural word boundaries but balloons on mixed units (`mL` → `m`, `L`) |

### ✅ Recommended Tokenizer: BERT (`bert-base-uncased`)

- Consistently lowest token counts — enables larger meaningful chunks within model limits
- Lowercase normalisation handles noisy ALL-CAPS OCR output cleanly
- Best task alignment for BM25 retrieval and grounded QA (Stage 2 & 3)

> **Caveat:** If the downstream model switches to T5/FLAN-T5, the tokenizer must also switch — tokenizer and model must always belong to the same family.

---

## ✂️ Chunking Strategy

**Parameters:** `chunk_size = 300 words`, `overlap = 50 words`

| Decision | Justification |
|----------|--------------|
| **300-word chunks** | Balances retrieval precision and context sufficiency; fits comfortably within 512-token BERT limit after tokenisation |
| **50-word overlap (~17%)** | Prevents boundary artefacts when a concept spans two consecutive chunks; improves recall for cross-boundary queries |
| **Content-type handling** | `activity` and `question` blocks that fit within the budget are kept whole; long blocks receive their own chunk sequence with overlap resetting at each semantic anchor |
| **Structured JSON output** | Preserves `page`, `section`, and `type` metadata per chunk for traceable retrieval |

> **Key insight:** Chunking quality has a greater impact on downstream QA performance than model selection.

---

## 🧹 Extraction Phases (Summary)

| Phase | Change | Outcome |
|-------|--------|---------|
| 1 — Raw extraction | `page.get_text()` via PyMuPDF | Raw noisy text |
| 2 — Cleaning | Regex removal of headers, footers, page numbers | Cleaner text |
| 3 — Layout awareness | `page.get_text("blocks")` sorted by position | Better paragraph grouping |
| 4 — Structured parsing | Anchor-based segmentation, state-based section tracking | Semantic blocks with `type` and `section` |
| 5 — OCR fallback | Tesseract on low-quality pages | Handles scanned/diagram pages |
| 6 — Chunking | Overlapping word-level windows | Retrieval-ready chunks |
| 7 — Tokenizer analysis | GPT-2 / BERT / T5 compared on corpus samples | BERT selected |

Full development log: [`StageDocumentation/Stage1.md`](StageDocumentation/Stage1.md)

---

## ⚠️ Key Challenges & Resolutions

| # | Challenge | Resolution |
|---|-----------|------------|
| 1 | Multiple sections per page | State-based section tracking |
| 2 | Activities embedded in concept text | Hard anchor split on `Activity X.X` |
| 3 | False section detection (`100`, `1000`) | Strict regex `^\d+\.\d+(\.\d+)*` |
| 4 | OCR noise and repeated artefacts | Regex cleaning + `is_noise()` filter |
| 5 | Over-splitting vs over-merging | Semantic anchors instead of line breaks |

---

## 📦 Output Files

| File | Description |
|------|-------------|
| `extracted_text/extracted_text.json` | Structured blocks with `page`, `section`, `type`, `text` |
| `extracted_text/extracted_text.txt` | Plain-text version for quick inspection |
| `extracted_text/versions/` | Extraction outputs v1–v4 |
| `chunk/chunked_text.json` | Overlapping word-level chunks with metadata |
| `tokening/tokenizer_analysis.txt` | Token counts and first-25 tokens per sample per tokenizer |
| `retrieval/results_v3.json` | Structured BM25 results (query → top-5 blocks with metadata) |
| `StageDocumentation/Stage1.md` | Extraction iterative development log |
| `StageDocumentation/Stage2.md` | Retrieval iterative development log |

---

## 🔮 Stages

| Stage | Status | Description |
|-------|--------|-------------|
| Stage 1 | ✅ Complete | PDF extraction, structuring, tokenizer analysis, chunking |
| Stage 2 | ✅ Complete | BM25 lexical retrieval with metadata filtering |
| Stage 3 | 🔜 Planned | Grounded QA with an LLM |
| Stage 4 | 🔜 Planned | Evaluation Engine (precision, recall, faithfulness) |
