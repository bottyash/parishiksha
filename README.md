# PariShiksha - NCERT Textbook Corpus Pipeline

> **Week 9 Assignment** - Taking a raw NCERT Science PDF and turning it into something a retrieval system can actually reason over. This covers ingestion, structured extraction, tokenizer comparison and chunking design.

---

## What We're Working With

| Item | Detail |
|------|--------|
| **Textbook** | NCERT Class 9 Science - *Matter in Our Surroundings* |
| **Chapters used** | Chapter 1 (`ch1.pdf`) |
| **Download** | [https://ncert.nic.in/textbook.php?iesc1=0-11](https://ncert.nic.in/textbook.php?iesc1=0-11) |
| **Local path** | `./pdfs/` - drop your downloaded PDFs here |

---

## How the Project is Laid Out

```
PariShiksha/
├── pdfs/                        # Input PDFs (ch1.pdf … ch12.pdf)
├── extracted_text/
│   ├── extracted_text.json      # Final structured extraction output (v4)
│   ├── extracted_text.txt       # Plain-text version for quick inspection
│   └── versions/                # v1-v4 incremental extraction outputs
├── chunk/
│   └── chunked_text.json        # Chunked output ready for retrieval
├── tokening/
│   └── tokenizer_analysis.txt   # GPT-2 / BERT / T5 comparison report
├── retrieval/
│   ├── cache/                   # Dense embedding matrix (.npy)
│   ├── results_v3.json          # BM25 structured results
│   └── dense_results_v3.json    # Dense structured results
├── StageDocumentation/
│   ├── Stage1.md                # Extraction dev log
│   └── Stage2.md                # Retrieval dev log (Lexical + Dense)
├── textextract.py               # Stage 1 - PDF extraction & structuring
├── tokening.py                  # Tokenizer comparison (GPT-2 / BERT / T5)
├── chunking.py                  # Overlapping chunking
├── retrieval.py                 # Stage 2a - BM25 lexical retrieval
├── dense_retrieval.py           # Stage 2b - MiniLM dense retrieval
├── hld.txt                      # High-level pipeline diagram
└── README.md
```

---

## The Big Picture

At its core, the pipeline takes a PDF and progressively shapes it into something useful:

```
NCERT PDF
  |
[Ingestion + OCR fallback]       textextract.py
  |
[Cleaning + Structuring]         extracted_text/extracted_text.json
  |
[Tokenizer Comparison]           tokening.py
  |
[Chunking + Metadata]            chunking.py
  |
[Lexical Retrieval (BM25)]       retrieval.py
  |
[Dense Retrieval (MiniLM)]       dense_retrieval.py
  |
[LLM (Grounded Generation)]      (Stage 3)
  |
[Evaluation Engine]              (Stage 4)
```

---

## Getting Started

### 1. Prerequisites

Python 3.10+ and `pip`. A virtual environment keeps things clean.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install pymupdf transformers tokenizers sentencepiece rank-bm25 sentence-transformers
```

> `rank-bm25` is needed for Stage 2a (Lexical). `sentence-transformers` is needed for Stage 2b (Dense). HuggingFace will pull down tokenizer and model weights automatically on first run ($~80MB$ for the MiniLM model).

### 3. Download the PDFs

Grab the chapters from:
[https://ncert.nic.in/textbook.php?iesc1=0-11](https://ncert.nic.in/textbook.php?iesc1=0-11)

Then drop them into `./pdfs/`:
```
pdfs/ch1.pdf
pdfs/ch2.pdf
```

### 4. Run the Pipeline

Run each script in order. Each one builds on the output of the previous.

---

#### Step 1 - Extract & Structure the PDF

```bash
python textextract.py
```

**Output:** `extracted_text/extracted_text.json`

This is v4 of the extractor, the most refined version. Instead of dumping raw text, it gives you structured blocks like:

```json
{
  "page": 3,
  "section": "1.1",
  "type": "concept",
  "text": "Matter is made up of particles..."
}
```

Every block is tagged as one of: `concept` | `activity` | `question`

**How the extractor evolved over time:**

| Version | File | Format | What Changed |
|---------|------|--------|--------------|
| v1 | `extracted_text/versions/v1.txt` | Plain text | Raw `page.get_text()` output, noisy with no structure |
| v2 | `extracted_text/versions/v2.txt` | Plain text | Regex cleaning to remove headers, footers and page numbers |
| v3 | `extracted_text/versions/v3.txt` | Plain text | Layout-aware blocks + sorted reading order |
| **v4** | `extracted_text/extracted_text.json` | **JSON** | Anchor-based segmentation, section tracking and content-type labels |

> The jump from v3 to v4 was the biggest leap, moving from "cleaner text" to "actually structured data."

---

#### Step 2 - Compare Tokenizers

```bash
python tokening.py
```

**Output:** `tokening/tokenizer_analysis.txt`

Runs GPT-2, BERT and T5 over five representative passages from the corpus to see how each handles the textbook's content: section headings, activity blocks and units like `mL`.

---

#### Step 3 - Chunk the Corpus

```bash
python chunking.py
```

**Output:** `chunk/chunked_text.json`

```json
{
  "chunk_id": 12,
  "page": 4,
  "text": "..."
}
```

---

### Stage 2 — Retrieval

We implement two distinct retrieval engines to compare their efficacy. Both scripts offer an interactive REPL when run from a terminal.

#### Stage 2a - BM25 Lexical Retrieval
```bash
python retrieval.py
```
**Output:** `retrieval/results_v3.txt` and `retrieval/results_v3.json`

Lexical retrieval excels at exact keyword matching.

#### Stage 2b - Dense Semantic Retrieval (MiniLM)
```bash
python dense_retrieval.py
```
**Output:** `retrieval/dense_results_v3.txt` and `retrieval/dense_results_v3.json`

Uses the bi-encoder `sentence-transformers/all-MiniLM-L6-v2`. Embeddings are cached to `retrieval/cache/corpus_embeddings.npy` on the first run, so subsequent runs are instant. Dense retrieval excels at semantic similarity (e.g. matching "evaporation cooling" to the correct conceptual paragraph even if the exact keyword overlaps are low).

---

**Interactive REPL usage (for both scripts):**

```
Query > evaporation cooling --concept
Query > dissolve salt beaker --activity
Query > stats
```

**How retrieval evolved:**

| Version | Output | Key Addition |
|---------|--------|--------------|
| v1 | `results_v1.txt` | Basic BM25 index + 5 demo queries |
| v2 | `results_v2.txt` | Metadata type filter + deduplication |
| **v3** | `results_v3.txt` + `results_v3.json` | Color output, term highlighting, REPL and JSON export |

**How dense retrieval evolved:**

| Version | File | Key Addition |
|---------|------|--------------|
| v1 | `dense_results_v1.txt` | `SentenceTransformer` encoded index + numpy cosine similarity array |
| v2 | `dense_results_v2.txt` | `.npy` disk cache so model evaluation only runs once |
| **v3** | `dense_results_v3.json` | Full feature parity with BM25: interactive REPL, JSON export, inline flags |

Full dev log: [`StageDocumentation/Stage2.md`](StageDocumentation/Stage2.md)

---

## Content-Type Classification

`textextract.py` classifies every extracted block into one of three types:

| Type | Detection Rule | Example |
|------|---------------|---------|
| `activity` | Line matches `Activity \d+\.\d+` | *"Activity 1.1 Take a 100 mL beaker..."* |
| `question` | Line contains `Questions` / `uestions` | *"Questions 1. Convert 300 K to..."* |
| `concept` | All other prose | *"Matter is made up of particles that are very small..."* |

Anchor-based segmentation starts a new block on every section heading (`1.1`, `1.2.3`), every `Activity` and every `Questions` marker. This prevents over-merging and over-fragmentation.

---

## Tokenizer Analysis

Five passages (at least one from each content type) were tokenised with all three tokenizers.

| Sample | GPT-2 BPE | BERT WordPiece | T5 SentencePiece |
|--------|-----------|----------------|-----------------|
| Concept text | 234 | **227** | 239 |
| Section heading | 79 | **73** | 82 |
| Activity block | 72 | 81 | 79 |
| Question | 50 | **47** | **47** |
| Mixed / figure content | 112 | **110** | 128 |

**Where the tokenizers disagree:**

| Tokenizer | Sub-word marker | Behaviour |
|-----------|----------------|-----------|
| GPT-2 BPE | `Ġ` (space prefix) | Aggressive splitting; numeric sections like `1.1.1` fragment into `1`, `.`, `1`, `.`, `1` |
| BERT WordPiece | `##` (continuation) | Lowercases input; compact on section headings; underscores expanded individually |
| T5 SentencePiece | `▁` (word-start) | More natural word boundaries but balloons on mixed units (`mL` becomes `m`, `L`) |

**Recommended tokenizer: BERT (`bert-base-uncased`)**

- Consistently lowest token counts, enabling larger meaningful chunks within model limits
- Lowercase normalisation handles noisy ALL-CAPS OCR output cleanly
- Best task alignment for BM25 retrieval and grounded QA (Stage 2 and 3)

> **Note:** If the downstream model switches to T5 or FLAN-T5, the tokenizer must switch too. Tokenizer and model always need to belong to the same family.

---

## Chunking Strategy

**Parameters:** `chunk_size = 300 words`, `overlap = 50 words`

| Decision | Justification |
|----------|--------------|
| **300-word chunks** | Balances retrieval precision and context sufficiency; fits comfortably within the 512-token BERT limit after tokenisation |
| **50-word overlap (~17%)** | Prevents boundary artefacts when a concept spans two consecutive chunks; improves recall for cross-boundary queries |
| **Content-type handling** | `activity` and `question` blocks that fit within the budget are kept whole; long blocks get their own chunk sequence with overlap resetting at each semantic anchor |
| **Structured JSON output** | Preserves `page`, `section` and `type` metadata per chunk for traceable retrieval |

> Chunking quality has a greater impact on downstream QA performance than model selection.

---

## Extraction Phases (Summary)

| Phase | Change | Outcome |
|-------|--------|---------|
| 1 - Raw extraction | `page.get_text()` via PyMuPDF | Raw noisy text |
| 2 - Cleaning | Regex removal of headers, footers and page numbers | Cleaner text |
| 3 - Layout awareness | `page.get_text("blocks")` sorted by position | Better paragraph grouping |
| 4 - Structured parsing | Anchor-based segmentation and state-based section tracking | Semantic blocks with `type` and `section` |
| 5 - OCR fallback | Tesseract on low-quality pages | Handles scanned and diagram pages |
| 6 - Chunking | Overlapping word-level windows | Retrieval-ready chunks |
| 7 - Tokenizer analysis | GPT-2, BERT and T5 compared on corpus samples | BERT selected |

Full development log: [`StageDocumentation/Stage1.md`](StageDocumentation/Stage1.md)

---

## Key Challenges and Resolutions

| # | Challenge | Resolution |
|---|-----------|------------|
| 1 | Multiple sections per page | State-based section tracking |
| 2 | Activities embedded in concept text | Hard anchor split on `Activity X.X` |
| 3 | False section detection (`100`, `1000`) | Strict regex `^\d+\.\d+(\.\d+)*` |
| 4 | OCR noise and repeated artefacts | Regex cleaning + `is_noise()` filter |
| 5 | Over-splitting vs over-merging | Semantic anchors instead of line breaks |

---

## Output Files

| File | Description |
|------|-------------|
| `extracted_text/extracted_text.json` | Structured blocks with `page`, `section`, `type` and `text` |
| `extracted_text/extracted_text.txt` | Plain-text version for quick inspection |
| `extracted_text/versions/` | Extraction outputs v1-v4 |
| `chunk/chunked_text.json` | Overlapping word-level chunks with metadata |
| `tokening/tokenizer_analysis.txt` | Token counts and first 25 tokens per sample per tokenizer |
| `retrieval/results_v3.json` | Structured BM25 results (query to top-5 blocks with metadata) |
| `retrieval/dense_results_v3.json` | Structured MiniLM dense results |
| `retrieval/cache/corpus_embeddings.npy` | Cached 384-dimensional dense vectors |
| `StageDocumentation/Stage1.md` | Extraction iterative development log |
| `StageDocumentation/Stage2.md` | Retrieval (BM25 + Dense) iterative log |

---

## Stages

| Stage | Status | Description |
|-------|--------|-------------|
| Stage 1 | Complete | PDF extraction, structuring, tokenizer analysis and chunking |
| Stage 2a | Complete | BM25 lexical retrieval with metadata filtering |
| Stage 2b | Complete | Dense bi-encoder retrieval (MiniLM) with numpy cache |
| Stage 3 | Planned | Grounded QA with an LLM |
| Stage 4 | Planned | Evaluation Engine (precision, recall and faithfulness) |