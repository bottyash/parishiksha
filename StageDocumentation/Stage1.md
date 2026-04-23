# Stage 1: Corpus Preparation
## Consolidated Development Log

---

## 📌 Objective

The goal of this stage was to convert an NCERT textbook PDF into a **clean, structured, and retrieval-ready corpus**.

Unlike versioned development, all improvements were made **incrementally in a single evolving script**, with each change addressing a specific limitation observed in previous outputs.

---

## 🔁 Development Approach

Instead of maintaining separate files for each version, the pipeline was **iteratively refined in-place**.  
Each improvement was tested, evaluated, and then integrated into the same script.

This document captures the **chronological evolution of the system**.

---

## 🧱 Phase 1: Initial Extraction

### Implementation
- Used PyMuPDF (`fitz`)
- Extracted text using:

```python
page.get_text()
```

### Output
Raw page-wise text

### Issues Identified
- Included:
  - Headers (`MATTER`, `SCIENCE`)
  - Footers (`Reprint 2025-26`)
  - Page numbers
- Broken formatting
- No separation of content types

### Learning
> Raw PDF extraction is noisy and unsuitable for downstream tasks.

---

## 🧹 Phase 2: Text Cleaning

### Changes Implemented
- Added regex-based cleaning:
  - Removed reprint lines
  - Removed known header words
  - Normalized whitespace

### Improvements
- Cleaner text
- Reduced noise

### Remaining Issues
- Content still unstructured
- Entire page treated as one block

### Learning
> Cleaning improves readability but does not create usable structure.

---

## 📐 Phase 3: Layout Awareness

### Changes Implemented
- Switched to:

```python
page.get_text("blocks")
```

- Sorted blocks by position

### Improvements
- Better paragraph grouping
- Improved reading order

### Remaining Issues
- No semantic understanding
- Activities and sections still merged

### Learning
> Layout preservation does not guarantee semantic segmentation.

---

## 🧠 Phase 4: Structured Parsing (Major Breakthrough)

This phase introduced the core logic that made the corpus usable.

---

### 🔹 4.1 Section Detection

**Problem**  
Needed to identify textbook hierarchy (e.g., `1.1`, `1.1.1`, `1.2.3`)

**Solution**  
Implemented strict regex:
```python
r'^\d+\.\d+(\.\d+)*'
```

**Fix Applied**  
- Ensured section is detected only at start of line
- Prevented false positives like `100`, `1000`

---

### 🔹 4.2 State-Based Section Tracking

**Problem**  
One page contains multiple sections

**Solution**  
Introduced:
```python
current_section = None
```
Updated whenever a new section appears

**Outcome**  
Each block correctly mapped to its section

---

### 🔹 4.3 Anchor-Based Segmentation

**Problem**  
Line-based splitting caused over-fragmentation ❌ and over-merging ❌

**Solution**  
Used semantic anchors to split text:
- Section headings → new block
- `Activity X.X` → new block
- Questions → new block

**Outcome**  
Meaningful block segmentation

---

### 🔹 4.4 Activity Extraction Fix

**Problem**  
Activities appeared inside concept blocks:
```
"... concept ... Activity 1.1 ..."
```

**Solution**  
- Treated `Activity` as a hard boundary
- Cleaned OCR artifacts: `Activity Activity Activity` → `Activity`

**Outcome**  
Activities correctly separated and tagged

---

### 🔹 4.5 Content Type Classification

Each block labeled as: `concept` | `activity` | `question`

**Logic**
```python
if "Activity"   → activity
elif "Questions" → question
else             → concept
```

---

### 🔹 4.6 Noise Filtering

Removed:
- Bullet points (`•`)
- Broken headers (`ATTER`, `UR`)
- Short meaningless lines
- Repeated OCR noise

---

## 📦 Final Structured Output

```json
{
  "page": 1,
  "section": "1.1.1",
  "type": "activity",
  "text": "Activity 1.1 Take a 100 mL beaker..."
}
```

---

## ✅ Problems Solved

| Issue | Resolution |
|---|---|
| Multiple sections per page | State tracking |
| Activity inside concept | Anchor splitting |
| Incorrect sections (100, 1000) | Strict regex |
| Fragmentation | Controlled grouping |
| OCR noise | Filtering |

---

## 🔍 Phase 5: OCR Fallback (Robustness)

### Motivation
Some pages had poor extraction quality

### Implementation
Applied OCR when extracted text was too small

### Tools
- Tesseract OCR

### Outcome
Improved handling of scanned content and diagrams

### Trade-offs
- Slower processing
- Slight noise introduced

---

## ✂️ Phase 6: Chunking (LLM Preparation)

### Goal
Prepare structured data for retrieval and LLM input

### Initial Approach
Fixed-size chunking

**Problem:** Broke semantic continuity — examples separated from explanations ❌

### Final Approach
Overlapping chunks:
```python
chunk_size = 300
overlap = 50
```

### Output
```json
{
  "chunk_id": 12,
  "page": 4,
  "section": "1.3.1",
  "type": "concept",
  "text": "..."
}
```

### Learning
> Chunking quality has a greater impact than model selection.

---

## 🧪 Phase 7: Tokenizer Analysis

### Goal
Understand how tokenization affects chunking and model input

### Implementation
- Used extracted corpus instead of static samples
- Compared:
  - GPT-2 (BPE)
  - BERT (WordPiece)
  - T5 (SentencePiece)

### Observations

| Tokenizer | Behavior |
|---|---|
| GPT-2 | Higher token count, aggressive splitting |
| BERT | Moderate splitting |
| T5 | More natural segmentation |

### Enhancement
Saved output to file: `extracted_text/tokenizer_analysis.txt`

### Learning
> Tokenization directly impacts chunk size, cost, and retrieval performance.

---

## ⚠️ Key Challenges

| # | Challenge | Solution |
|---|---|---|
| 1 | Multiple sections per page | State tracking |
| 2 | Activities embedded in text | Anchor-based segmentation |
| 3 | Incorrect section detection | Strict regex matching |
| 4 | OCR noise | Regex cleaning and filtering |
| 5 | Over-splitting vs over-merging | Semantic anchors |

---

## 🧠 Key Design Decisions

### 1. Anchor-Based Segmentation
Used semantic markers instead of formatting

### 2. State-Based Parsing
Maintained context across lines

### 3. Structured JSON Output
Enabled retrieval, debugging, and traceability

### 4. Hybrid Extraction Strategy
Combined native extraction with OCR fallback

---

## 📊 Final Outcome

The corpus is now:

- ✅ Clean
- ✅ Structured
- ✅ Section-aware
- ✅ Activity-aware
- ✅ Question-aware
- ✅ Retrieval-ready

---

## 🚀 Conclusion

The extraction pipeline evolved from a simple text dump into a structured, semantically segmented corpus through iterative refinement.

> **Key Insight:** The quality of extraction and chunking determines system performance more than the choice of LLM.

This corpus forms the foundation for:

- **Stage 2:** Retrieval (BM25)
- **Stage 3:** Grounded QA
- **Stage 4:** Evaluation

---

## 🔤 Tokenizer Analysis & Selection

### Goal
Understand how tokenization affects chunking and model input by comparing three tokenizers on real extracted corpus samples.

### Tokenizers Compared

| Tokenizer | Model | Algorithm |
|---|---|---|
| GPT-2 | `gpt2` | BPE (Byte-Pair Encoding) |
| BERT | `bert-base-uncased` | WordPiece |
| T5 | `t5-small` | SentencePiece |

---

### Token Count Comparison (5 Samples)

| Sample | GPT-2 | BERT | T5 |
|---|---|---|---|
| Concept text | 234 | **227** | 239 |
| Section heading | 79 | **73** | 82 |
| Activity | 72 | 81 | 79 |
| Question | 50 | **47** | **47** |
| Mixed / Fig content | 112 | **110** | 128 |

---

### Sample Observations

**Sample 2 — Section Heading** (`1.1.1 IS MADE UP OF PARTICLES`)
- GPT-2 splits aggressively: `['1', '.', '1', '.', '1', 'Ġ', 'ĠIS', 'ĠM', 'ADE', ...]`
- BERT lowercases and normalizes cleanly: `['1', '.', '1', '.', '1', 'is', 'made', 'up', ...]`
- T5 over-splits: `['▁1.', '1.1', '▁IS', '▁M', 'ADE', ...]`

**Sample 3 — Activity Block** (`Activity ______ 1.1 Take a 100 mL beaker...`)
- BERT expands underscores individually (`_`, `_`, `_`, ...) — adds noise tokens
- GPT-2 groups them: `['________', '______']` — slightly better
- T5: `['_______', '_______']` — comparable to GPT-2

**Sample 5 — Mixed Content** (`100 mL of water. Fig. 1.1: When we dissolve...`)
- T5 balloons to **128 tokens** due to unit splitting (`m`, `L` separately)
- BERT handles it most compactly at **110 tokens**
- GPT-2 at 112 tokens — close but slightly worse

---

### ✅ Finalised Tokenizer: BERT (`bert-base-uncased`)

**Reasons:**

1. **Efficiency** — Consistently lowest or near-lowest token counts. Fewer tokens = larger meaningful chunks within model limits.

2. **Section heading normalization** — Lowercasing handles noisy ALL-CAPS OCR output cleanly without extra overhead.

3. **No generation bias** — GPT-2 is designed for text *generation*, not retrieval or QA. Its `Ġ` spacing tokens add overhead with no retrieval benefit.

4. **T5 weakness on mixed content** — Units like `mL`, figure labels, and decimals balloon token counts significantly in T5 (128 vs BERT's 110 on Sample 5). This corpus has abundant such content.

5. **Task alignment** — Stage 2 is BM25 retrieval and Stage 3 is Grounded QA — both pair naturally with BERT-family models.

---

### ⚠️ Caveat

> If the downstream QA model is switched to a **T5 / FLAN-T5** based model, the tokenizer must also switch to T5. The tokenizer and model must always belong to the **same family**.

---

### Output
Analysis saved to: `extracted_text/tokenizer_analysis.txt`