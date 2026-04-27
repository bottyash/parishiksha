# Stage 3: Grounded QA (RAG)
## Development Log

---

## Objective

Connect the retrieval engines from Stage 2 to a generative model to produce grounded natural-language answers with source attribution. The pipeline is: **Query → Dense Retrieve → Context Assembly → FLAN-T5 Generate → Answer + Sources**.

---

## Architecture

```
User Question
     |
DenseRetriever (MiniLM)
     |  top-K blocks (with page, section, type, score)
     |
Context Assembly (cap at 600 words, preserve metadata labels)
     |
FLAN-T5-base (google/flan-t5-base)
     |
Grounded Answer + Source List
```

---

## Iterative Development

### Iteration 1 — Basic RAG (`qa_v1.txt`)

- Loaded `DenseRetriever` (reusing `retrieval/cache/corpus_embeddings.npy` from Stage 2b).
- Assembled context as plain concatenated block text.
- Passed a minimal prompt: *"Answer the question based only on the context below."*
- Used `num_beams=4`, `max_new_tokens=200`.

**Results:**
- Answers were factually correct but very short (extractive fragments).
- e.g., "latent heat of vaporisation" → *"heat energy required to change 1 kg of a liquid to gas"* ✓

---

### Iteration 2 — Improved Prompts (`qa_v2.txt`)

**Changes:**
- Made FLAN-T5 act as a **science teacher**: *"You are a science teacher... Give a complete, factual answer."*
- Added **metadata labels** to each block in context: `[CONCEPT | p.9 | sec 1.5.2]`.
- Set `length_penalty=1.5` and `max_new_tokens=256` to encourage fuller sentences.
- Capped context at **600 words** to prevent T5's 1024-token input limit from truncating the question.
- Added `no_repeat_ngram_size=3` to prevent repetitive output.

**Results (improvement):**
| Question | v1 Answer | v2 Answer |
|----------|-----------|-----------|
| Evaporation cooling | *(fragment)* | *"The particles at the surface... gain energy from the surroundings... change into vapour."* |
| Dissolve salt | *"...particles of salt get into the spaces..."* | *"The particles of salt get into the spaces between particles of water."* |

---

### Iteration 3 — Interactive REPL (`qa_v3.json`)

**Changes:**
- Added ANSI colour output: answers in green, sources in cyan.
- Added **interactive REPL** — type any science question, get a grounded answer with citations.
- Session auto-saved to `qa_output/session.json` on exit.
- Full JSON export for all demo answers.

---

## Model Details

| Component | Model | Role |
|-----------|-------|------|
| Retriever | `sentence-transformers/all-MiniLM-L6-v2` | Dense semantic retrieval (384-dim bi-encoder) |
| Reader | `google/flan-t5-base` | Instruction-tuned seq2seq for answer generation |

**Why FLAN-T5-base?**
- Runs fully locally on CPU (no GPU required)
- Instruction-tuned — follows role prompts reliably
- Compact (~248MB) vs. larger models

**Why not GPT/Claude API?**
- Full local pipeline = reproducible and self-contained
- Appropriate for a corpus-grounded task where hallucination should be minimal

---

## Sample Output (Iteration 3)

| Question | Answer | Top Source |
|----------|--------|------------|
| Why does evaporation cause cooling? | *"The particles at the surface of the liquid gain energy from the surroundings or body surface and change into vapour."* | p.9, sec 1.5.2, concept (score 0.73) |
| What is latent heat of vaporisation? | *"The heat energy required to change 1 kg of a liquid to gas"* | p.11, sec 1, concept (score 0.55) |
| What happens when you dissolve salt in water? | *"The particles of salt get into the spaces between particles of water."* | p.1, sec 1.1.1, activity (score 0.63) |

---

## Next: Stage 4 — Evaluation Engine

Measure the quality of grounded answers using:
- **Faithfulness**: Is the answer supported by the retrieved context?
- **Answer Relevance**: Does the answer address the question?
- **Retrieval Recall**: Are the right blocks being retrieved?
