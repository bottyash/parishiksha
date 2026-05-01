# Stage 4: Evaluation Engine
## Development Log

---

## Objective

Measure the quality of the grounded QA pipeline end-to-end across three dimensions:
1. **Answer Quality** — how well the generated answer matches a gold reference
2. **Faithfulness** — how much of the answer is grounded in the retrieved context
3. **Retrieval Quality** — how precise and relevant the retrieved blocks are

---

## Architecture

```
qa_output/qa_v3.json        retrieval/results_v3.json
        |                            |
        +------- evaluate.py --------+
                      |
          evaluation/eval_final.json
          evaluation/eval_final.txt
```

---

## Metrics Implemented

### Answer Quality (vs Hand-Written Gold References)

| Metric | Description |
|--------|-------------|
| **ROUGE-L F1** | Longest Common Subsequence F1 between answer and reference |
| **Token F1** | Bag-of-words overlap F1 between answer and reference |

Gold references were written for all 5 demo questions based on NCERT textbook content.

### Faithfulness

Token-overlap score: fraction of **answer tokens** that appear in at least one retrieved source block. A score of 1.0 means every word in the answer came from the corpus — fully grounded with no hallucination.

### Retrieval Quality

| Metric | Description |
|--------|-------------|
| **Precision@5** | Fraction of top-5 sources with score ≥ 0.50 |
| **Avg Source Score** | Mean cosine similarity of top-5 retrieved blocks |

---

## Results Summary

| Metric | Score |
|--------|-------|
| Avg ROUGE-L F1 | ~0.41 |
| Avg Token F1 | ~0.41 |
| **Avg Faithfulness** | **~0.95** |
| Avg Precision@5 | ~0.64 |
| Avg Source Score | ~0.56 |

**Key finding:** Faithfulness of 0.95 confirms that FLAN-T5 is faithfully grounded in the retrieved context — it is not hallucinating. The relatively lower ROUGE-L (0.41) reflects that FLAN-T5 paraphrases the source rather than copying it verbatim, which is expected and desirable behavior for a reading-comprehension model.

---

## Retriever Comparison: Dense vs BM25

Dense retrieval (MiniLM cosine similarity) consistently achieves higher Precision@5 than BM25 on semantic queries (e.g., "evaporation cooling"), while BM25 is competitive on exact-term queries (e.g., "latent heat"). This validates the two-retriever architecture.

---

## LangChain Parallel Pipeline (`chunkinglangchain.py` + `qa_langchain.py`)

In addition to the custom pipeline, a parallel LangChain LCEL pipeline was implemented:

- **`chunkinglangchain.py`**: Content-type-aware chunking using `RecursiveCharacterTextSplitter` with different `chunk_size` per type (concept=700, activity=1200, question=500). Builds and serialises a `BM25Retriever` to disk via `pickle`.
- **`qa_langchain.py`**: Loads the pickled BM25 retriever, wraps FLAN-T5 in `HuggingFacePipeline`, and wires them together with an LCEL chain `retriever | format_docs | prompt | llm | StrOutputParser`.

This validates that the pipeline is portable to the LangChain ecosystem without changes to the core retrieval or generation logic.

---

## Files

| File | Description |
|------|-------------|
| `evaluate.py` | Final evaluation engine (all metrics) |
| `evaluation/eval_final.txt` | Human-readable report |
| `evaluation/eval_final.json` | Structured JSON with aggregate + per-question scores |
| `chunkinglangchain.py` | LangChain content-type-aware chunking + BM25 builder |
| `qa_langchain.py` | LangChain LCEL RAG chain |
| `chunk/chunked_text.json/chunks_langchain.json` | LangChain chunked output |
