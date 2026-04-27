# qa.py — Stage 3: Grounded QA (RAG)
# Iteration 1: Dense retrieval -> FLAN-T5 context assembly -> grounded answer

import json
import warnings
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
CORPUS_PATH  = "extracted_text/extracted_text.json"
EMB_CACHE    = Path("retrieval/cache/corpus_embeddings.npy")
RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL        = "google/flan-t5-base"
TOP_K        = 4
OUT_DIR      = Path("qa_output")
OUT_DIR.mkdir(exist_ok=True)


# ── corpus ────────────────────────────────────────────────────────────────────
def load_corpus(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if len(d["text"].split()) > 5]


# ── retriever ─────────────────────────────────────────────────────────────────
class DenseRetriever:
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self.model  = SentenceTransformer(RETRIEVER_MODEL)
        if EMB_CACHE.exists():
            print("  Loading cached embeddings ...", flush=True)
            self.embeddings = np.load(str(EMB_CACHE)).astype("float32")
        else:
            print(f"  Encoding {len(corpus)} blocks ...", flush=True)
            self.embeddings = self.model.encode(
                [d["text"] for d in corpus],
                batch_size=32, show_progress_bar=True, normalize_embeddings=True
            ).astype("float32")
            np.save(str(EMB_CACHE), self.embeddings)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        q_emb  = self.model.encode([query], normalize_embeddings=True).astype("float32")[0]
        scores = self.embeddings @ q_emb
        ranked = np.argsort(scores)[::-1]
        return [{**self.corpus[i], "score": round(float(scores[i]), 4)} for i in ranked[:top_k]]


# ── reader (FLAN-T5) ──────────────────────────────────────────────────────────
class FlanT5Reader:
    def __init__(self, model_name: str = QA_MODEL):
        print(f"  Loading QA model: {model_name} ...", flush=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

    def answer(self, question: str, context_blocks: list[dict]) -> str:
        context = "\n".join(b["text"] for b in context_blocks)
        prompt  = (
            f"Answer the question based only on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            max_length=1024, truncation=True
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ── demo ──────────────────────────────────────────────────────────────────────
DEMO_QUESTIONS = [
    "What are the characteristics of particles of matter?",
    "Why does evaporation cause cooling?",
    "What is the difference between solid, liquid and gas?",
    "What is latent heat of vaporisation?",
    "What happens when you dissolve salt in water?",
]

def main():
    print("\n=== Stage 3 - Grounded QA [Iteration 1: FLAN-T5] ===\n")

    corpus    = load_corpus(CORPUS_PATH)
    retriever = DenseRetriever(corpus)
    reader    = FlanT5Reader()

    records   = []
    out_lines = ["Stage 3 - Grounded QA [Iteration 1: FLAN-T5 base]\n"]

    for question in DEMO_QUESTIONS:
        sep = "-" * 70
        print(f"\n{sep}\nQ: {question}\n{sep}")
        out_lines += ["", sep, f"Q: {question}", sep]

        blocks = retriever.retrieve(question, top_k=TOP_K)
        answer = reader.answer(question, blocks)

        print(f"A: {answer}\n")
        print("Sources:")
        out_lines.append(f"A: {answer}")
        out_lines.append("Sources:")

        for i, b in enumerate(blocks, 1):
            src = f"  [{i}] page={b['page']} sec={b['section']} type={b['type']} score={b['score']}"
            print(src)
            out_lines.append(src)

        records.append({
            "question": question,
            "answer":   answer,
            "sources":  [{"page": b["page"], "section": b["section"],
                          "type": b["type"], "score": b["score"],
                          "text": b["text"][:300]} for b in blocks],
        })

    # save
    txt = OUT_DIR / "qa_v1.txt"
    jsn = OUT_DIR / "qa_v1.json"
    txt.write_text("\n".join(out_lines), encoding="utf-8")
    jsn.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Results -> {txt}")
    print(f"[OK] JSON    -> {jsn}")

if __name__ == "__main__":
    main()
