# qa.py — Stage 3: Grounded QA (RAG)
# Iteration 3: Interactive REPL + JSON export + retriever choice

import json
import sys
import warnings
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
CORPUS_PATH     = "extracted_text/extracted_text.json"
EMB_CACHE       = Path("retrieval/cache/corpus_embeddings.npy")
RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL        = "google/flan-t5-base"
TOP_K           = 5
MAX_CTX_WORDS   = 600
OUT_DIR         = Path("qa_output")
OUT_DIR.mkdir(exist_ok=True)


# ── corpus ────────────────────────────────────────────────────────────────────
def load_corpus(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if len(d["text"].split()) > 5]


# ── dense retriever ───────────────────────────────────────────────────────────
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


# ── reader ────────────────────────────────────────────────────────────────────
class FlanT5Reader:
    def __init__(self, model_name: str = QA_MODEL):
        print(f"  Loading QA model: {model_name} ...", flush=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

    def _build_context(self, blocks: list[dict]) -> list[dict]:
        ctx, used = [], []
        for b in blocks:
            if sum(len(x["text"].split()) for x in ctx) + len(b["text"].split()) > MAX_CTX_WORDS:
                break
            ctx.append(b)
            used.append(b)
        return used

    def _build_prompt(self, question: str, blocks: list[dict]) -> str:
        blocks = self._build_context(blocks)
        context = "\n\n".join(
            f"[{b['type'].upper()} | p.{b['page']} | sec {b['section']}]\n{b['text']}"
            for b in blocks
        )
        return (
            "You are a science teacher answering student questions based strictly on "
            "the textbook passages below. Give a complete, factual answer.\n\n"
            f"Textbook passages:\n{context}\n\n"
            f"Student question: {question}\n\n"
            "Teacher answer:"
        )

    def answer(self, question: str, blocks: list[dict]) -> str:
        prompt = self._build_prompt(question, blocks)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            length_penalty=1.5,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ── formatting ────────────────────────────────────────────────────────────────
BOLD = "\033[1m"
GRN  = "\033[92m"
CYN  = "\033[96m"
RST  = "\033[0m"

def print_qa(question: str, answer: str, blocks: list[dict]):
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"{BOLD}Q: {question}{RST}")
    print(sep)
    print(f"{GRN}A: {answer}{RST}\n")
    print(f"{CYN}Sources:{RST}")
    for i, b in enumerate(blocks, 1):
        print(f"  [{i}] page={b['page']} | sec={b['section']} | "
              f"type={b['type']} | score={b['score']}")


# ── demo ──────────────────────────────────────────────────────────────────────
DEMO_QUESTIONS = [
    "What are the characteristics of particles of matter?",
    "Why does evaporation cause cooling?",
    "What is the difference between solid, liquid and gas?",
    "What is latent heat of vaporisation?",
    "What happens when you dissolve salt in water?",
]


def run_demo(retriever: DenseRetriever, reader: FlanT5Reader):
    records   = []
    out_lines = ["Stage 3 - Grounded QA [Iteration 3: Interactive REPL]\n"]

    for question in DEMO_QUESTIONS:
        blocks = retriever.retrieve(question, top_k=TOP_K)
        answer = reader.answer(question, blocks)

        print_qa(question, answer, blocks)
        out_lines += [
            "", "-" * 70,
            f"Q: {question}", "-" * 70,
            f"A: {answer}", "Sources:",
        ]
        for i, b in enumerate(blocks, 1):
            out_lines.append(f"  [{i}] page={b['page']} sec={b['section']} "
                             f"type={b['type']} score={b['score']}")

        records.append({
            "question": question,
            "answer":   answer,
            "sources": [{"page": b["page"], "section": b["section"],
                         "type": b["type"], "score": b["score"],
                         "text": b["text"][:300]} for b in blocks],
        })

    txt = OUT_DIR / "qa_v3.txt"
    jsn = OUT_DIR / "qa_v3.json"
    txt.write_text("\n".join(out_lines), encoding="utf-8")
    jsn.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Results -> {txt}")
    print(f"[OK] JSON    -> {jsn}")
    return records


# ── interactive REPL ──────────────────────────────────────────────────────────
def interactive_mode(retriever: DenseRetriever, reader: FlanT5Reader):
    print(f"\n{BOLD}=== Grounded QA — Ask a question about NCERT Science ==={RST}")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    session = []
    while True:
        try:
            q = input("Ask > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            print("Bye!")
            break

        blocks = retriever.retrieve(q, top_k=TOP_K)
        answer = reader.answer(q, blocks)
        print_qa(q, answer, blocks)
        session.append({"question": q, "answer": answer,
                        "sources": [{"page": b["page"], "section": b["section"],
                                     "type": b["type"]} for b in blocks]})

    if session:
        sf = OUT_DIR / "session.json"
        sf.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Session saved -> {sf}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import os; os.system("")  # enable ANSI on Windows
    print(f"\n{BOLD}=== Stage 3 - Grounded QA [Iteration 3] ==={RST}\n")

    corpus    = load_corpus(CORPUS_PATH)
    retriever = DenseRetriever(corpus)
    reader    = FlanT5Reader()

    run_demo(retriever, reader)

    if sys.stdin.isatty():
        interactive_mode(retriever, reader)


if __name__ == "__main__":
    main()
