# dense_retrieval.py — Stage 2b: Dense Retrieval
# Iteration 1: Encode corpus with MiniLM, cosine similarity search

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT   = "extracted_text/extracted_text.json"
OUT_DIR = Path("retrieval")
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── load corpus ───────────────────────────────────────────────────────────────

def load_corpus(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if len(d["text"].split()) > 5]


# ── dense retriever ───────────────────────────────────────────────────────────

class DenseRetriever:
    def __init__(self, corpus: list[dict], model_name: str = MODEL_NAME):
        print(f"  Loading model: {model_name} …", flush=True)
        self.model  = SentenceTransformer(model_name)
        self.corpus = corpus

        print(f"  Encoding {len(corpus)} blocks …", flush=True)
        texts = [d["text"] for d in corpus]
        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,   # unit vectors → dot product = cosine sim
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        content_type: str | None = None,
    ) -> list[dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q_emb          # cosine similarity for all docs

        ranked = np.argsort(scores)[::-1]
        results = []
        for i in ranked:
            doc = self.corpus[i]
            if content_type and doc["type"] != content_type:
                continue
            results.append({**doc, "score": round(float(scores[i]), 4)})
            if len(results) >= top_k:
                break
        return results


# ── demo ─────────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    ("what are the characteristics of particles of matter", None),
    ("evaporation causes cooling explain",                 "concept"),
    ("difference between solid liquid and gas",           "concept"),
    ("latent heat of fusion and vaporisation",            "concept"),
    ("activity dissolve salt in water",                   "activity"),
]

def main():
    print("\n=== Stage 2b — Dense Retrieval [Iteration 1: MiniLM + Cosine] ===\n")

    corpus    = load_corpus(INPUT)
    retriever = DenseRetriever(corpus)

    out_lines = ["=== Stage 2b — Dense Retrieval [Iteration 1] ===\n"]

    for query, ctype in DEMO_QUERIES:
        filter_str = f" [filter: {ctype}]" if ctype else ""
        header = f"Query: {query}{filter_str}"
        sep    = "─" * 70

        print(f"\n{sep}\n{header}\n{sep}")
        out_lines += ["", sep, header, sep]

        results = retriever.search(query, top_k=5, content_type=ctype)
        for rank, r in enumerate(results, 1):
            snippet = r["text"][:200].replace("\n", " ")
            if len(r["text"]) > 200:
                snippet += "…"
            line = (
                f"  [{rank}] score={r['score']:.4f} | "
                f"page={r['page']} | sec={r['section']} | type={r['type']}\n"
                f"      {snippet}"
            )
            print(line)
            out_lines.append(line)

    out_file = OUT_DIR / "dense_results_v1.txt"
    out_file.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n✓ Results saved to {out_file}")

if __name__ == "__main__":
    main()
