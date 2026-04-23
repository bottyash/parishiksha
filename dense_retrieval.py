# dense_retrieval.py — Stage 2b: Dense Retrieval
# Iteration 2: Embedding cache (encode once, search fast with numpy)

import json
import warnings
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

INPUT     = "extracted_text/extracted_text.json"
OUT_DIR   = Path("retrieval")
CACHE_DIR = Path("retrieval/cache")
EMB_CACHE = CACHE_DIR / "corpus_embeddings.npy"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── corpus ────────────────────────────────────────────────────────────────────

def load_corpus(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if len(d["text"].split()) > 5]


# ── dense retriever with FAISS + cache ───────────────────────────────────────

class DenseRetriever:
    """
    Bi-encoder retriever using sentence-transformers/all-MiniLM-L6-v2.

    Embeddings are cached to disk so encoding only happens once —
    subsequent runs load from cache in milliseconds.
    Cosine similarity via numpy dot product on normalised vectors.
    (Sufficient for corpora of a few hundred blocks; swap in FAISS for scale.)
    """

    def __init__(self, corpus: list[dict], model_name: str = MODEL_NAME):
        self.corpus = corpus
        self.model  = SentenceTransformer(model_name)

        if EMB_CACHE.exists():
            print("  Loading cached embeddings …", flush=True)
            self.embeddings = np.load(str(EMB_CACHE)).astype("float32")
        else:
            print(f"  Encoding {len(corpus)} blocks with {model_name} …", flush=True)
            texts = [d["text"] for d in corpus]
            self.embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
            ).astype("float32")
            np.save(str(EMB_CACHE), self.embeddings)
            print(f"  Saved embeddings -> {EMB_CACHE}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        content_type: str | None = None,
    ) -> list[dict]:
        q_emb  = self.model.encode([query], normalize_embeddings=True).astype("float32")[0]
        scores = self.embeddings @ q_emb       # cosine similarity (vecs are normalised)
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

    def stats(self) -> dict:
        types: dict[str, int] = {}
        for doc in self.corpus:
            types[doc["type"]] = types.get(doc["type"], 0) + 1
        return {"total": len(self.corpus), "by_type": types}


# ── demo queries ──────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    ("what are the characteristics of particles of matter", None),
    ("evaporation causes cooling explain",                 "concept"),
    ("difference between solid liquid and gas",           "concept"),
    ("latent heat of fusion and vaporisation",            "concept"),
    ("activity dissolve salt in water",                   "activity"),
]


def run_demo(retriever: DenseRetriever) -> tuple[list[dict], list[str]]:
    records:   list[dict] = []
    out_lines: list[str]  = [
        "=== Stage 2b — Dense Retrieval [Iteration 2: FAISS + Cache] ===\n"
    ]

    for query, ctype in DEMO_QUERIES:
        filter_str = f" [filter: {ctype}]" if ctype else ""
        header     = f"Query: {query}{filter_str}"
        sep        = "─" * 70

        print(f"\n{sep}\n{header}\n{sep}")
        out_lines += ["", sep, header, sep]

        results = retriever.search(query, top_k=5, content_type=ctype)
        for rank, r in enumerate(results, 1):
            snippet = r["text"][:220].replace("\n", " ")
            if len(r["text"]) > 220:
                snippet += "…"
            line = (
                f"  [{rank}] score={r['score']:.4f} | "
                f"page={r['page']} | sec={r['section']} | type={r['type']}\n"
                f"      {snippet}"
            )
            print(line)
            out_lines.append(line)

        records.append({"query": query, "filter": ctype, "results": results})

    return records, out_lines


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Stage 2b — Dense Retrieval [Iteration 2: Embedding Cache] ===\n")

    corpus    = load_corpus(INPUT)
    retriever = DenseRetriever(corpus)
    s = retriever.stats()
    print(f"\nCorpus: {s['total']} blocks")
    print("  " + " | ".join(f"{t}={n}" for t, n in s["by_type"].items()))

    records, out_lines = run_demo(retriever)

    txt = OUT_DIR / "dense_results_v2.txt"
    jsn = OUT_DIR / "dense_results_v2.json"
    txt.write_text("\n".join(out_lines), encoding="utf-8")
    jsn.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Plain results -> {txt}")
    print(f"✓ JSON results  -> {jsn}")
    print(f"✓ Cache         -> {CACHE_DIR}/")


if __name__ == "__main__":
    main()
