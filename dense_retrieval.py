# dense_retrieval.py — Stage 2b: Dense Retrieval
# Iteration 3: Interactive CLI + JSON export + color highlighting

import json
import os
import sys
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


# ── retriever ─────────────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Bi-encoder retriever using sentence-transformers/all-MiniLM-L6-v2.
    Embeddings are cached to disk so encoding only happens once.
    Cosine similarity via numpy dot product on normalised vectors.
    """

    def __init__(self, corpus: list[dict], model_name: str = MODEL_NAME):
        self.corpus = corpus
        self.model  = SentenceTransformer(model_name)

        if EMB_CACHE.exists():
            print("  Loading cached dense embeddings …", flush=True)
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
        scores = self.embeddings @ q_emb
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


# ── formatting ────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
CYAN = "\033[96m"
GRN  = "\033[92m"
YLW  = "\033[93m"
RST  = "\033[0m"

TYPE_COLOR = {"concept": GRN, "activity": YLW, "question": CYAN}

def fmt_result(rank: int, r: dict, color: bool = True) -> str:
    tc = TYPE_COLOR.get(r["type"], "") if color else ""
    snippet = r["text"][:220].replace("\n", " ") + ("…" if len(r["text"]) > 220 else "")
    return (
        f"  [{rank}] score={r['score']:.4f} | "
        f"page={r['page']} | sec={r['section']} | {tc}type={r['type']}{RST if color else ''}\n"
        f"      {snippet}"
    )


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
    out_lines: list[str]  = ["Stage 2b — Dense Retrieval [Iteration 3: REPL]\n"]

    for query, ctype in DEMO_QUERIES:
        filter_str = f" [filter: {ctype}]" if ctype else ""
        header     = f"Query: {query}{filter_str}"
        sep        = "─" * 70

        print(f"\n{sep}\n{BOLD}{header}{RST}\n{sep}")
        out_lines += ["", sep, header, sep]

        results = retriever.search(query, top_k=5, content_type=ctype)
        if not results:
            msg = "  (no results)"
            print(msg); out_lines.append(msg)
        else:
            for rank, r in enumerate(results, 1):
                print(fmt_result(rank, r, color=True))
                out_lines.append(fmt_result(rank, r, color=False))
        records.append({"query": query, "filter": ctype, "results": results})

    return records, out_lines


# ── interactive REPL ──────────────────────────────────────────────────────────

def interactive_mode(retriever: DenseRetriever):
    print(f"\n{BOLD}=== Dense Search (MiniLM) ==={RST}")
    print("Commands:  <query>             — search all types")
    print("           <query> --concept   — filter concept blocks")
    print("           <query> --activity  — filter activity blocks")
    print("           <query> --question  — filter question blocks")
    print("           quit / exit         — exit\n")

    while True:
        try:
            raw = input("🧠 Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw: continue
        if raw.lower() in ("quit", "exit"):
            print("Bye!")
            break

        ctype = None
        for flag in ("--concept", "--activity", "--question"):
            if flag in raw:
                ctype = flag.lstrip("-")
                raw   = raw.replace(flag, "").strip()

        results = retriever.search(raw, top_k=5, content_type=ctype)
        filter_str = f" [filter: {ctype}]" if ctype else ""
        print(f"\n{BOLD}Results for: {raw}{filter_str}{RST}")

        if not results:
            print("  No results found.")
        else:
            for rank, r in enumerate(results, 1):
                print(fmt_result(rank, r, color=True))
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.system("") # enable ansi formatting on windows
    print(f"\n{BOLD}=== Stage 2b — Dense Retrieval [Iteration 3] ==={RST}\n")

    corpus    = load_corpus(INPUT)
    retriever = DenseRetriever(corpus)
    s = retriever.stats()
    print(f"Corpus: {s['total']} blocks | " + " | ".join(f"{t}={n}" for t, n in s["by_type"].items()))

    records, out_lines = run_demo(retriever)

    txt = OUT_DIR / "dense_results_v3.txt"
    jsn = OUT_DIR / "dense_results_v3.json"
    txt.write_text("\n".join(out_lines), encoding="utf-8")
    jsn.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Plain results -> {txt}")
    print(f"✓ JSON results  -> {jsn}")

    if sys.stdin.isatty():
        interactive_mode(retriever)

if __name__ == "__main__":
    main()
