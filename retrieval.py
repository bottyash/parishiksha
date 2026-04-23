# retrieval.py — Stage 2: BM25 Retrieval
# Iteration 1: Basic BM25 search over extracted corpus

import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

INPUT   = "extracted_text/extracted_text.json"
OUT_DIR = Path("retrieval")
OUT_DIR.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_corpus(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # filter out very short / noisy blocks
    return [d for d in data if len(d["text"].split()) > 5]


# ── BM25 index ───────────────────────────────────────────────────────────────

def build_index(corpus: list[dict]) -> BM25Okapi:
    tokenized = [tokenize(doc["text"]) for doc in corpus]
    return BM25Okapi(tokenized)


# ── retrieval ────────────────────────────────────────────────────────────────

def search(query: str, bm25: BM25Okapi, corpus: list[dict], top_k: int = 5) -> list[dict]:
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [
        {**corpus[i], "score": round(float(scores[i]), 4)}
        for i in ranked[:top_k]
    ]


# ── demo ─────────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "what are the characteristics of particles of matter",
    "evaporation causes cooling explain",
    "difference between solid liquid and gas",
    "latent heat of fusion and vaporisation",
    "activity dissolve salt in water",
]

def main():
    print("\n=== Stage 2 — BM25 Retrieval  [Iteration 1: Basic] ===\n")

    corpus = load_corpus(INPUT)
    print(f"Corpus size: {len(corpus)} blocks\n")

    bm25 = build_index(corpus)
    print("BM25 index built.\n")

    all_results = {}
    out_lines   = []

    for query in DEMO_QUERIES:
        results = search(query, bm25, corpus)
        all_results[query] = results

        header = f"Query: {query}"
        out_lines.append("=" * 70)
        out_lines.append(header)
        out_lines.append("=" * 70)
        print("=" * 70)
        print(header)
        print("=" * 70)

        for rank, r in enumerate(results, 1):
            line = (
                f"  [{rank}] score={r['score']:.4f} | "
                f"page={r['page']} | section={r['section']} | type={r['type']}\n"
                f"      {r['text'][:200]}..."
            )
            print(line)
            out_lines.append(line)
        print()
        out_lines.append("")

    # save results
    out_file = OUT_DIR / "results_v1.txt"
    out_file.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n✓ Results saved to {out_file}")

if __name__ == "__main__":
    main()
