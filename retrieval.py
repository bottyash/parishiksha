# retrieval.py — Stage 2: BM25 Retrieval
# Iteration 2: Metadata filtering + deduplication + pretty output

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
    return [d for d in data if len(d["text"].split()) > 5]


# ── BM25 index ───────────────────────────────────────────────────────────────

class BM25Retriever:
    """BM25 retriever with optional content-type filtering and deduplication."""

    def __init__(self, corpus: list[dict]):
        self.corpus    = corpus
        self.tokenized = [tokenize(doc["text"]) for doc in corpus]
        self.bm25      = BM25Okapi(self.tokenized)

    def search(
        self,
        query: str,
        top_k: int = 5,
        content_type: str | None = None,   # filter by 'concept'|'activity'|'question'
        deduplicate: bool = True,
    ) -> list[dict]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # rank all
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        seen_texts: set[str] = set()
        results: list[dict]  = []

        for i in ranked:
            if scores[i] == 0:
                break  # no more relevant results

            doc = self.corpus[i]

            # optional type filter
            if content_type and doc["type"] != content_type:
                continue

            # deduplication: skip near-duplicate text (same first 80 chars)
            snippet = doc["text"][:80].strip()
            if deduplicate and snippet in seen_texts:
                continue
            seen_texts.add(snippet)

            results.append({**doc, "score": round(float(scores[i]), 4)})

            if len(results) >= top_k:
                break

        return results


# ── pretty print ─────────────────────────────────────────────────────────────

def format_result(rank: int, r: dict, width: int = 200) -> str:
    snippet = r["text"][:width].replace("\n", " ")
    if len(r["text"]) > width:
        snippet += "…"
    return (
        f"  [{rank}] score={r['score']:.4f} | "
        f"page={r['page']} | sec={r['section']} | type={r['type']}\n"
        f"      {snippet}"
    )


# ── demo queries with optional filters ───────────────────────────────────────

DEMO_QUERIES = [
    # (query, content_type_filter)
    ("what are the characteristics of particles of matter", None),
    ("evaporation causes cooling explain",                 "concept"),
    ("difference between solid liquid and gas",           "concept"),
    ("latent heat of fusion and vaporisation",            "concept"),
    ("activity dissolve salt in water",                   "activity"),
]

def main():
    print("\n=== Stage 2 — BM25 Retrieval  [Iteration 2: Metadata Filtering] ===\n")

    corpus    = load_corpus(INPUT)
    retriever = BM25Retriever(corpus)
    print(f"Corpus: {len(corpus)} blocks indexed.\n")

    out_lines: list[str] = [
        "=== Stage 2 — BM25 Retrieval  [Iteration 2: Metadata Filtering] ===\n"
    ]

    for query, ctype in DEMO_QUERIES:
        filter_label = f"  (filter: type='{ctype}')" if ctype else "  (no filter)"
        header = f"Query: {query}{filter_label}"
        sep    = "─" * 70

        print(sep)
        print(header)
        print(sep)
        out_lines += [sep, header, sep]

        results = retriever.search(query, top_k=5, content_type=ctype)

        if not results:
            msg = "  (no results after filtering)"
            print(msg)
            out_lines.append(msg)
        else:
            for rank, r in enumerate(results, 1):
                line = format_result(rank, r)
                print(line)
                out_lines.append(line)

        print()
        out_lines.append("")

    # save
    out_file = OUT_DIR / "results_v2.txt"
    out_file.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"✓ Results saved to {out_file}")


if __name__ == "__main__":
    main()
