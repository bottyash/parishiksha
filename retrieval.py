# retrieval.py — Stage 2: BM25 Retrieval
# Iteration 3: Interactive CLI + JSON export + evaluation metrics

import json
import re
import sys
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


# ── BM25 retriever ───────────────────────────────────────────────────────────

class BM25Retriever:
    """
    BM25 retriever with:
    - optional content-type metadata filtering
    - duplicate-aware deduplication
    - query-term highlighting in snippets
    """

    def __init__(self, corpus: list[dict]):
        self.corpus    = corpus
        self.tokenized = [tokenize(doc["text"]) for doc in corpus]
        self.bm25      = BM25Okapi(self.tokenized)

    def search(
        self,
        query: str,
        top_k: int = 5,
        content_type: str | None = None,
        deduplicate: bool = True,
    ) -> list[dict]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        seen: set[str]   = set()
        results: list[dict] = []

        for i in ranked:
            if scores[i] == 0:
                break
            doc = self.corpus[i]
            if content_type and doc["type"] != content_type:
                continue
            snippet = doc["text"][:80].strip()
            if deduplicate and snippet in seen:
                continue
            seen.add(snippet)
            results.append({**doc, "score": round(float(scores[i]), 4)})
            if len(results) >= top_k:
                break

        return results

    def stats(self) -> dict:
        types = {}
        for doc in self.corpus:
            t = doc["type"]
            types[t] = types.get(t, 0) + 1
        return {"total_blocks": len(self.corpus), "by_type": types}


# ── formatting ───────────────────────────────────────────────────────────────

BOLD = "\033[1m"
CYAN = "\033[96m"
GRN  = "\033[92m"
YLW  = "\033[93m"
RST  = "\033[0m"

TYPE_COLOR = {"concept": GRN, "activity": YLW, "question": CYAN}


def highlight(text: str, query_tokens: list[str], width: int = 220) -> str:
    """Bold-highlight query terms in the snippet (terminal only)."""
    snippet = text[:width]
    for tok in set(query_tokens):
        pattern = re.compile(re.escape(tok), re.IGNORECASE)
        snippet = pattern.sub(f"{BOLD}\\g<0>{RST}", snippet)
    if len(text) > width:
        snippet += "…"
    return snippet


def fmt_result(rank: int, r: dict, query_tokens: list[str], color: bool = True) -> str:
    tc = TYPE_COLOR.get(r["type"], "") if color else ""
    snippet = highlight(r["text"], query_tokens) if color else r["text"][:220] + ("…" if len(r["text"]) > 220 else "")
    return (
        f"  [{rank}] score={r['score']:.4f} | "
        f"page={r['page']} | sec={r['section']} | {tc}type={r['type']}{RST if color else ''}\n"
        f"      {snippet}"
    )


def fmt_result_plain(rank: int, r: dict) -> str:
    snippet = r["text"][:220] + ("…" if len(r["text"]) > 220 else "")
    return (
        f"  [{rank}] score={r['score']:.4f} | "
        f"page={r['page']} | sec={r['section']} | type={r['type']}\n"
        f"      {snippet}"
    )


# ── demo run ─────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    ("what are the characteristics of particles of matter", None),
    ("evaporation causes cooling explain",                 "concept"),
    ("difference between solid liquid and gas",           "concept"),
    ("latent heat of fusion and vaporisation",            "concept"),
    ("activity dissolve salt in water",                   "activity"),
]


def run_demo(retriever: BM25Retriever) -> list[dict]:
    """Run predefined demo queries, print results, return structured records."""
    records = []
    out_lines: list[str] = [
        "Stage 2 — BM25 Retrieval  [Iteration 3: Interactive CLI + JSON Export]\n"
    ]

    for query, ctype in DEMO_QUERIES:
        filter_label = f" [filter: {ctype}]" if ctype else ""
        header = f"Query: {query}{filter_label}"
        sep    = "─" * 72

        print(f"\n{sep}")
        print(f"{BOLD}{header}{RST}")
        print(sep)
        out_lines += ["", sep, header, sep]

        results = retriever.search(query, top_k=5, content_type=ctype)
        qtoks   = tokenize(query)

        if not results:
            msg = "  (no results)"
            print(msg); out_lines.append(msg)
        else:
            for rank, r in enumerate(results, 1):
                print(fmt_result(rank, r, qtoks, color=True))
                out_lines.append(fmt_result_plain(rank, r))

        records.append({"query": query, "filter": ctype, "results": results})

    return records, out_lines


# ── interactive REPL ─────────────────────────────────────────────────────────

def interactive_mode(retriever: BM25Retriever):
    print(f"\n{BOLD}=== BM25 Interactive Search ==={RST}")
    print("Commands:  <query>             — search all types")
    print("           <query> --concept   — filter concept blocks")
    print("           <query> --activity  — filter activity blocks")
    print("           <query> --question  — filter question blocks")
    print("           stats               — show corpus stats")
    print("           quit / exit         — exit\n")

    while True:
        try:
            raw = input("🔍 Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if raw.lower() == "stats":
            s = retriever.stats()
            print(f"  Total blocks: {s['total_blocks']}")
            for t, n in s["by_type"].items():
                print(f"    {t}: {n}")
            continue

        # parse optional type flag
        ctype = None
        for flag in ("--concept", "--activity", "--question"):
            if flag in raw:
                ctype = flag.lstrip("-")
                raw   = raw.replace(flag, "").strip()

        results = retriever.search(raw, top_k=5, content_type=ctype)
        qtoks   = tokenize(raw)
        filter_str = f" [filter: {ctype}]" if ctype else ""
        print(f"\n{BOLD}Results for: {raw}{filter_str}{RST}")

        if not results:
            print("  No results found.")
        else:
            for rank, r in enumerate(results, 1):
                print(fmt_result(rank, r, qtoks, color=True))
        print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}=== Stage 2 — BM25 Retrieval  [Iteration 3] ==={RST}\n")

    corpus    = load_corpus(INPUT)
    retriever = BM25Retriever(corpus)
    s = retriever.stats()
    print(f"Corpus: {s['total_blocks']} blocks  "
          + " | ".join(f"{t}={n}" for t, n in s['by_type'].items()))

    # demo mode
    records, out_lines = run_demo(retriever)

    # save plain text
    txt_file = OUT_DIR / "results_v3.txt"
    txt_file.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n✓ Plain results → {txt_file}")

    # save structured JSON
    json_file = OUT_DIR / "results_v3.json"
    json_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"✓ JSON results  → {json_file}")

    # interactive mode if run from terminal (not piped)
    if sys.stdin.isatty():
        interactive_mode(retriever)


if __name__ == "__main__":
    main()
