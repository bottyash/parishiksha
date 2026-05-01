# evaluate.py — Stage 4: Final Evaluation Engine
# Metrics: ROUGE-L, Token F1, Faithfulness, Precision@K, BM25 vs Dense comparison

import json
import re
import os
from pathlib import Path
from collections import Counter, defaultdict

QA_INPUT   = "qa_output/qa_v3.json"
BM25_INPUT = "retrieval/results_v3.json"
OUT_DIR    = Path("evaluation")
OUT_DIR.mkdir(exist_ok=True)

RELEVANCE_THRESHOLD = 0.50

BOLD = "\033[1m"
GRN  = "\033[92m"
CYN  = "\033[96m"
YLW  = "\033[93m"
RST  = "\033[0m"

# ── text utils ────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())

def strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)

# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def lcs_length(a: list, b: list) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(curr[j - 1], prev[j])
        prev = curr
    return prev[len(b)]

def rouge_l(hyp: str, ref: str) -> dict:
    h, r = tokenize(hyp), tokenize(ref)
    if not h or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs  = lcs_length(h, r)
    prec = lcs / len(h)
    rec  = lcs / len(r)
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

# ── Token F1 ──────────────────────────────────────────────────────────────────

def token_f1(pred: str, ref: str) -> float:
    pc = Counter(tokenize(pred))
    rc = Counter(tokenize(ref))
    common = sum((pc & rc).values())
    if not common:
        return 0.0
    prec = common / sum(pc.values())
    rec  = common / sum(rc.values())
    return round(2 * prec * rec / (prec + rec), 4)

# ── Faithfulness ──────────────────────────────────────────────────────────────

def faithfulness(answer: str, sources: list) -> float:
    atoks = set(tokenize(answer))
    if not atoks:
        return 0.0
    ctx = set()
    for s in sources:
        ctx.update(tokenize(s.get("text", "")))
    return round(len(atoks & ctx) / len(atoks), 4)

# ── Precision@K ───────────────────────────────────────────────────────────────

def precision_at_k(sources: list, k: int = 5) -> float:
    topk = sources[:k]
    rel  = [s for s in topk if s.get("score", 0) >= RELEVANCE_THRESHOLD]
    return round(len(rel) / k, 4) if topk else 0.0

def avg_score(sources: list) -> float:
    return round(sum(s.get("score", 0) for s in sources) / len(sources), 4) if sources else 0.0

# ── Gold references ───────────────────────────────────────────────────────────

REFERENCES = {
    "What are the characteristics of particles of matter?":
        "Particles of matter have space between them, they are continuously moving "
        "and they attract each other.",
    "Why does evaporation cause cooling?":
        "During evaporation, particles at the surface of liquid gain energy from "
        "the surroundings and escape into vapour, making surroundings cooler.",
    "What is the difference between solid, liquid and gas?":
        "Solids have fixed shape and volume. Liquids have fixed volume but no fixed "
        "shape. Gases have neither fixed shape nor fixed volume.",
    "What is latent heat of vaporisation?":
        "Latent heat of vaporisation is the heat energy required to change 1 kg of "
        "liquid to gas at its boiling point at atmospheric pressure.",
    "What happens when you dissolve salt in water?":
        "The particles of salt get into the spaces between particles of water "
        "and the water level does not rise.",
}

# ── Score one record ──────────────────────────────────────────────────────────

def score_record(rec: dict) -> dict:
    q       = rec["question"]
    answer  = rec["answer"]
    ref     = REFERENCES.get(q, "")
    sources = rec.get("sources", [])

    rl    = rouge_l(answer, ref) if ref else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tf1   = token_f1(answer, ref) if ref else 0.0
    fth   = faithfulness(answer, sources)
    p5    = precision_at_k(sources, k=5)
    avgs  = avg_score(sources)

    type_counts: dict = defaultdict(int)
    for s in sources:
        type_counts[s.get("type", "unknown")] += 1

    return {
        "question":          q,
        "answer":            answer,
        "reference":         ref,
        "rouge_l_f1":        rl["f1"],
        "rouge_l_prec":      rl["precision"],
        "rouge_l_rec":       rl["recall"],
        "token_f1":          tf1,
        "faithfulness":      fth,
        "precision_at_5":    p5,
        "avg_source_score":  avgs,
        "source_type_counts": dict(type_counts),
    }

def aggregate(results: list) -> dict:
    n    = len(results)
    keys = ["rouge_l_f1", "token_f1", "faithfulness", "precision_at_5", "avg_source_score"]
    return {k: round(sum(r[k] for r in results) / n, 4) for k in keys}

# ── Retriever comparison ──────────────────────────────────────────────────────

def compare_retrievers(dense_records: list, bm25_path: str) -> str:
    lines = [f"{BOLD}Retriever Comparison: Dense (MiniLM) vs BM25{RST}", ""]
    if not Path(bm25_path).exists():
        lines.append("  [BM25 results file not found — skipped]")
        return "\n".join(lines)

    with open(bm25_path, "r", encoding="utf-8") as f:
        bm25_records = json.load(f)

    bm25_by_q = {r["query"]: r for r in bm25_records}
    lines.append(f"  {'Question':<48} {'Dense P@5':>9} {'BM25 P@5':>9} {'Dense AvgS':>10}")
    lines.append("  " + "-" * 80)

    for rec in dense_records:
        q     = rec["question"]
        d_src = rec.get("sources", [])
        d_p5  = precision_at_k(d_src, 5)
        d_avg = avg_score(d_src)
        b_src = bm25_by_q.get(q, {}).get("results", [])
        b_p5  = precision_at_k(b_src, 5)
        short = (q[:45] + "...") if len(q) > 48 else q
        lines.append(f"  {short:<48} {d_p5:>9.4f} {b_p5:>9.4f} {d_avg:>10.4f}")

    return "\n".join(lines)

# ── Report ────────────────────────────────────────────────────────────────────

def build_report(results: list, comparison: str) -> str:
    sep  = "=" * 70
    lns  = [sep, f"{BOLD}Stage 4 — Evaluation Report (Final){RST}", sep, ""]

    for r in results:
        type_str = "  ".join(f"{k}={v}" for k, v in r["source_type_counts"].items())
        lns += [
            f"{BOLD}Q:{RST}  {r['question']}",
            f"{GRN}A:{RST}  {r['answer']}",
            f"    {CYN}ROUGE-L{RST}  F1={r['rouge_l_f1']:.4f}  "
            f"P={r['rouge_l_prec']:.4f}  R={r['rouge_l_rec']:.4f}",
            f"    {CYN}Token F1{RST}={r['token_f1']:.4f}  "
            f"{CYN}Faithfulness{RST}={r['faithfulness']:.4f}  "
            f"{CYN}P@5{RST}={r['precision_at_5']:.4f}  "
            f"AvgSrc={r['avg_source_score']:.4f}",
            f"    Sources: {type_str}",
            "",
        ]

    avgs = aggregate(results)
    lns += [
        sep,
        f"{BOLD}AGGREGATE SCORES{RST}",
        sep,
        f"  {YLW}Avg ROUGE-L F1   {RST}: {avgs['rouge_l_f1']:.4f}",
        f"  {YLW}Avg Token F1     {RST}: {avgs['token_f1']:.4f}",
        f"  {YLW}Avg Faithfulness {RST}: {avgs['faithfulness']:.4f}",
        f"  {YLW}Avg Precision@5  {RST}: {avgs['precision_at_5']:.4f}",
        f"  {YLW}Avg Source Score {RST}: {avgs['avg_source_score']:.4f}",
        sep,
        "",
        comparison,
    ]
    return "\n".join(lns), avgs

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.system("")  # enable ANSI on Windows
    print(f"\n{BOLD}=== Stage 4 — Evaluation Engine (Final) ==={RST}\n")

    with open(QA_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)

    results    = [score_record(r) for r in records]
    comparison = compare_retrievers(records, BM25_INPUT)
    report, avgs = build_report(results, comparison)

    print(report)

    txt = OUT_DIR / "eval_final.txt"
    jsn = OUT_DIR / "eval_final.json"
    txt.write_text(strip_ansi(report), encoding="utf-8")
    jsn.write_text(json.dumps({"aggregate": avgs, "per_question": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Report -> {txt}")
    print(f"[OK] JSON   -> {jsn}")


if __name__ == "__main__":
    main()
