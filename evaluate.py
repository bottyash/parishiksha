# evaluate.py — Stage 4: Evaluation Engine
# Iteration 1: Token-overlap metrics — ROUGE-L, F1, Faithfulness score

import json
import re
import math
from pathlib import Path
from collections import Counter

QA_INPUT = "qa_output/qa_v3.json"
OUT_DIR  = Path("evaluation")
OUT_DIR.mkdir(exist_ok=True)


# ── tokenise ──────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ── ROUGE-L ───────────────────────────────────────────────────────────────────
def lcs_length(a: list, b: list) -> int:
    """Longest Common Subsequence length (space-efficient)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(curr[j - 1], prev[j])
        prev = curr
    return prev[len(b)]


def rouge_l(hypothesis: str, reference: str) -> dict:
    h, r   = tokenize(hypothesis), tokenize(reference)
    if not h or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs    = lcs_length(h, r)
    prec   = lcs / len(h)
    rec    = lcs / len(r)
    f1     = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# ── token-level F1 ────────────────────────────────────────────────────────────
def token_f1(prediction: str, reference: str) -> float:
    pred_tok = Counter(tokenize(prediction))
    ref_tok  = Counter(tokenize(reference))
    common   = sum((pred_tok & ref_tok).values())
    if common == 0:
        return 0.0
    prec = common / sum(pred_tok.values())
    rec  = common / sum(ref_tok.values())
    return round(2 * prec * rec / (prec + rec), 4)


# ── faithfulness score ────────────────────────────────────────────────────────
def faithfulness(answer: str, sources: list[dict]) -> float:
    """
    Token-overlap faithfulness: what fraction of answer tokens appear
    in at least one retrieved source block?
    """
    answer_toks = set(tokenize(answer))
    if not answer_toks:
        return 0.0
    ctx_toks = set()
    for s in sources:
        ctx_toks.update(tokenize(s.get("text", "")))
    supported = answer_toks & ctx_toks
    return round(len(supported) / len(answer_toks), 4)


# ── reference answers (hand-written) ─────────────────────────────────────────
REFERENCES = {
    "What are the characteristics of particles of matter?":
        "Particles of matter have space between them, they are continuously moving, "
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
        "and the water level does not change significantly.",
}


# ── evaluate ──────────────────────────────────────────────────────────────────
def evaluate(records: list[dict]) -> list[dict]:
    results = []
    for rec in records:
        q      = rec["question"]
        answer = rec["answer"]
        ref    = REFERENCES.get(q, "")
        sources = rec.get("sources", [])

        rl  = rouge_l(answer, ref) if ref else {"precision": 0, "recall": 0, "f1": 0}
        tf1 = token_f1(answer, ref) if ref else 0.0
        fth = faithfulness(answer, sources)

        results.append({
            "question":    q,
            "answer":      answer,
            "reference":   ref,
            "rouge_l_f1":  rl["f1"],
            "rouge_l_prec": rl["precision"],
            "rouge_l_rec": rl["recall"],
            "token_f1":    tf1,
            "faithfulness": fth,
        })
    return results


# ── summary ───────────────────────────────────────────────────────────────────
def print_summary(results: list[dict]) -> str:
    sep  = "=" * 70
    lines = [sep, "Stage 4 — Evaluation Report [Iteration 1]", sep, ""]

    for r in results:
        lines += [
            f"Q:  {r['question']}",
            f"A:  {r['answer']}",
            f"    ROUGE-L F1={r['rouge_l_f1']:.4f}  "
            f"P={r['rouge_l_prec']:.4f}  R={r['rouge_l_rec']:.4f}",
            f"    Token F1={r['token_f1']:.4f}  |  "
            f"Faithfulness={r['faithfulness']:.4f}",
            "",
        ]

    avg_rl  = sum(r["rouge_l_f1"]   for r in results) / len(results)
    avg_tf1 = sum(r["token_f1"]     for r in results) / len(results)
    avg_fth = sum(r["faithfulness"] for r in results) / len(results)

    lines += [
        sep,
        "AGGREGATE SCORES",
        sep,
        f"  Avg ROUGE-L F1   : {avg_rl:.4f}",
        f"  Avg Token F1     : {avg_tf1:.4f}",
        f"  Avg Faithfulness : {avg_fth:.4f}",
        sep,
    ]
    report = "\n".join(lines)
    print(report)
    return report


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Stage 4 — Evaluation Engine [Iteration 1] ===\n")

    with open(QA_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)

    results = evaluate(records)
    report  = print_summary(results)

    # save
    txt = OUT_DIR / "eval_v1.txt"
    jsn = OUT_DIR / "eval_v1.json"
    txt.write_text(report, encoding="utf-8")
    jsn.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Report -> {txt}")
    print(f"[OK] JSON   -> {jsn}")


if __name__ == "__main__":
    main()
