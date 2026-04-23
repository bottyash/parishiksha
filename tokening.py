# v7_tokenizer.py (SAVE OUTPUT)

import json
from transformers import AutoTokenizer
from pathlib import Path

INPUT = "extracted_text/extracted_text.json"
OUTPUT = Path("tokening/tokenizer_analysis.txt")

models = {
    "GPT2": "gpt2",
    "BERT": "bert-base-uncased",
    "T5": "t5-small"
}

def load_samples(n=5):
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        text = item["text"]
        if len(text.split()) > 20:
            samples.append(text)
        if len(samples) >= n:
            break

    return samples


def main():
    samples = load_samples()

    with open(OUTPUT, "w", encoding="utf-8") as out:

        for name, model_name in models.items():
            header = f"\n==== {name} ({model_name}) ====\n"
            print(header)
            out.write(header)

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            for i, text in enumerate(samples):
                tokens = tokenizer.tokenize(text)

                block = f"""
--- Sample {i+1} ---
Text: {text[:120]}...
Token Count: {len(tokens)}
Tokens: {tokens[:25]}...
"""
                print(block)
                out.write(block)

    print(f"\nSaved tokenizer analysis to: {OUTPUT}")


if __name__ == "__main__":
    main()