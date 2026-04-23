# v2_clean.py
import fitz, re
from pathlib import Path

PDF= "pdfs/ch1.pdf"
output= Path("extracted_text")
output.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    text = re.sub(r'Reprint\s*\d{4}-\d{2}', '', text)
    text = re.sub(r'Page\s*\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    doc = fitz.open(PDF)
    out_file = output / "extracted_text.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        for i, page in enumerate(doc):
            text = clean_text(page.get_text())
            f.write(f"\n--- PAGE {i+1} ---\n{text}\n")

    print(f"[v2] Wrote: {out_file}")

if __name__ == "__main__":
    main()