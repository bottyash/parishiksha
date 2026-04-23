# v6_chunking.py
import json
from pathlib import Path

IN_FILE = "extracted_text/extracted_text.json"
OUT_DIR = Path("chunk/chunked_text.json")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 300
OVERLAP = 50

def chunk_text(pages):
    chunks = []
    cid = 0

    for p in pages:
        words = p["text"].split()

        for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
            chunk = " ".join(words[i:i+CHUNK_SIZE])
            chunks.append({
                "chunk_id": cid,
                "page": p["page"],
                "text": chunk
            })
            cid += 1

    return chunks

def main():
    pages = json.load(open(IN_FILE, encoding="utf-8"))
    chunks = chunk_text(pages)

    out_file = OUT_DIR / "chunks.json"
    json.dump(chunks, open(out_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"[v6] Wrote: {out_file} | total chunks={len(chunks)}")

if __name__ == "__main__":
    main()