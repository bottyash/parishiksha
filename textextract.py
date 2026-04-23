# v4 json for better structure (SECTION-AWARE version)

import fitz, re, json
from pathlib import Path

PDF = "pdfs/ch1.pdf"
output = Path("extracted_text")
output.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    text = re.sub(r'Reprint\s*\d{4}-\d{2}', '', text)
    text = re.sub(r'\bSCIENCE\b', '', text)
    text = re.sub(r'\bMATTER\b', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text  #DO NOT collapse newlines

def is_noise(line):
    if len(line.strip()) < 3:
        return True
    if line.strip() in ["•", "*", "-", ""]:
        return True
    if re.match(r'^[A-Z\s]{1,10}$', line):
        return True
    return False

def detect_type(text):
    if re.search(r'Activity\s*[_\s]*\d+\.\d+', text):
        return "activity"
    elif re.search(r'uestions|Questions', text):
        return "question"
    return "concept"

def extract_section(line):
    match = re.match(r'^\d+(\.\d+)*', line.strip())
    return match.group(0) if match else None


def group_lines_into_blocks(lines):
    blocks = []
    current = ""
    current_section = None

    for line in lines:
        line = line.strip()

        if is_noise(line):
            continue

        # 🔥 Clean repeated "Activity Activity Activity"
        line = re.sub(r'(Activity\s*){2,}', 'Activity ', line)

        # 🔥 Extract section
        section = extract_section(line)
        if section:
            current_section = section

        # 🔥 STRONG ACTIVITY DETECTION
        is_activity = re.search(r'Activity\s*[_\s]*\d+\.\d+', line)

        is_question = re.search(r'uestions|Questions', line)

        is_section = section is not None

        # 🚀 START NEW BLOCK on ANY anchor
        if is_activity or is_question or is_section:
            if current:
                blocks.append((current.strip(), current_section))
            current = line  # start fresh block
        else:
            current += " " + line

    if current:
        blocks.append((current.strip(), current_section))

    return blocks


def main():
    doc = fitz.open(PDF)
    data = []

    for i, page in enumerate(doc):
        raw = page.get_text()
        raw = clean_text(raw)

        lines = raw.split("\n")

        blocks = group_lines_into_blocks(lines)

        for block_text, section in blocks:
            entry = {
                "page": i+1,
                "section": section,   #correct section tracking
                "type": detect_type(block_text),
                "text": block_text
            }
            data.append(entry)

    out_file = output / "extracted_text.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[v4 FINAL] Wrote: {out_file}")

if __name__ == "__main__":
    main()