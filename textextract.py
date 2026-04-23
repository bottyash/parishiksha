import fitz
doc = fitz.open("pdfs/ch1.pdf")

for page in doc:
    text = page.get_text()
    print(text)
    print("\n\n")

    with open("extracted_text/extracted_text.txt", "a", encoding="utf-8") as file:
        file.write(text)


