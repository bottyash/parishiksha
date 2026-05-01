import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader

IN_FILE = "extracted_text/extracted_text.json"
OUT_DIR = Path("chunk/chunked_text.json")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TXT_OUT_FILE = "extracted_text/extracted_text.txt"

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["page"] = record.get("page")
    metadata["section"] = record.get("section", "unknown")
    metadata["type"] = record.get("type", "concept")
    return metadata

def chunk_docs(documents):
    prose_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    example_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    question_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    structured_chunks = []
    
    for doc in documents:
        ctype = doc.metadata.get("type", "concept")
        text = doc.page_content
        
        if ctype == "activity" or ctype == "worked_example":
            splitter = example_splitter
            ctype_label = "worked_example"
        elif ctype == "question" or ctype == "exercise":
            splitter = question_splitter
            ctype_label = "question_or_exercise"
        else:
            splitter = prose_splitter
            ctype_label = "prose"
            
        splits = splitter.split_text(text)
        
        for c in splits:
            structured_chunks.append({
                "chunk_id": len(structured_chunks),
                "page": doc.metadata.get("page"),
                "section": doc.metadata.get("section", "unknown"),
                "content_type": ctype_label,
                "text": c
            })
            
    return structured_chunks

def load_and_write_txt(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Error: {file_path} not found.")

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        content_key="text",
        metadata_func=metadata_func
    )

    documents = []
    print(f"Lazy loading documents and writing to {TXT_OUT_FILE}...")
    with open(TXT_OUT_FILE, "w", encoding="utf-8") as f:
        for doc in loader.lazy_load():
            f.write(doc.page_content + "\n\n")
            documents.append(doc)
    
    print(f"Loaded {len(documents)} documents.")
    return documents

def build_bm25_retriever(chunks):
    out_file = OUT_DIR / "chunks_langchain.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[langchain-structured] Wrote: {out_file} | total chunks={len(chunks)}")

    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    import pickle
    
    print("Building BM25 Retriever from chunks...")
    final_docs = [Document(page_content=c["text"], metadata=c) for c in chunks]
    
    try:
        retriever = BM25Retriever.from_documents(final_docs)
        bm25_out = OUT_DIR / "bm25_retriever.pkl"
        with open(bm25_out, "wb") as f:
            pickle.dump(retriever, f)
        print(f"[langchain-structured] Saved BM25 Retriever to {bm25_out}")
        return "SUCCESS"
    except ImportError:
        print("Error: Could not build BM25Retriever. Make sure 'rank_bm25' is installed: pip install rank_bm25")
        return "FAILED"

def main():
    from langchain_core.runnables import RunnableLambda

    # Compose the LCEL Extraction & Chunking Chain
    ingestion_chain = (
        RunnableLambda(load_and_write_txt)
        | RunnableLambda(chunk_docs)
        | RunnableLambda(build_bm25_retriever)
    )

    print("\n--- Executing LCEL Ingestion Chain ---")
    retriever = ingestion_chain.invoke(IN_FILE)
    
    if retriever:
        print("Ingestion Chain completed successfully!")

if __name__ == "__main__":
    main()
