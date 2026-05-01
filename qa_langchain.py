import pickle
import sys
from pathlib import Path
import warnings

# Disable irrelevant huggingface/transformers warnings
warnings.filterwarnings("ignore")

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Config ────────
BM25_INDEX = Path("chunk/chunked_text.json/bm25_retriever.pkl")
QA_MODEL = "google/flan-t5-base"

def format_docs(docs):
    """Format matching documents into a single contextual string."""
    context_blocks = []
    word_count = 0
    MAX_WORDS = 600

    for d in docs:
        b_type = d.metadata.get("content_type", "unknown").upper()
        b_page = d.metadata.get("page", "?")
        b_sec = d.metadata.get("section", "?")
        text = d.page_content

        # Rough limit handling similarly to qa.py
        block_words = len(text.split())
        if word_count + block_words > MAX_WORDS:
            break
            
        context_blocks.append(f"[{b_type} | p.{b_page} | sec {b_sec}]\n{text}")
        word_count += block_words

    return "\n\n".join(context_blocks)

def main():
    print("Loading BM25 Retriever from disk...")
    if not BM25_INDEX.exists():
        print(f"Error: {BM25_INDEX} not found. Run chunkinglangchain.py first.")
        return

    with open(BM25_INDEX, "rb") as f:
        retriever = pickle.load(f)
    print("Loaded BM25 Index.")

    print(f"Loading QA Model: {QA_MODEL}")
    pipe = pipeline(
        "text2text-generation", 
        model=QA_MODEL, 
        max_new_tokens=256,
        model_kwargs={"num_beams": 4, "length_penalty": 1.5, "early_stopping": True, "no_repeat_ngram_size": 3}
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Note: the prompt matches the behavior implemented in qa.py
    prompt = PromptTemplate.from_template(
        "You are a science teacher answering student questions based strictly on "
        "the textbook passages below. Give a complete, factual answer.\n\n"
        "Textbook passages:\n{context}\n\n"
        "Student question: {question}\n\n"
        "Teacher answer:"
    )

    # ── LCEL Chain Construction ────────
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n=== End-to-End LangChain RAG pipeline initialized ===")
    
    # Interactive REPL
    if sys.stdin.isatty():
        print("\nType your question and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                q = input("Ask > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not q:
                continue
            if q.lower() in ("quit", "exit"):
                print("Bye!")
                break

            print("\nGenerating answer...")
            
            # Using the LCEL chain to stream or process
            response = rag_chain.invoke(q)
            
            # Fetch the raw documents as well to display sources (optional parallel execution capability in LCEL)
            docs = retriever.invoke(q)

            print("-" * 70)
            print(f"Q: {q}")
            print("-" * 70)
            print(f"A: {response}")
            print("\nSources:")
            for i, d in enumerate(docs[:3], 1):  # Display top 3 sources
                print(f"  [{i}] page={d.metadata.get('page')} sec={d.metadata.get('section')} type={d.metadata.get('content_type')}")
            print()

if __name__ == "__main__":
    main()
