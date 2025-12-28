# Mr.PolicyHelperAI – Enhanced RAG System Using LangChain

A compact Retrieval‑Augmented Generation (RAG) pipeline built with **LangChain**, that ingests multiple insurance policy PDFs, chunks pages, embeds text using **HuggingFace `all‑MiniLM‑L6‑v2`**, stores vectors in **FAISS**, and answers queries via a LangChain **Retriever**.

---

## What it does
- **Parse PDFs** with `pdfplumber` across a folder (e.g., `Policy+Documents/`) and build a dataframe: *Page No.*, *Page_Text*, *Document Name*, and per‑page **Metadata**.
- **Fixed‑size chunking** (~500) per page to create retrieval units enriched with metadata (policy name, page no., chunk no.).
- **Embeddings**: `HuggingFaceEmbeddings('all‑MiniLM‑L6‑v2')` applied to each chunk.
- **Vector store**: `FAISS.from_texts(texts, metadatas, embedding=...)` to build the index.
- **Semantic search**: `insurance_collection.as_retriever()` to fetch top matches for a query (e.g., *“What is the policy on eye issues?”*).

---

## Files
- `Mr.PolicyHelperAI_LangChain_RAG_System_Project.ipynb` — end‑to‑end notebook/script: PDF parsing, chunking, HuggingFace embeddings, FAISS store, and LangChain retriever.
- 7 PDF documents from HDFC Life Insurance Policies

---

## Requirements
- Python 3.9+
- Packages: `pdfplumber`, `pandas`, `langchain`, `langchain-huggingface`, `faiss-cpu` (or `faiss`), `tiktoken`, `openai` (optional), plus standard libs.

Install (example):
```bash
pip install pdfplumber pandas langchain langchain-huggingface faiss-cpu tiktoken openai
```

---

## Quick start
1. **Place PDFs** under `Policy+Documents/`.
2. Run the script to:
   - extract pages & metadata,
   - chunk to ~500 tokens/words,
   - generate embeddings,
   - build the FAISS index,
   - query with `retriever.invoke("<your question>")`.

---

## Notes
- The dataframe includes page text length filtering to drop near‑empty pages before chunking.
- Metadata retained in FAISS (policy name, page, chunk) aids traceability in answers.

Maintainer: Gouri Karthik Gembali