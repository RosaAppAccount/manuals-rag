#!/usr/bin/env python3
"""Create FAISS vectorstore from PDFs in ./manuals"""

import os
import pathlib
import pickle

# Ensure Hugging Face token is set for langchain_huggingface
hf_token = os.getenv("HF_API_TOKEN")
if not hf_token:
    raise RuntimeError("HF_API_TOKEN not set. Add your Hugging Face token as secret 'HF_API_TOKEN'.")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Community & huggingface-based imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Commented for future reference:
# from langchain.embeddings import OpenAIEmbeddings  # optional OpenAI usage


def main():
    # Directory where you upload your PDF manuals
    DATA_DIR = pathlib.Path("manuals")
    assert DATA_DIR.exists(), (
        "Put your manuals (PDF) inside the ./manuals directory"
    )

    # Load all PDF documents
    docs = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    # Split into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150"))
    )
    splits = splitter.split_documents(docs)

    # --- Use free Hugging Face embeddings ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Original OpenAI embeddings (commented out) ---
    # embeddings = OpenAIEmbeddings()

    # Build FAISS index
    vstore = FAISS.from_documents(splits, embeddings)
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vstore, f)
    print(f"Vectorstore saved with {len(splits)} chunks.")


if __name__ == "__main__":
    main()
