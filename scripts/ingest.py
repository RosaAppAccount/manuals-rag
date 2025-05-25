#!/usr/bin/env python3
"""Create FAISS vectorstore from PDFs in ./manuals"""
import os
import pathlib
import pickle

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings  # Commented for potential future OpenAI usage
from langchain.embeddings import HuggingFaceEmbeddings


def main():
    # Directory where you upload your PDF manuals
    DATA_DIR = pathlib.Path("manuals")
    assert DATA_DIR.exists(), (
        "Put your manuals (PDF) inside the ./manuals directory"
    )

    # Load documents
    docs = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150"))
    )
    splits = splitter.split_documents(docs)

    # --- Use free Hugging Face embeddings ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_TOKEN")
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
