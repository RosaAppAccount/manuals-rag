#!/usr/bin/env python3
"""
GitHub Action step: answer the issue or comment using the RAG chain and post back.
"""

import os
import pathlib
import pickle
import textwrap
import subprocess
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Ensure HF token is set
hf_token = os.getenv("HF_API_TOKEN")
if not hf_token:
    print("HF_API_TOKEN not set; aborting.")
    sys.exit(1)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Load vectorstore
VECTOR_PATH = pathlib.Path("vectorstore.pkl")
if not VECTOR_PATH.exists():
    print("Vectorstore missing – run ingest first.")
    sys.exit(1)

with VECTOR_PATH.open("rb") as f:
    vstore = pickle.load(f)

# Initialize the Retrieval-QA chain with Mistral-7B
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature": 0, "max_length": 1024}
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vstore.as_retriever(search_kwargs={"k": 4}),
)

# Determine the question
question = os.getenv("COMMENT_BODY") or os.getenv("ISSUE_BODY")
if not question:
    print("No question found.")
    sys.exit(0)

# Run the chain
answer = qa_chain.run(question)

# Post back as a comment
body = textwrap.dedent(f"""
**Answer (automated)**

{answer}
""")

issue_number = os.getenv("ISSUE_NUMBER")
subprocess.run([
    "gh", "api",
    f"repos/{os.getenv('GITHUB_REPOSITORY')}/issues/{issue_number}/comments",
    "--method", "POST",
    "--field", f"body={body}"
], check=True)
