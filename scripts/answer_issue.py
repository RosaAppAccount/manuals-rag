#!/usr/bin/env python3
"""
GitHub Action step: answer the issue or comment using the RAG chain and post back.
"""
import os, pathlib, pickle, textwrap, subprocess, sys

# Zet het HF‐token in de omgevingsvariabelen voor langchain_huggingface
hf = os.getenv("HF_API_TOKEN")
if not hf:
    print("HF_API_TOKEN not set – aborting.")
    sys.exit(1)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf

# Imports vanuit de community‐packages
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Inladen vectorstore
V = pathlib.Path("vectorstore.pkl")
if not V.exists():
    print("Vectorstore missing – run ingest first.")
    sys.exit(1)
with open(V, "rb") as f:
    vstore = pickle.load(f)

# INIT RAG‐chain
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=hf,
    model_kwargs={"temperature": 0, "max_length": 1024}
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vstore.as_retriever(search_kwargs={"k": 4}),
)

# Vraag ophalen
q = os.getenv("COMMENT_BODY") or os.getenv("ISSUE_BODY")
if not q:
    print("No question found, skipping.")
    sys.exit(0)

# Antwoord genereren
# we gaan eerst debuggen ans = qa.run(q)
import traceback

try:
    ans = qa.run(q)
except Exception as e:
    print("⚠️ Error tijdens QA-run:")
    traceback.print_exc()
    sys.exit(1)

# Comment back
body = textwrap.dedent(f"""
**Answer (automated)**

{ans}
""")
num = os.getenv("ISSUE_NUMBER")
repo = os.getenv("GITHUB_REPOSITORY")
subprocess.run([
    "gh", "api", f"repos/{repo}/issues/{num}/comments",
    "--method", "POST", "--field", f"body={body}"
], check=True)
