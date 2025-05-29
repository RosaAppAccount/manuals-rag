#!/usr/bin/env python3
"""
GitHub Action step: answer the issue or comment using RAG
(FAISS + Hugging Face Inference API chat endpoint) and post back via the GitHub CLI.
"""

import os
import sys
import pathlib
import pickle
import textwrap
import subprocess

from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# 1) Initialiseer de HF InferenceClient met je token
hf_token = os.getenv("HF_API_TOKEN")
if not hf_token:
    print("HF_API_TOKEN not set – aborting.")
    sys.exit(1)
client = InferenceClient(token=hf_token)

# 2) Laad de FAISS-vectorstore
V = pathlib.Path("vectorstore.pkl")
if not V.exists():
    print("Vectorstore missing – run ingest first.")
    sys.exit(1)
with open(V, "rb") as f:
    vstore = pickle.load(f)

# 3) Haal de vraag op uit het Issue of de comment
q = os.getenv("COMMENT_BODY") or os.getenv("ISSUE_BODY")
if not q:
    print("No question found – skipping.")
    sys.exit(0)

# 4) Retrieval: zoek de top-4 relevante chunks
docs = vstore.similarity_search(q, k=4)
context = "\n\n".join([d.page_content for d in docs])

# 5) Bouw de chat-messages
messages = [
    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
    {"role": "user", "content": f"Beantwoord de volgende vraag op basis van de onderstaande informatie:\n\n{context}\n\nVraag: {q}"}
]

# 6) Genereer antwoord via chat.complete
response = client.chat.complete(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=messages
)
# Haal assistant content uit de eerste keuze
ans = response.choices[0].message.content.strip()

# 7) Post het antwoord als GitHub comment
decorated = textwrap.dedent(f"""
**Answer (automated)**

{ans}
""")
issue_num = os.getenv("ISSUE_NUMBER")
repo = os.getenv("GITHUB_REPOSITORY")

subprocess.run([
    "gh", "api",
    f"/repos/{repo}/issues/{issue_num}/comments",
    "--method", "POST",
    "--field", f"body={decorated}"
], check=True)
