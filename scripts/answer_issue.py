#!/usr/bin/env python3
"""
GitHub Action step: answer the issue or comment using RAG
(FAISS + Hugging Face Inference API) and post back via the GitHub CLI.
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

# 3) Haal de vraag op uit het Issue of de Comment
q = os.getenv("COMMENT_BODY") or os.getenv("ISSUE_BODY")
if not q:
    print("No question found – skipping.")
    sys.exit(0)

# 4) Retrieval: zoek de top-4 relevante chunks
docs = vstore.similarity_search(q, k=4)
context = "\n\n".join([d.page_content for d in docs])

# 5) Bouw je prompt
prompt = f'''Beantwoord de volgende vraag op basis van de onderstaande informatie:

{context}

Vraag: {q}
Antwoord: '''

# 6) Generatie via HF Inference API (positional args)
response = client.text_generation(
    "mistralai/Mistral-7B-Instruct-v0.2",
    prompt,
    max_new_tokens=512,
    temperature=0.0
)
generated = response[0].generated_text

# Strip de prompt uit de output, zodat je alleen het antwoord overhoudt
if generated.startswith(prompt):
    ans = generated[len(prompt):].strip()
else:
    ans = generated.strip()

# 7) Plaats het antwoord als comment via de GitHub CLI
body = textwrap.dedent(f"""
**Answer (automated)**

{ans}
""")
issue_num = os.getenv("ISSUE_NUMBER")
repo = os.getenv("GITHUB_REPOSITORY")

subprocess.run([
    "gh", "api",
    f"/repos/{repo}/issues/{issue_num}/comments",
    "--method", "POST",
    "--field", f"body={body}"
], check=True)
