#!/usr/bin/env python3
"""
GitHub Action step: answer the issue or comment using RAG (FAISS + Hugging Face Inference API)
and post back via the GitHub CLI.
"""

import os
import sys
import pathlib
import pickle
import textwrap
import subprocess

from langchain_community.vectorstores import FAISS

# 1) HF token en InferenceClient
from huggingface_hub import InferenceClient

hf_token = os.getenv("HF_API_TOKEN")
if not hf_token:
    print("HF_API_TOKEN not set – aborting.")
    sys.exit(1)
client = InferenceClient(token=hf_token)

# 2) Laad je vectorstore
V = pathlib.Path("vectorstore.pkl")
if not V.exists():
    print("Vectorstore missing – run ingest first.")
    sys.exit(1)
with open(V, "rb") as f:
    vstore = pickle.load(f)

# 3) Haal de vraag op
q = os.getenv("COMMENT_BODY") or os.getenv("ISSUE_BODY")
if not q:
    print("No question found – skipping.")
    sys.exit(0)

# 4) Retrieval: vind top-k documenten
docs = vstore.similarity_search(q, k=4)
context = "\n\n".join([d.page_content for d in docs])

# 5) Bouw een prompt
prompt = f\"\"\"Beantwoord de volgende vraag op basis van de onderstaande informatie:

{context}

Vraag: {q}
Antwoord:\"\"\"

# 6) Generatie via HF Inference API
#    We gebruiken text_generation; pas parameters aan indien gewenst.
response = client.text_generation(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    inputs=prompt,
    parameters={"max_new_tokens": 512, "temperature": 0.0}
)
# Het antwoord zit in generated_text
ans = response[0].generated_text[len(prompt):].strip()

# 7) Post het comment terug
body = textwrap.dedent(f\"\"\"
**Answer (automated)**

{ans}
\"\"\")
issue_num = os.getenv("ISSUE_NUMBER")
repo = os.getenv("GITHUB_REPOSITORY")

subprocess.run([
    "gh", "api",
    f"/repos/{repo}/issues/{issue_num}/comments",
    "--method", "POST",
    "--field", f"body={body}"
], check=True)
