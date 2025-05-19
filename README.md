# Manuals-QA (RAG) Demo

This repository spins up a **Retrieval‑Augmented Generation** (RAG) API over your PDF manuals with **zero local installs**.  
Everything runs in:

1. **GitHub Codespaces** – instant dev environment.  
2. **Docker / GHCR** – automated container image build & publish via GitHub Actions.

## Quick Start

### 1 ↳ Create a codespace  
- Click **Code → Codespaces → “+”** to launch a new codespace on branch `main`.  
- When prompted, add your `OPENAI_API_KEY` under **Codespace Secrets** (or edit later via _Codespace → Secrets_).

### 2 ↳ Add PDF manuals  
```bash
mkdir -p data
# drag‑and‑drop – or:
wget https://example.com/my_manual.pdf -O data/my_manual.pdf
```

### 3 ↳ Ingest embeddings  
```bash
python scripts/ingest.py
```

### 4 ↳ Run the API  
```bash
uvicorn app.main:app --reload --port 8000 --host 0.0.0.0
```
Open **`http://localhost:8000/docs`** in the Codespace preview.

## Repository Layout
```
.
├── app/                ← FastAPI application
│   └── main.py
├── scripts/
│   └── ingest.py       ← builds FAISS vectorstore
├── data/               ← place your manuals (ignored by git)
├── requirements.txt
├── Dockerfile
├── .devcontainer/
│   └── devcontainer.json
└── .github/workflows/
    └── docker-publish.yml
```

## Deploying a Container Image
Every push on **`main`** triggers _docker‑publish_ to build & push  
`ghcr.io/<owner>/<repo>:latest`. You can then run it on any cloud VM
(Azure Container Apps, Fly.io, AWS AppRunner, etc.):

```bash
docker run -p 8000:8000 ghcr.io/<owner>/<repo>:latest
```

## Configuration
| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | Your OpenAI key for embeddings + LLM | — |
| `MODEL_NAME` | Chat model (e.g. `gpt-4o-mini`) | `gpt-4o-mini` |
| `CHUNK_SIZE` | Token length per chunk | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 150 |

**Happy querying!**
