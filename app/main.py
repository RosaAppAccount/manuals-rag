from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
# from langchain.chat_models import ChatOpenAI  # Gecommentarieerd voor later
from langchain.chains import RetrievalQA

import os, pickle, pathlib

app = FastAPI(title="Manuals QA (RAG)")

class Question(BaseModel):
    query: str

# Pad naar de vooraf gebouwde FAISS-vectorstore
VSTORE_PATH = pathlib.Path("vectorstore.pkl")
if not VSTORE_PATH.exists():
    raise RuntimeError("Vectorstore not found. Run scripts/ingest.py first.")

# Laad de vectorstore
with VSTORE_PATH.open("rb") as f:
    vectorstore = pickle.load(f)

# --- Gratis HF-model (Mistral-7B-Instruct) ---
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
    model_kwargs={"temperature": 0, "max_length": 1024}
)

# --- Oorspronkelijke OpenAI-client als comment voor later ---
# llm = ChatOpenAI(
#     model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
#     temperature=0
# )

# Opzet van de RetrievalQA-chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
)

@app.post("/ask")
def ask(q: Question):
    try:
        answer = qa_chain.run(q.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
