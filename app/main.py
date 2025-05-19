from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import os, pickle, pathlib

app = FastAPI(title="Manuals QA (RAG)")

class Question(BaseModel):
    query: str

VSTORE_PATH = pathlib.Path("vectorstore.pkl")
if not VSTORE_PATH.exists():
    raise RuntimeError("Vectorstore not found. Run scripts/ingest.py first.")

with VSTORE_PATH.open("rb") as f:
    vectorstore = pickle.load(f)

llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0)

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
