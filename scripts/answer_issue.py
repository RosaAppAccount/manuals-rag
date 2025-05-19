"""GitHub Action step: answer the issue or comment using the RAG chain and post back."""
import os, pathlib, pickle, json, textwrap, subprocess, sys
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

vector_path = pathlib.Path("vectorstore.pkl")
if not vector_path.exists():
    print("Vectorstore missing – run ingest first.")
    sys.exit(1)

with vector_path.open("rb") as f:
    vstore = pickle.load(f)

llm = ChatOpenAI(model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vstore.as_retriever(search_kwargs={"k": 4}),
)

question = os.getenv("COMMENT_BODY") or os.getenv("ISSUE_BODY")
if not question:
    print("No question found.")
    sys.exit(0)

answer = qa_chain.run(question)
body = textwrap.dedent(f"""
**Answer (automated)**

{answer}
""")

issue_number = os.getenv("ISSUE_NUMBER")
subprocess.run([
    "gh", "api",
    f"repos/${{GITHUB_REPOSITORY}}/issues/{issue_number}/comments",
    "--method", "POST",
    "--field", f"body={body}"
], check=True)
