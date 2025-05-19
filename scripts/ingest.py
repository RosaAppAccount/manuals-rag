"""Create FAISS vectorstore from PDFs in ./data"""
import pathlib, pickle, os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

DATA_DIR = pathlib.Path("data")
assert DATA_DIR.exists(), "Put your manuals (PDF) inside the ./data directory"

docs = []
for pdf_path in DATA_DIR.glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_path))
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size= int(os.getenv("CHUNK_SIZE", "1000")),
                                          chunk_overlap=int(os.getenv("CHUNK_OVERLAP","150")))
splits = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vstore = FAISS.from_documents(splits, embeddings)
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vstore, f)

print(f"Vectorstore saved with {len(splits)} chunks.")
