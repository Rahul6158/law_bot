#injest.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import fitz

# Load PDF file
pdf_path = '/content/ipc_law.pdf'
loader = PyPDFLoader(pdf_path)  # Load the PDF using PyPDFLoader
documents = loader.load()  # Load the PDF into a list of documents

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})

# Creates vector embeddings and saves it in the FAISS DB
faiss_db = FAISS.from_documents(documents, embeddings)

# Saves and exports the vector embeddings database
faiss_db.save_local("ipc_vector_db")
