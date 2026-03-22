#injest.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Load PDF file
# Update path to be relative or absolute to the local environment
pdf_path = os.path.join(os.path.dirname(__file__), 'ipc_law.pdf')
loader = PyPDFLoader(pdf_path)  # Load the PDF using PyPDFLoader
documents = loader.load()  # Load the PDF into a list of documents

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Creates vector embeddings and saves it in the FAISS DB
faiss_db = FAISS.from_documents(documents, embeddings)

# Saves and exports the vector embeddings database
faiss_db.save_local("ipc_vector_db")
