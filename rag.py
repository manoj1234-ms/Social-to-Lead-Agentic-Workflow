import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def _setup_rag():
    if not os.path.exists("knowledge.md"):
        print("Knowledge base file missing.", file=sys.stderr)
        return None
    
    with SuppressOutput():
        loader = TextLoader("knowledge.md")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        # Use HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        retriever_obj = db.as_retriever(search_kwargs={"k": 2})
        
    return retriever_obj

# Initialize globally when imported
retriever = _setup_rag()

@tool
def search_knowledge_base(query: str) -> str:
    """Useful for answering questions about AutoStream's pricing, features, and policies."""
    if not retriever:
        return "Knowledge base is unavailable."
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])
