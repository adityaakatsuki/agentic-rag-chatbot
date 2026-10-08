import glob
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "agentic_rag"

# --------------------------------------------------
# Embeddings Model
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_vectorstore(persist_directory: str = CHROMA_DIR) -> Chroma:
    """Load PDFs from documents/, split into chunks, and persist to Chroma."""
    pdf_files = glob.glob("documents/*.pdf")
    documents = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf_file

        documents.extend(docs)

    print(f"PDFs loaded: {len(pdf_files)}")
    print(f"Total pages: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory
    )
    print("Vector store created and saved to disk!")
    return vs


def get_vectorstore(persist_directory: str = CHROMA_DIR) -> Chroma:
    """Load vectorstore from disk if available, otherwise build it."""
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vector store from {persist_directory}...")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
    else:
        print("No existing vector store found. Building a new one...")
        return build_vectorstore(persist_directory)


# Initialize vectorstore and retriever
vectorstore = get_vectorstore(CHROMA_DIR)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)

print("RAG system ready!")

if __name__ == "__main__":
    print("Re-indexing documents...")
    build_vectorstore(CHROMA_DIR)