import glob

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load environment variables
load_dotenv()


# --------------------------------------------------
# 1. Find all PDF files
# --------------------------------------------------

pdf_files = glob.glob("documents/*.pdf")

documents = []

for pdf_file in pdf_files:

    loader = PyPDFLoader(pdf_file)

    docs = loader.load()

    # Save PDF filename in metadata
    for doc in docs:
        doc.metadata["source"] = pdf_file

    documents.extend(docs)


print("PDFs loaded:", len(pdf_files))
print("Total pages:", len(documents))


# --------------------------------------------------
# 2. Split documents into chunks
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 4. Create Chroma vector store
# --------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="agentic_rag"
)


# --------------------------------------------------
# 5. Create MMR retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)


print("RAG system ready!")