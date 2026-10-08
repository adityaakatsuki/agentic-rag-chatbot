# Agentic RAG Chatbot

A lightweight RAG (Retrieval-Augmented Generation) chatbot that lets you ask questions about PDF documents stored in a local `documents/` folder. It uses LangChain, ChromaDB, Hugging Face embeddings, and Groq-hosted LLMs to search relevant document chunks and answer grounded questions.

## Features

- Upload and index PDF files from `documents/`
- Chunk and embed documents with `sentence-transformers`
- Search the document corpus using vector similarity
- Answer questions through a LangChain agent
- Includes a lightweight calculator tool for math queries
- Works with both:
  - Streamlit web UI
  - Terminal CLI

## Project Structure

```text
agentic-rag-chatbot/
├── app.py                 # Streamlit app and CLI runner
├── rag.py                 # PDF loading, chunking, embedding, and vector DB setup
├── tools.py               # RAG search tool and calculator tool
├── documents/             # Folder for PDF files
├── chroma_db/             # Persisted vector database
├── requirements.txt       # Python dependencies
├── .env                   # Local environment variables (not committed)
├── .streamlit/            # Streamlit config
└── README.md              # Project documentation
```

## Requirements

- Python 3.10+
- A valid Groq API key
- PDF files placed in the `documents/` directory

## Setup

1. Clone the repository
2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

5. Add your PDFs into the `documents/` folder.

6. Run the app:

```bash
streamlit run app.py
```

The app will automatically build or load the Chroma vector store from `chroma_db/`.

## CLI Usage

You can also run the app in terminal mode:

```bash
python app.py cli
```

Type a question and press Enter. Type `exit` to quit.

## How It Works

- PDFs are loaded from `documents/*.pdf`
- Text is split into chunks using a recursive text splitter
- Each chunk is embedded with `all-MiniLM-L6-v2`
- ChromaDB stores the embeddings for retrieval
- The LLM answers by using the retrieved context and the custom tools

## Notes

- The app searches only the documents in `documents/`.
- If a question is not found in the indexed PDFs, the system should say so clearly.
- The vector database is persisted in `chroma_db/`, so reindexing is not required unless files change.

## Environment Variables

The project uses:

- `GROQ_API_KEY` — required for the Groq-backed model

## Troubleshooting

### No response from the app
- Ensure your Groq API key is valid
- Confirm `documents/` contains at least one PDF
- Reinstall dependencies if imports fail

### Search returns nothing
- Rebuild the vector database by deleting or clearing `chroma_db/` and rerunning the app

## License

This project does not include a specific license file. If you plan to distribute or reuse it, add an appropriate open-source license before publishing.
