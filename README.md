# RAG_intelligence_project
problem solved - Enterprise documents—such as policy manuals, reports, and internal documentation—are unstructured and difficult to search accurately. Simple keyword searches fail on semantic meaning, and generic LLMs frequently hallucinate when asked about proprietary data.

I engineered an end-to-end RAG pipeline using Python and FastAPI. For ingestion, documents are parsed and split via recursive character chunking with overlap to preserve contextual continuity. Chunks are embedded using SentenceTransformers and stored in ChromaDB.

structure:
rag_project/
├── backend/
│   ├── rag_engine.py       # Document chunking, embeddings, vector indexing, retrieval logic
│   ├── app.py              # FastAPI REST endpoints (/upload, /query, /health)
│   ├── uploads/            # Temporary storage for incoming PDF/text files
│   └── chroma_db/          # Persistent ChromaDB vector database
├── frontend/
│   └── RAGDashboard.jsx    # React.js UI showing chat interface + retrieved citation drawer
└── requirements.txt        # Core dependencies

1. Setup Backend
cd rag_project/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install fastapi uvicorn langchain langchain-community chromadb sentence-transformers pypdf
python app.py

2. Setup Frontend
npm create vite@latest frontend -- --template react
cd frontend
npm install
# Place RAGDashboard.jsx into src/ and render it in App.jsx
npm run dev

other info:
When a query arrives, we execute top-$k$ similarity search with relevance scoring. The retrieved context chunks and metadata (document name, page number) are injected into a constrained prompt template. This guarantees the LLM answers strictly based on the provided text and provides clear inline citations, directly addressing enterprise data trust and governance.

The service is exposed via clean REST endpoints and connected to a React.js dashboard featuring a side-by-side citation viewer so users can independently verify the ground-truth text.
