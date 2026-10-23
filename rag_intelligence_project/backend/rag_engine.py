"""
rag_engine.py - Core Ingestion, Indexing, and Retrieval Augmented Generation logic.
"""
import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Use a lightweight, high-performance open-source embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_db")

class RAGEngine:
    def __init__(self, persist_dir: str = PERSIST_DIRECTORY):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def ingest_document(self, file_path: str, filename: str) -> int:
        """Parses, chunks, and indexes a PDF or TXT document into ChromaDB."""
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        else:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

        # Add explicit metadata
        for d in docs:
            d.metadata["source_name"] = filename

        chunks = self.text_splitter.split_documents(docs)
        self.vector_store.add_documents(chunks)
        self.vector_store.persist()
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Performs similarity search with relevance scoring."""
        results_with_scores = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        retrieved_contexts = []
        for doc, score in results_with_scores:
            retrieved_contexts.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source_name", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "relevance_score": round(float(score), 4)
            })
        return retrieved_contexts

    def generate_response(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Synthesizes grounded response using prompt constraints to avoid hallucinations.
        If using local LLM (e.g. Ollama/HuggingFace) or API (OpenAI/Anthropic/Gemini).
        """
        context_str = "\n\n".join(
            [f"[Source: {c['source']}, Page {c['page']}]\n{c['content']}" for c in contexts]
        )
        
        system_prompt = (
            "You are an enterprise AI assistant answering questions based strictly on provided context.\n"
            "Rules:\n"
            "1. Rely only on the provided context. If the answer is not contained within, state: 'The provided documents do not contain sufficient information to answer this question.'\n"
            "2. Always cite the source file and page numbers inline (e.g., [Doc.pdf, p. 2]).\n"
            "3. Keep your answers concise, structured, and factual.\n\n"
            f"Context:\n{context_str}"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        # Demonstrates execution pipeline / fallback simulation if no remote LLM key is configured
        formatted_prompt = prompt_template.format(query=query)
        
        # When plugged into an LLM (e.g., ChatOllama or ChatOpenAI):
        # return (prompt_template | llm | StrOutputParser()).invoke({"query": query})
        return formatted_prompt
