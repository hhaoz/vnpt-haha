"""Knowledge base ingestion utilities for Qdrant vector store."""

from pathlib import Path

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import DATA_INPUT_DIR, settings

_embeddings: HuggingFaceEmbeddings | None = None
_qdrant_client: QdrantClient | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create embeddings model singleton."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_qdrant_client() -> QdrantClient:
    """Get or create persistent Qdrant client singleton."""
    global _qdrant_client
    if _qdrant_client is None:
        db_path = settings.vector_db_path_resolved
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(db_path))
    return _qdrant_client


def load_knowledge_base(file_path: Path | None = None) -> str:
    """Load knowledge base text file."""
    if file_path is None:
        file_path = DATA_INPUT_DIR / "knowledge_base.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        return f.read()


def ingest_knowledge_base(file_path: Path | None = None, force: bool = False) -> QdrantVectorStore:
    """Ingest knowledge base into Qdrant vector store.
    
    Args:
        file_path: Path to knowledge base file. Defaults to DATA_INPUT_DIR/knowledge_base.txt.
        force: If True, re-ingest even if collection exists. Defaults to False.
    
    Returns:
        QdrantVectorStore instance.
    """
    embeddings = get_embeddings()
    client = get_qdrant_client()

    collection_exists = client.collection_exists(settings.qdrant_collection)

    if collection_exists and not force:
        print(f"Loading existing vector store from disk: {settings.vector_db_path_resolved}")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=embeddings,
        )
        return vector_store

    if collection_exists and force:
        print(f"Force re-ingesting: deleting existing collection '{settings.qdrant_collection}'")
        client.delete_collection(settings.qdrant_collection)

    print(f"Ingesting knowledge base into collection '{settings.qdrant_collection}'...")
    text = load_knowledge_base(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_text(text)

    sample_embedding = embeddings.embed_query("test")
    vector_size = len(sample_embedding)

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )

    vector_store.add_texts(chunks, batch_size=64)

    print(f"Ingested {len(chunks)} chunks into collection '{settings.qdrant_collection}'")
    return vector_store

