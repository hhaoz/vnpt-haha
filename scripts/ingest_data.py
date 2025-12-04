#!/usr/bin/env python
"""CLI script to ingest documents into Qdrant vector database.

Supports:
- JSON files from web crawler
- PDF files
- DOCX files
- TXT files
- Directory of files
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.utils.ingestion import (
    get_embeddings,
    get_qdrant_client,
    ingest_from_crawled_data,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams


def load_pdf(file_path: Path) -> str:
    """Load text from PDF file."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(file_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        print("Install pypdf: pip install pypdf")
        sys.exit(1)


def load_docx(file_path: Path) -> str:
    """Load text from DOCX file."""
    try:
        import docx
        doc = docx.Document(str(file_path))
        return "\n".join([para.text for para in doc.paragraphs])
    except ImportError:
        print("Install python-docx: pip install python-docx")
        sys.exit(1)


def load_txt(file_path: Path) -> str:
    """Load text from TXT file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_file(file_path: Path) -> tuple[str, dict]:
    """Load file and return (text, metadata)."""
    ext = file_path.suffix.lower()
    
    if ext == ".json":
        # Crawled JSON - handled separately
        return None, None
    elif ext == ".pdf":
        text = load_pdf(file_path)
    elif ext == ".docx":
        text = load_docx(file_path)
    elif ext == ".txt":
        text = load_txt(file_path)
    else:
        print(f"Unsupported file type: {ext}")
        return None, None
    
    metadata = {
        "source_file": str(file_path),
        "file_name": file_path.name,
        "file_type": ext[1:],
    }
    return text, metadata


def ingest_files(
    file_paths: list[Path],
    collection_name: str | None = None,
    append: bool = False,
) -> int:
    """Ingest multiple files into Qdrant."""
    embeddings = get_embeddings()
    client = get_qdrant_client()
    
    coll_name = collection_name or settings.qdrant_collection
    collections = [c.name for c in client.get_collections().collections]
    
    # Create or recreate collection
    if not append or coll_name not in collections:
        if coll_name in collections:
            client.delete_collection(coll_name)
        
        sample_embedding = embeddings.embed_query("test")
        vector_size = len(sample_embedding)
        
        client.create_collection(
            collection_name=coll_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=coll_name,
        embedding=embeddings,
    )
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    
    total_chunks = 0
    total_docs = 0
    
    for file_path in file_paths:
        if file_path.suffix.lower() == ".json":
            # Handle crawled JSON
            try:
                ingest_from_crawled_data(file_path, coll_name, append=True)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                total_docs += len(data.get("documents", []))
                print(f"✓ {file_path.name}")
            except Exception as e:
                print(f"✗ {file_path.name}: {e}")
            continue
        
        # Handle other file types
        text, metadata = load_file(file_path)
        if not text:
            continue
        
        chunks = splitter.split_text(text)
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = i
            chunk_meta["total_chunks"] = len(chunks)
            metadatas.append(chunk_meta)
        
        vector_store.add_texts(chunks, metadatas=metadatas)
        total_chunks += len(chunks)
        total_docs += 1
        print(f"✓ {file_path.name} ({len(chunks)} chunks)")
    
    print(f"\nTotal: {total_docs} documents, {total_chunks} chunks")
    print(f"Collection: '{coll_name}'")
    return total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single JSON file from crawler
  python ingest_data.py data/crawled/file.json
  
  # Ingest multiple files
  python ingest_data.py data/crawled/*.json
  
  # Ingest a PDF file
  python ingest_data.py documents/report.pdf
  
  # Ingest all files in a directory
  python ingest_data.py --dir data/documents
  
  # Append to existing collection
  python ingest_data.py data/crawled/*.json --append
  
  # Specify collection name
  python ingest_data.py data/crawled/*.json --collection my_collection
        """
    )
    parser.add_argument("files", nargs="*", help="Files to ingest (JSON, PDF, DOCX, TXT)")
    parser.add_argument("--dir", help="Directory containing files to ingest")
    parser.add_argument("--collection", help=f"Collection name (default: {settings.qdrant_collection})")
    parser.add_argument("--append", action="store_true", help="Append to existing collection")
    
    args = parser.parse_args()
    
    # Gather files to process
    file_paths = []
    
    if args.files:
        for f in args.files:
            path = Path(f)
            if path.exists():
                file_paths.append(path)
            else:
                print(f"Warning: File not found: {f}")
    
    if args.dir:
        dir_path = Path(args.dir)
        if dir_path.is_dir():
            for ext in ["*.json", "*.pdf", "*.docx", "*.txt"]:
                file_paths.extend(dir_path.glob(ext))
        else:
            print(f"Error: Directory not found: {args.dir}")
            sys.exit(1)
    
    if not file_paths:
        parser.print_help()
        print("\nError: No files specified")
        sys.exit(1)
    
    print(f"Files to ingest: {len(file_paths)}")
    print("-" * 40)
    
    try:
        ingest_files(file_paths, args.collection, args.append)
        print("\nDone!")
    except KeyboardInterrupt:
        print("\nCancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
