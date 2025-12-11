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
import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import settings
from src.utils.ingestion import ingest_files

EPILOG = """
Examples:
  python scripts/ingest.py data/crawl/file.json
  python scripts/ingest.py data/crawl/*.json --append
  python scripts/ingest.py documents/report.pdf --collection my_collection
  python scripts/ingest.py --dir data/documents
"""


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("files", nargs="*", help="Files to ingest (JSON, PDF, DOCX, TXT)")
    parser.add_argument("--dir", help="Directory containing files to ingest")
    parser.add_argument("--collection", help=f"Collection name (default: {settings.qdrant_collection})")
    parser.add_argument("--append", action="store_true", help="Append to existing collection")
    
    args = parser.parse_args()
    
    file_paths = []
    
    if args.files:
        for f in args.files:
            path = Path(f)
            if path.exists():
                file_paths.append(path)
            else:
                print(f"[Warning] File not found: {f}")
    
    if args.dir:
        dir_path = Path(args.dir)
        if dir_path.is_dir():
            for ext in ["*.json", "*.pdf", "*.docx", "*.txt"]:
                file_paths.extend(dir_path.glob(ext))
        else:
            print(f"[Error] Directory not found: {args.dir}")
            sys.exit(1)
    
    if not file_paths:
        parser.print_help()
        print("\n[Error] No files specified")
        sys.exit(1)
    
    print(f"[Ingest] Files to process: {len(file_paths)}")
    print("-" * 40)
    
    try:
        ingest_files(file_paths, args.collection, args.append)
        print("\n[Done]")
    except KeyboardInterrupt:
        print("\n[Cancelled]")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
