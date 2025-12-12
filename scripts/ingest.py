#!/usr/bin/env python
"""CLI script to ingest documents into Qdrant vector database.

Supports:
- JSON files from web crawler
- PDF files
- DOCX files
- TXT files
- Directory of files

Adds:
- Deduplication by content hash (skips files already ingested)
- Persists dedup index at DATA_CRAWLED_DIR/ingested_index.json
"""

import argparse
import sys
import os
import json
import hashlib
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import settings, DATA_CRAWLED_DIR
from src.utils.ingestion import ingest_files

EPILOG = """
Examples:
  python scripts/ingest.py data/crawl/file.json
  python scripts/ingest.py data/crawl/*.json --append
  python scripts/ingest.py documents/report.pdf --collection my_collection
  python scripts/ingest.py --dir data/documents
"""


def _read_all_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except Exception:
        try:
            return path.read_text(encoding='utf-8', errors='ignore').encode('utf-8')
        except Exception:
            return b""


def _compute_hash(path: Path) -> str:
    data = _read_all_bytes(path)
    return hashlib.sha256(data).hexdigest()


def _load_dedup_index(index_path: Path) -> dict:
    if not index_path.exists():
        return {}
    try:
        return json.loads(index_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _save_dedup_index(index_path: Path, index: dict) -> None:
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant vector database (with dedup by content)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("files", nargs="*", help="Files to ingest (JSON, PDF, DOCX, TXT)")
    parser.add_argument("--dir", help="Directory containing files to ingest")
    parser.add_argument("--collection", help=f"Collection name (default: {settings.qdrant_collection})")
    parser.add_argument("--append", action="store_true", help="Append to existing collection")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable deduplication")
    args = parser.parse_args()

    # Prefer local embeddings if VNPT API is not reachable (optional): users can override via env
    os.environ.setdefault("USE_VNPT_API", "false")

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

    # Dedup: compute content hashes and skip files already ingested
    dedup_index_path = Path(DATA_CRAWLED_DIR) / "ingested_index.json"
    dedup_index = _load_dedup_index(dedup_index_path) if not args.no_dedupe else {}

    unique_files: list[Path] = []
    skipped = 0
    for p in file_paths:
        h = _compute_hash(p)
        if dedup_index and h in dedup_index:
            skipped += 1
            continue
        unique_files.append(p)

    if not unique_files:
        print("[Ingest] No new files to ingest (all are duplicates)")
        sys.exit(0)

    print(f"[Ingest] Files to process: {len(unique_files)} (skipped {skipped} duplicates)")
    print("-" * 40)

    try:
        ingest_files(unique_files, args.collection, args.append)
        # Update dedup index on success
        if not args.no_dedupe:
            for p in unique_files:
                dedup_index[_compute_hash(p)] = str(p)
            _save_dedup_index(dedup_index_path, dedup_index)
        print("\n[Done]")
    except KeyboardInterrupt:
        print("\n[Cancelled]")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
