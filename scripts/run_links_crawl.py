#!/usr/bin/env python
"""
Run crawling for a set of links (from a file or by extracting from ~/wiki) using
`scripts/crawl.py`. On failure retry a few times (default wait 5s between retries).
After all links are processed, call `scripts/ingest.py`.

Usage examples:
  # Use existing links file
  python scripts/run_links_crawl.py --links-file ~/links.txt --max-pages 5 --retries 3

  # Generate links from ~/wiki using the extractor, then crawl
  python scripts/run_links_crawl.py --wiki-dir ~/wiki --max-pages 5 --retries 3

Options:
  --links-file  path to newline-separated links. If omitted the script runs
                scripts/extract_category_links.py to produce links from --wiki-dir
  --wiki-dir    directory to scan when extractor is used (default ~/wiki)
  --max-pages   passed to crawl.py --max-pages
  --retries     number of attempts per link (default 3)
  --retry-delay seconds to wait between retries (default 5)
  --api-key     Firecrawl API key to pass to crawl.py (optional if env var present)
  --verbose     show command stdout/stderr
"""
from pathlib import Path
import subprocess
import sys
import os
import argparse
import time
from typing import List, Optional
from urllib.parse import urlparse
import json
import hashlib
from datetime import datetime
from datetime import UTC as _UTC

# Load .env early so os.getenv sees values defined there
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # optional service role
SUPABASE_TABLE = os.getenv("SUPABASE_LINKS_TABLE", "crawl_links")

_supabase = None


def get_supabase():
    global _supabase
    if _supabase is not None:
        return _supabase
    # import supabase lazily to avoid hard dependency
    try:
        from supabase import create_client
    except Exception:
        print("[Supabase] Python client not installed. Tracking disabled.")
        return None
    if not SUPABASE_URL:
        print("[Supabase] SUPABASE_URL not set. Tracking disabled.")
        return None
    key = SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
    if not key:
        print("[Supabase] No key provided (SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY). Tracking disabled.")
        return None
    try:
        _supabase = create_client(SUPABASE_URL, key)
        print(f"[Supabase] Client initialized. Table='{SUPABASE_TABLE}'")
        return _supabase
    except Exception as e:
        print(f"[Supabase] Init failed: {e}. Tracking disabled.")
        return None


def supabase_get_link_status(url: str) -> Optional[dict]:
    client = get_supabase()
    if not client:
        return None
    try:
        resp = client.table(SUPABASE_TABLE).select("*").eq("url", url).limit(1).execute()
        rows = getattr(resp, "data", []) or []
        if rows:
            print(f"[Supabase] Found existing row for url: {url} -> status={rows[0].get('status')}")
        return rows[0] if rows else None
    except Exception as e:
        print(f"[Supabase] Query failed for url={url}: {e}")
        return None


def supabase_upsert_link(url: str, status: str, note: str = "", output_files: Optional[List[str]] = None) -> None:
    client = get_supabase()
    if not client:
        return
    payload = {
        "url": url,
        "status": status,  # e.g., pending, crawled, failed, ingested
        "note": note,
        "output_files": json.dumps(output_files or []),
        "last_crawled_at": datetime.now(_UTC).isoformat(),
    }
    try:
        client.table(SUPABASE_TABLE).upsert(payload, on_conflict="url").execute()
        print(f"[Supabase] Upsert OK: url={url}, status={status}")
    except Exception as e:
        print(f"[Supabase] Upsert failed for url={url}: {e}")


def run_subprocess(cmd: List[str], verbose: bool = False) -> int:
    if verbose:
        print("Running:", " ".join(cmd))
    try:
        res = subprocess.run(cmd, check=False, capture_output=not verbose, text=True)
        if not verbose:
            if res.stdout:
                print(res.stdout.strip())
            if res.stderr:
                print(res.stderr.strip())
        return res.returncode
    except Exception as e:
        print(f"Error running command {' '.join(cmd)}: {e}")
        return 2


def read_links_file(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines


def generate_links_with_extractor(extractor_script: Path, wiki_dir: str, verbose: bool = False) -> List[str]:
    cmd = [sys.executable, str(extractor_script), "--wiki-dir", wiki_dir]
    if not verbose:
        # ensure extractor prints only links to stdout
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Extractor failed (rc={res.returncode}): {res.stderr.strip() if res.stderr else ''}")
            return []
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip() and ln.strip().startswith("http")]
        return lines
    else:
        # verbose: stream output
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Extractor failed (rc={res.returncode})")
            return []
        # if streaming we still need to read the file output; so rerun non-verbose to capture
        res2 = subprocess.run([sys.executable, str(extractor_script), "--wiki-dir", wiki_dir], check=False, capture_output=True, text=True)
        if res2.returncode != 0:
            return []
        return [ln.strip() for ln in res2.stdout.splitlines() if ln.strip() and ln.strip().startswith("http")]


def main():
    parser = argparse.ArgumentParser(description="Run crawl.py on a list of links with retries and then call ingest.py")
    parser.add_argument("--links-file", help="Path to newline-separated links. If omitted the extractor will be run.")
    parser.add_argument("--wiki-dir", default=os.path.expanduser("~/wiki"), help="Wiki dir to scan if extractor is used")
    parser.add_argument("--max-pages", type=int, default=10, help="Pass --max-pages to crawl.py")
    parser.add_argument("--retries", type=int, default=3, help="Number of tries per link (default 3)")
    parser.add_argument("--retry-delay", type=int, default=5, help="Seconds to wait between retries (default 5)")
    parser.add_argument("--api-key", help="Optional Firecrawl API key to pass to crawl.py")
    parser.add_argument("--verbose", action="store_true", help="Show subprocess output")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    extractor_script = repo_root / "scripts" / "extract_category_links.py"
    crawl_script = repo_root / "scripts" / "crawl.py"
    ingest_script = repo_root / "scripts" / "ingest.py"

    # Ensure repo root is importable so we can call ingestion functions directly
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import DATA_CRAWLED_DIR and ingestion helper
    try:
        from src.config import DATA_CRAWLED_DIR
        from src.utils.ingestion import ingest_files
    except Exception:
        DATA_CRAWLED_DIR = None
        ingest_files = None

    if args.links_file:
        links_path = Path(os.path.expanduser(args.links_file))
        if not links_path.exists():
            print(f"Links file not found: {links_path}")
            sys.exit(1)
        links = read_links_file(links_path)
    else:
        if not extractor_script.exists():
            print(f"Extractor script not found: {extractor_script}")
            sys.exit(1)
        print(f"Generating links by running extractor on {args.wiki_dir}...")
        links = generate_links_with_extractor(extractor_script, args.wiki_dir, verbose=args.verbose)

    if not links:
        print("No links to process.")
        return

    print(f"Found {len(links)} links. Starting crawling...")
    # Print Supabase status once
    if get_supabase():
        print("[Supabase] Tracking enabled")
    else:
        print("[Supabase] Tracking disabled")

    successes = []
    failures = []

    # Ensure data dir exists if available
    data_dir = Path(DATA_CRAWLED_DIR) if DATA_CRAWLED_DIR else None
    if data_dir:
        data_dir = data_dir if data_dir.is_absolute() else Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

    # Dedup index: map content_hash -> stored text file path
    dedup_index_path = data_dir / 'ingested_index.json' if data_dir else None
    dedup_index = {}
    if dedup_index_path and dedup_index_path.exists():
        try:
            dedup_index = json.loads(dedup_index_path.read_text(encoding='utf-8'))
        except Exception:
            dedup_index = {}

    for idx, link in enumerate(links, start=1):
        # Supabase: skip if already crawled/ingested
        sb_row = supabase_get_link_status(link)
        if sb_row and sb_row.get("status") in {"crawled", "ingested"}:
            print(f"[Skip] Already processed (Supabase): {link}")
            continue
        supabase_upsert_link(link, status="pending", note="queued")

        print(f"[{idx}/{len(links)}] Crawling: {link}")
        attempt = 0
        success = False
        output_files: List[str] = []
        while attempt < args.retries and not success:
            attempt += 1

            # prepare deterministic output filename for this crawl
            timestamp = int(time.time())
            parsed = urlparse(link)
            domain = (parsed.netloc or 'site').replace(':', '_')
            out_name = f"crawl_{idx:04d}_{domain}_{timestamp}.json"

            cmd = [sys.executable, str(crawl_script), "--url", link, "--mode", "links", "--max-pages", str(args.max_pages)]
            if args.api_key:
                cmd.extend(["--api-key", args.api_key])
            if data_dir:
                cmd.extend(["--output-dir", str(data_dir), "--output-file", out_name])

            # run
            rc = run_subprocess(cmd, verbose=args.verbose)
            if rc == 0:
                # if crawl succeeded, attempt to ingest the produced file into Qdrant
                success = True
                successes.append(link)
                print(f"Success: {link}")

                produced = data_dir / out_name if data_dir else None
                if produced and produced.exists():
                    output_files.append(str(produced))

                if data_dir and produced and produced.exists():
                    # Convert crawled JSON to plain text files (one per document) before ingestion
                    txt_files = []
                    try:
                        with open(produced, 'r', encoding='utf-8') as jf:
                            crawled = json.load(jf)
                    except Exception as e:
                        print(f"Failed to read crawl json {produced}: {e}")
                        crawled = None

                    if crawled and isinstance(crawled, dict):
                        docs = crawled.get('documents', [])
                        for j, doc in enumerate(docs, start=1):
                            content = doc.get('content') or doc.get('text') or ''
                            if not content or not content.strip():
                                continue
                            # compute content hash to avoid duplicates
                            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                            if content_hash in dedup_index:
                                # already ingested before, skip creating this text file
                                continue
                            # sanitize title for filename
                            title = (doc.get('title') or '').strip() or f'doc{j}'
                            title_safe = ''.join(c for c in title if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
                            txt_name = f"{produced.stem}_doc{j}_{title_safe}.txt"
                            txt_path = produced.with_name(txt_name)
                            try:
                                txt_path.write_text(content, encoding='utf-8')
                                dedup_index[content_hash] = str(txt_path)
                                txt_files.append(txt_path)
                                output_files.append(str(txt_path))
                            except Exception as e:
                                print(f"Failed to write text file {txt_path}: {e}")

                    # if no per-doc text files created, fall back to single aggregated text file
                    if not txt_files:
                        try:
                            aggregate = produced.with_name(f"{produced.stem}.txt")
                            with open(produced, 'r', encoding='utf-8') as jf:
                                crawled = json.load(jf)
                            parts = []
                            for doc in (crawled.get('documents', []) if isinstance(crawled, dict) else []):
                                parts.append(doc.get('content','') or '')
                            # dedupe aggregate by content hash
                            aggregate_text = '\n\n'.join([p for p in parts if p])
                            if aggregate_text.strip():
                                agg_hash = hashlib.sha256(aggregate_text.encode('utf-8')).hexdigest()
                                if agg_hash not in dedup_index:
                                    aggregate.write_text(aggregate_text, encoding='utf-8')
                                    dedup_index[agg_hash] = str(aggregate)
                                    txt_files = [aggregate]
                                    output_files.append(str(aggregate))
                                else:
                                    txt_files = []
                            else:
                                txt_files = []
                        except Exception as e:
                            print(f"Failed to create aggregate text file from {produced}: {e}")

                    if not txt_files:
                        print(f"No text files produced from {produced}; recording for later review.")
                        try:
                            failed_file = (Path(DATA_CRAWLED_DIR) if DATA_CRAWLED_DIR else Path('.')) / 'failed_ingest.txt'
                            failed_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(failed_file, 'a', encoding='utf-8') as fh:
                                fh.write(str(produced) + '\n')
                        except Exception as e:
                            print(f"Failed to record failed file: {e}")
                        supabase_upsert_link(link, status="crawled", note="no_text_files", output_files=output_files)
                    else:
                        # attempt to ingest via CLI script to centralize dedup and error handling
                        ingest_attempt = 0
                        ingest_ok = False
                        ingest_cmd = [sys.executable, str(ingest_script)] + [str(p) for p in txt_files] + ["--append"]
                        while ingest_attempt < args.retries and not ingest_ok:
                            ingest_attempt += 1
                            try:
                                print(f"Running ingest.py on {len(txt_files)} text file(s) (attempt {ingest_attempt})")
                                res = subprocess.run(ingest_cmd, check=False, capture_output=True, text=True)
                                if res.stdout:
                                    print(res.stdout.strip())
                                if res.stderr:
                                    print(res.stderr.strip())
                                if res.returncode == 0:
                                    ingest_ok = True
                                else:
                                    raise RuntimeError(f"ingest.py rc={res.returncode}")
                            except Exception as e:
                                print(f"ingest.py failed for {produced}: {e}")
                                if ingest_attempt < args.retries:
                                    print(f"Retrying ingest in {args.retry_delay}s...")
                                    time.sleep(args.retry_delay)
                        if not ingest_ok:
                            print(f"Giving up ingesting text files from {produced} after {args.retries} attempts via ingest.py")
                            try:
                                failed_file = (Path(DATA_CRAWLED_DIR) if DATA_CRAWLED_DIR else Path('.') ) / 'failed_ingest.txt'
                                failed_file.parent.mkdir(parents=True, exist_ok=True)
                                with open(failed_file, 'a', encoding='utf-8') as fh:
                                    for t in txt_files:
                                        fh.write(str(t) + '\n')
                                print(f"Recorded failed ingest file(s) for later retry: {failed_file}")
                            except Exception as e:
                                print(f"Failed to record failed ingest file: {e}")
                            supabase_upsert_link(link, status="failed", note="ingest_failed", output_files=output_files)
                        else:
                            supabase_upsert_link(link, status="ingested", note="ok", output_files=output_files)

                else:
                    print(f"Expected crawl output not found: {produced}")
                    supabase_upsert_link(link, status="failed", note="no_json")

            else:
                print(f"Attempt {attempt} failed for {link} (rc={rc}). Retrying in {args.retry_delay}s...")
                time.sleep(args.retry_delay)

        if not success:
            print(f"Failed after {args.retries} attempts: {link}")
            failures.append(link)
            supabase_upsert_link(link, status="failed", note="crawl_failed")

    print("\nCrawling complete.")
    print(f"  succeeded: {len(successes)}")
    print(f"  failed: {len(failures)}")

    # Final ingest step is no longer necessary since we ingest per-file during crawl.
    if failures:
        print("Some links failed to crawl. See the list above.")
        sys.exit(2)


if __name__ == '__main__':
    main()
