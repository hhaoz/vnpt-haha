#!/usr/bin/env python
"""CLI script to crawl websites for RAG data.

Modes:
  single  - Scrape only the given URL
  links   - Scrape URL + all links found on that page
  search  - Use map API to find URLs by topic, then scrape
  domain  - Use crawl API to crawl entire domain
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import DATA_CRAWLED_DIR
from src.utils.web_crawler import crawl_website, save_crawled_data

EPILOG = """
Examples:
  python scripts/crawl.py --url https://example.com/article --mode single
  python scripts/crawl.py --url https://example.com --mode links --topic "keyword1,keyword2" --max-pages 20
  python scripts/crawl.py --url https://example.com --mode search --topic "history"
  python scripts/crawl.py --url https://example.com --mode domain --max-pages 50
"""


def main():
    parser = argparse.ArgumentParser(
        description="Crawl websites for RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("--url", required=True, help="Website URL to crawl")
    parser.add_argument("--mode", choices=["single", "links", "search", "domain"], 
                        default="links", help="Crawl mode (default: links)")
    parser.add_argument("--topic", help="Topic filter (required for links/search mode)")
    parser.add_argument("--max-pages", type=int, default=10, help="Max pages (default: 10)")
    parser.add_argument("--output-dir", default=str(DATA_CRAWLED_DIR), help="Output directory")
    parser.add_argument("--output-file", help="Output filename (auto if not set)")
    parser.add_argument("--api-key", help="Firecrawl API key")

    args = parser.parse_args()
    api_key = args.api_key or os.getenv("FIRECRAWL_API_KEY")
    
    if not api_key:
        print("[Error] Firecrawl API key required (--api-key or FIRECRAWL_API_KEY env)")
        sys.exit(1)
    
    if args.mode == "search" and not args.topic:
        print("[Error] --topic required for search mode")
        sys.exit(1)
    
    if args.mode == "links" and not args.topic:
        print("[Error] --topic required for links mode (keywords separated by comma)")
        sys.exit(1)

    try:
        data = crawl_website(
            url=args.url,
            mode=args.mode,
            topic=args.topic,
            max_pages=args.max_pages,
            api_key=api_key,
        )
        
        output_path = save_crawled_data(data, args.output_dir, args.output_file)
        print(f"\n[Done] Output: {output_path}")
        print(f"[Done] Documents: {data['total_pages']}")
        
    except KeyboardInterrupt:
        print("\n[Cancelled]")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
