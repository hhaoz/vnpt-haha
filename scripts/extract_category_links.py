#!/usr/bin/env python
"""
Extract category links from HTML files under a directory (default: ~/wiki).

Logic follows the provided JS snippet:
- Find divs matching "div.mw-category" or "div.mw-category-columns".
- Skip a div if it contains an <a title="mở"> or an element with class "CategoryTreeEmptyBullet".
- From remaining divs, collect <a> hrefs that start with "/wiki/" and convert to full
  URLs using https://vi.wikipedia.org + href.
- Deduplicate and print each link and the total count. Optionally save to an output file.

Usage:
  python scripts/extract_category_links.py --wiki-dir ~/wiki --output links.txt --verbose

"""
from pathlib import Path
import os
import argparse
from bs4 import BeautifulSoup
from typing import List, Set


def extract_links_from_html(html: str) -> List[str]:
    """Extract links from a single HTML string following the JS logic."""
    soup = BeautifulSoup(html, "html.parser")
    result_links: List[str] = []

    cats = soup.select("div.mw-category, div.mw-category-columns")
    for div in cats:
        # hasToggle: a[title="mở"] or .CategoryTreeEmptyBullet
        has_toggle = div.select_one('a[title="mở"]') or div.select_one('.CategoryTreeEmptyBullet')
        if has_toggle:
            continue

        links = div.find_all('a', href=True)
        for a in links:
            raw_path = a.get('href')
            if raw_path and raw_path.startswith('/wiki/'):
                full_link = 'https://vi.wikipedia.org' + raw_path
                full_link = full_link.replace('http://', 'https://')
                result_links.append(full_link)

    return result_links


def find_html_files(root: Path) -> List[Path]:
    return list(root.rglob('*.html')) + list(root.rglob('*.htm'))


def main():
    parser = argparse.ArgumentParser(description='Extract Wikipedia category links from HTML files')
    parser.add_argument('--wiki-dir', default=os.path.expanduser('~/wiki'), help='Directory to scan (default: ~/wiki)')
    parser.add_argument('--output', help='Optional output file path to write links (one per line)')
    parser.add_argument('--verbose', action='store_true', help='Show per-file progress')
    parser.add_argument('--no-dedupe', action='store_true', help='Disable deduplication of links')
    args = parser.parse_args()

    wiki_dir = Path(os.path.expanduser(args.wiki_dir))
    if not wiki_dir.exists():
        print(f"Wiki directory does not exist: {wiki_dir}")
        return

    files = find_html_files(wiki_dir)
    if not files:
        print(f"No HTML files found under {wiki_dir}")
        return

    all_links: List[str] = []
    for f in files:
        try:
            text = f.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            if args.verbose:
                print(f"Failed to read {f}: {e}")
            continue

        links = extract_links_from_html(text)
        if links and args.verbose:
            print(f"{f}: found {len(links)} links")
        all_links.extend(links)

    if not args.no_dedupe:
        # preserve order while deduplicating
        seen: Set[str] = set()
        deduped: List[str] = []
        for l in all_links:
            if l not in seen:
                seen.add(l)
                deduped.append(l)
        all_links = deduped

    for link in all_links:
        print(link)

    print(f"Tổng số link: {len(all_links)}")

    if args.output:
        out_path = Path(args.output).expanduser()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(all_links), encoding='utf-8')
            print(f"Saved {len(all_links)} links to {out_path}")
        except Exception as e:
            print(f"Failed to write output file {out_path}: {e}")


if __name__ == '__main__':
    main()

