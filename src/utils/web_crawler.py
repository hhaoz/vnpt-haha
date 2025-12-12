# python
"""Web crawler module using Firecrawl API for collecting RAG data.

Rewritten to:
- scrape HTML only,
- remove unwanted HTML parts via BeautifulSoup,
- convert to Markdown with markdownify,
- clean Markdown with regex,
- extract links from cleaned HTML,
- links mode no longer requires a topic filter.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse, urljoin

from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from tqdm import tqdm
from bs4 import BeautifulSoup
import markdownify
import re

from src.config import DATA_CRAWLED_DIR
from src.utils.common import remove_diacritics

load_dotenv()

DELAY_SECONDS = 6


def remove_unwanted_html(html: str) -> str:
    """Remove site-notice, catlinks and other unwanted parts from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # remove divs with class that contains "vector-sitenotice-container"
    for tag in soup.find_all("div", class_=lambda c: c and "vector-sitenotice-container" in c):
        tag.decompose()

    # remove div id="catlinks"
    cat = soup.find("div", id="catlinks")
    if cat:
        cat.decompose()

    # remove jump links
    for tag in soup.find_all("a", class_="mw-jump-link"):
        tag.decompose()

    # remove header container
    for tag in soup.find_all("div", class_=lambda c: c and "vector-header-container" in c):
        tag.decompose()

    # remove siteSub
    site_sub = soup.find("div", id="siteSub")
    if site_sub:
        site_sub.decompose()

    # remove documentation-container and printfooter
    for tag in soup.find_all("div", class_=lambda c: c and ("documentation-container" in c or "printfooter" in c)):
        tag.decompose()

    # remove edit section spans
    for tag in soup.find_all("span", class_=lambda c: c and "mw-editsection" in c):
        tag.decompose()

    return str(soup)


def clean_markdown(md: str) -> str:
    """Basic cleanup for markdown text."""
    md = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', md)
    md = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', md)
    md = re.sub(r'<a\s+[^>]*>(.*?)</a>', r'\1', md, flags=re.DOTALL)
    md = re.sub(r'https?://\S+', '', md)
    md = re.sub(r'<br\s*/?>', '', md)
    md = re.sub(r'[*\^]', '', md)
    return md.strip()


class WebCrawler:
    """Web crawler with multiple crawl modes using HTML -> Markdown pipeline."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY required")
        self.app = FirecrawlApp(api_key=self.api_key)
        self._last_request = 0.0

    def _wait_rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < DELAY_SECONDS:
            time.sleep(DELAY_SECONDS - elapsed)
        self._last_request = time.time()

    def _scrape_page(self, url: str, get_links: bool = False) -> dict | None:
        """Scrape a single page: fetch HTML, clean it, convert to Markdown, extract metadata & links."""
        self._wait_rate_limit()

        try:
            result = self.app.scrape(url, formats=["html"], only_main_content=True)
            if result is None:
                return None

            raw_html = getattr(result, "html", "") or ""
            if not raw_html:
                return None

            cleaned_html = remove_unwanted_html(raw_html)
            md = markdownify.markdownify(cleaned_html)
            cleaned_md = clean_markdown(md)

            # attempt to get metadata
            meta = result.metadata if hasattr(result, "metadata") else None
            title = getattr(meta, "title", "") if meta else ""
            description = getattr(meta, "description", "") if meta else ""
            keywords = getattr(meta, "keywords", "") if meta else ""

            # extract links from cleaned HTML when requested
            links = []
            if get_links:
                soup = BeautifulSoup(cleaned_html, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("#") or href.lower().startswith("javascript:"):
                        continue
                    full = urljoin(url, href)
                    links.append(full)

            return {
                "url": url,
                "title": title or "",
                "content": cleaned_md,
                "summary": "",  # legacy field kept empty
                "description": description or "",
                "keywords": keywords or "",
                "links": links,
            }

        except Exception as e:
            print(f"[Crawler] Error scraping {url}: {e}")
        return None

    def crawl_single(self, url: str) -> list[dict]:
        print(f"[Crawler] Mode: single | URL: {url}")
        result = self._scrape_page(url)
        if result:
            result.pop("links", None)
            return [result]
        return []

    def crawl_links(self, url: str, max_pages: int = 10, topic: str | None = None) -> list[dict]:
        """Mode: links - Crawl URL + follow links on the page (no topic required)."""
        print(f"[Crawler] Mode: links | URL: {url}")
        print(f"[Crawler] Scraping main page...")

        documents = []
        main_result = self._scrape_page(url, get_links=True)
        if not main_result:
            return []

        links = main_result.pop("links", [])
        documents.append(main_result)
        print(f"[Crawler] Found {len(links)} total links")

        if not links or max_pages <= 1:
            return documents

        source_domain = urlparse(url).netloc

        skip_patterns = [
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx',
            '/Special:', '/Đặc_biệt:', '/Tập_tin:', '/File:',
            '/Thể_loại:', '/Category:', '/Help:', '/Wikipedia:',
            'action=edit', 'action=history', 'oldid=', 'diff=',
            '/w/index.php', '/wiki/index.php',
            '#cite', '#ref', '#mw-',
        ]

        urls_to_scrape = []
        for link_url in links:
            if not link_url:
                continue
            link_domain = urlparse(link_url).netloc
            if link_domain and link_domain != source_domain:
                continue
            decoded_url = unquote(link_url)
            if any(p.lower() in decoded_url.lower() for p in skip_patterns):
                continue
            urls_to_scrape.append(link_url)

        print(f"[Crawler] Candidate links: {len(urls_to_scrape)}")

        scrape_count = min(len(urls_to_scrape), max_pages - 1)
        print(f"[Crawler] Scraping {scrape_count} pages...")

        for page_url in tqdm(urls_to_scrape[:scrape_count], desc="Scraping"):
            if page_url == url:
                continue
            result = self._scrape_page(page_url)
            if result:
                result.pop("links", None)
                documents.append(result)

        return documents

    def crawl_search(self, url: str, topic: str | None = None, max_pages: int = 10) -> list[dict]:
        """Mode: search - Use map API to find URLs by topic (topic optional) then scrape."""
        print(f"[Crawler] Mode: search | URL: {url} | Topic: {topic}")
        documents = []

        try:
            # if topic is None, call map without search param
            if topic:
                map_result = self.app.map(url=url, search=topic, limit=max_pages)
            else:
                map_result = self.app.map(url=url, limit=max_pages)

            urls_to_scrape = []
            if hasattr(map_result, "links"):
                for link in map_result.links:
                    if hasattr(link, "url"):
                        urls_to_scrape.append(link.url)
                    elif isinstance(link, str):
                        urls_to_scrape.append(link)
            elif isinstance(map_result, list):
                for link in map_result:
                    if isinstance(link, str):
                        urls_to_scrape.append(link)

        except Exception as e:
            print(f"[Crawler] Error: Map API failed - {e}")
            return []

        print(f"[Crawler] Found {len(urls_to_scrape)} URLs")

        for page_url in tqdm(urls_to_scrape[:max_pages], desc="Scraping"):
            result = self._scrape_page(page_url)
            if result:
                result.pop("links", None)
                documents.append(result)

        return documents

    def crawl_domain(self, url: str, max_pages: int = 10, topic: str | None = None) -> list[dict]:
        print(f"[Crawler] Mode: domain | URL: {url}")
        documents = []

        crawl_options = {
            "limit": max_pages,
            "scrape_options": {
                "formats": ["html"],
                "only_main_content": True,
            }
        }

        if topic:
            crawl_options["include_paths"] = [f"*{topic}*"]

        try:
            result = self.app.crawl(url=url, **crawl_options)

            pages = []
            if hasattr(result, 'data'):
                pages = result.data or []
            elif isinstance(result, dict) and 'data' in result:
                pages = result['data'] or []

            for page in pages:
                if hasattr(page, 'html') and page.html:
                    cleaned_html = remove_unwanted_html(page.html)
                    md = markdownify.markdownify(cleaned_html)
                    cleaned_md = clean_markdown(md)
                    meta = page.metadata if hasattr(page, 'metadata') else None
                    documents.append({
                        "url": getattr(meta, 'sourceURL', '') if meta else "",
                        "title": getattr(meta, 'title', '') if meta else "",
                        "content": cleaned_md,
                        "description": getattr(meta, 'description', '') if meta else "",
                        "keywords": getattr(meta, 'keywords', '') if meta else "",
                    })

        except Exception as e:
            print(f"[Crawler] Error: Crawl API failed - {e}")

        print(f"[Crawler] Crawled {len(documents)} pages")
        return documents


def crawl_website(
    url: str,
    mode: str = "links",
    topic: str | None = None,
    max_pages: int = 10,
    api_key: str | None = None,
) -> dict:
    crawler = WebCrawler(api_key)

    if mode == "single":
        documents = crawler.crawl_single(url)
    elif mode == "links":
        documents = crawler.crawl_links(url, max_pages, topic)
    elif mode == "search":
        documents = crawler.crawl_search(url, topic, max_pages)
    elif mode == "domain":
        documents = crawler.crawl_domain(url, max_pages, topic)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use: single, links, search, domain")

    print(f"[Crawler] Total: {len(documents)} documents")

    return {
        "source": url,
        "domain": urlparse(url).netloc,
        "mode": mode,
        "topic": topic,
        "crawled_at": datetime.now().isoformat(),
        "total_pages": len(documents),
        "documents": documents,
    }


def save_crawled_data(data: dict, output_dir: str | Path | None = None, filename: str | None = None) -> Path:
    if output_dir is None:
        output_path = DATA_CRAWLED_DIR
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not filename:
        domain = data.get("domain", "unknown").replace(".", "_")
        filename = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    path = output_path / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[Crawler] Saved: {path}")
    return path
