# python
#!/usr/bin/env python
"""
Extract text from all <tr> rows in an HTML file and overwrite the file
with plain text: one row per line (cells joined with a single space).
Usage:
    python scripts/clean_html.py path/to/file.html
"""
import sys
from pathlib import Path
from bs4 import BeautifulSoup

def extract_tr_text_and_overwrite(path: str):
    file = Path(path)
    if not file.exists():
        print("❌ File does not exist")
        return

    raw_html = file.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")

    trs = soup.find_all("tr")
    if not trs:
        print("⚠️ No <tr> elements found. File unchanged.")
        return

    lines = []
    for tr in trs:
        cells = tr.find_all(["td", "th"])
        if cells:
            text = " ".join(cell.get_text(separator=" ", strip=True) for cell in cells)
        else:
            # fallback to any text inside <tr>
            text = tr.get_text(separator=" ", strip=True)
        if text:
            lines.append(text)

    if not lines:
        print("⚠️ No textual content found inside <tr> elements. File unchanged.")
        return

    file.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Extracted {len(lines)} rows and overwrote: {path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("   python scripts/clean_html.py path/to/file.html")
        sys.exit(1)
    extract_tr_text_and_overwrite(sys.argv[1])
