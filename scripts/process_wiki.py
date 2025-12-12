#!/usr/bin/env python
"""
Walk ~/wiki/ recursively and for each .html/.htm file:
 - if it contains an HTML <tr> with a <td>, call the existing scripts/clean_html.py on that file
 - otherwise skip

Uses the same Python interpreter (sys.executable) to invoke the existing script so it runs in the same environment.

Usage:
    python scripts/process_wiki.py --wiki-dir ~/wiki --verbose
"""
from pathlib import Path
import re
import subprocess
import sys
import argparse
import os

TR_TD_RE = re.compile(r"<tr[^>]*>.*?<td[^>]*>.*?</td>.*?</tr>", re.IGNORECASE | re.DOTALL)


def file_has_tr_td(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return bool(TR_TD_RE.search(text))


def run_clean_script(clean_script: Path, target_file: Path, verbose: bool = False) -> int:
    cmd = [sys.executable, str(clean_script), str(target_file)]
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, check=False, capture_output=not verbose, text=True)
        if not verbose:
            # print stdout/stderr short summary
            if res.stdout:
                print(f"[stdout] {res.stdout.strip()}")
            if res.stderr:
                print(f"[stderr] {res.stderr.strip()}")
        return res.returncode
    except Exception as e:
        print(f"Error running clean script on {target_file}: {e}")
        return 2


def main():
    parser = argparse.ArgumentParser(description="Process ~/wiki HTML files and run clean_html.py on files that contain <tr><td>.")
    parser.add_argument("--wiki-dir", default=os.path.expanduser("~/wiki"), help="Directory to scan (default: ~/wiki)")
    parser.add_argument("--scripts-dir", default=str(Path(__file__).resolve().parent), help="Location of scripts (default: scripts/ in repo)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed command output")
    args = parser.parse_args()

    wiki_dir = Path(os.path.expanduser(args.wiki_dir))
    if not wiki_dir.exists():
        print(f"Wiki directory does not exist: {wiki_dir}")
        sys.exit(1)

    scripts_dir = Path(args.scripts_dir)
    clean_script = scripts_dir / "clean_html.py"
    if not clean_script.exists():
        print(f"clean_html.py not found in scripts dir: {clean_script}")
        sys.exit(1)

    html_files = list(wiki_dir.rglob("*.html")) + list(wiki_dir.rglob("*.htm"))
    if not html_files:
        print(f"No HTML files found under {wiki_dir}")
        return

    processed = 0
    cleaned = 0
    skipped = 0

    for f in html_files:
        processed += 1
        if file_has_tr_td(f):
            print(f"[CLEAN] {f}")
            rc = run_clean_script(clean_script, f, verbose=args.verbose)
            if rc == 0:
                cleaned += 1
            else:
                print(f"clean_html.py returned non-zero ({rc}) for {f}")
        else:
            skipped += 1
            if args.verbose:
                print(f"[SKIP] {f} (no <tr><td>)")

    print("\nSummary:")
    print(f"  scanned: {processed}")
    print(f"  cleaned: {cleaned}")
    print(f"  skipped: {skipped}")


if __name__ == "__main__":
    main()

