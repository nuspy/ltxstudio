#!/usr/bin/env python
"""Build the two-column commercial PDF of the AI Creative Center manual.

Concatenates the numbered chapters, renders Markdown -> HTML, wraps the body in
`.manual-body` (so style.css applies its two-column layout), then prints to PDF
with headless Chrome. Pure-stdlib + `markdown`; no network, no npm.

Usage (from docs/manual/):  python build/make_pdf.py
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANUAL = os.path.dirname(HERE)
OUT_HTML = os.path.join(HERE, "manual.html")
OUT_PDF = os.path.join(HERE, "AI_Creative_Center_User_Manual.pdf")

try:
    import markdown  # type: ignore
except ImportError:
    sys.exit("The 'markdown' package is required: pip install markdown")


def find_chrome() -> str | None:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        shutil.which("chrome"), shutil.which("chromium"), shutil.which("msedge"),
    ]
    return next((c for c in candidates if c and os.path.exists(c)), None)


def build_html() -> str:
    chapters = sorted(glob.glob(os.path.join(MANUAL, "[0-9][0-9]-*.md")))
    md_text = "\n\n".join(open(c, encoding="utf-8").read() for c in chapters)

    md = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list", "toc"])
    body_html = md.convert(md_text)

    # Keep the cover out of the multicolumn flow: split on the first <hr/>.
    parts = re.split(r"(<hr\s*/?>)", body_html, maxsplit=1)
    if len(parts) == 3:
        cover, hr, rest = parts
        inner = f'{cover}{hr}<div class="manual-body">{rest}</div>'
    else:
        inner = f'<div class="manual-body">{body_html}</div>'

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>AI Creative Center — User Manual</title>
<link rel="stylesheet" href="style.css">
</head><body>
{inner}
</body></html>"""
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    return OUT_HTML


def to_pdf(html_path: str) -> bool:
    chrome = find_chrome()
    if not chrome:
        print("Chrome/Edge not found — HTML written, open it and Ctrl+P to PDF.")
        return False
    url = "file:///" + html_path.replace("\\", "/")
    # --headless=new + an isolated profile avoids hanging on a running Chrome;
    # --virtual-time-budget forces layout to settle so the process exits cleanly.
    profile = os.path.join(HERE, "_chrome_profile")
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           "--virtual-time-budget=15000", f"--user-data-dir={profile}",
           "--no-first-run", f"--print-to-pdf={OUT_PDF}", url]
    try:
        subprocess.run(cmd, check=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("Chrome timed out. Is the output PDF open in a viewer? Close it and retry.")
        return False
    return os.path.exists(OUT_PDF)


if __name__ == "__main__":
    html_path = build_html()
    print(f"HTML: {html_path}")
    if to_pdf(html_path):
        size = os.path.getsize(OUT_PDF) // 1024
        print(f"PDF : {OUT_PDF} ({size} KB)")
