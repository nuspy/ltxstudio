# Building the two-column commercial PDF

The chapters in `docs/manual/*.md` are the **source of truth**. This folder turns them into the
polished, two-column A4 booklet used for commercial distribution.

`style.css` provides the layout (two columns via `column-count: 2`, A4 page geometry, badges, tip
boxes, running page numbers).

Pick whichever toolchain you have installed. All commands are run from `docs/manual/`.

> **Quickest (no Pandoc needed).** If you have Python with the `markdown` package and any Chromium
> browser, just run:
> ```bash
> python build/make_pdf.py
> ```
> It concatenates the chapters, wraps them in `.manual-body`, applies `style.css`, and prints
> `build/AI_Creative_Center_User_Manual.pdf` via headless Chrome. This is the script used to produce
> the reference build (~58 A4 pages, two columns).

---

## Chapter order

Concatenate the files in numeric order (the cover first, no `README.md`):

```
00-cover-toc.md 01-getting-started.md 02-images.md 03-video.md 04-quality-restore.md
05-audio-voice.md 06-music.md 07-music-video.md 08-3d.md 09-cinema-studio.md 10-training.md
11-import-models.md 12-advanced.md 13-steganography.md 14-wizards.md 15-rest-api.md
16-tips-and-tricks.md 17-appendix.md
```

---

## Option A — Pandoc → HTML → headless Chrome → PDF (recommended, free)

Renders `column-count` correctly because Chrome's print engine honours multicol.

```bash
# 1) Markdown -> single self-contained HTML, wrapping the body in .manual-body for the CSS columns
pandoc $(ls [0-9]*.md | sort) \
  --standalone --embed-resources \
  --metadata title="AI Creative Center — User Manual" \
  --css build/style.css \
  --wrap=none \
  -o build/manual.html

# 2) HTML -> PDF (Chrome / Edge / Brave — any Chromium). Windows example:
"C:/Program Files/Google/Chrome/Application/chrome.exe" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="build/AI_Creative_Center_User_Manual.pdf" \
  "file:///E:/ai/ltxstudio/docs/manual/build/manual.html"
```

> If the two columns do not appear, make sure the manual content is inside an element with
> `class="manual-body"`. The simplest way is to add a tiny wrapper: after step 1, open
> `build/manual.html` and wrap everything between `<body>` and `</body>` (except the cover) in
> `<div class="manual-body"> … </div>`, or use the Pandoc template in `build/` if present.

## Option B — Prince or Paged.js (best print fidelity, page numbers, avoids column breaks)

```bash
# Prince (commercial, best CSS Paged Media support)
prince build/manual.html -o build/AI_Creative_Center_User_Manual.pdf

# or Paged.js CLI (free)
npx pagedjs-cli build/manual.html -o build/AI_Creative_Center_User_Manual.pdf
```

## Option C — Pandoc → LaTeX (two-column via `multicol`)

Use only if you prefer a LaTeX pipeline; the CSS is ignored and columns come from LaTeX:

```bash
pandoc $(ls [0-9]*.md | sort) \
  -V documentclass=article -V classoption=twocolumn \
  -V geometry:a4paper,margin=14mm \
  -V fontsize=10pt \
  --toc -o build/AI_Creative_Center_User_Manual.pdf
```

---

## Notes

- The manual is authored in **GitHub-flavoured Markdown**; any renderer that supports tables and
  fenced code blocks will display it. Only the **two-column** commercial layout needs the build step
  above — for on-screen reading the raw `.md` files (or a normal Markdown viewer) are enough.
- Keep `style.css` next to the generated `manual.html`, or pass an absolute path to `--css`.
- To brand the PDF for a specific reseller, edit the cover block in `00-cover-toc.md` and the
  `@bottom-center` running footer in `style.css`.
