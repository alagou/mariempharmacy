import sys, fitz, pathlib
sys.stdout.reconfigure(encoding="utf-8")
for p in sys.argv[1:]:
    doc = fitz.open(p)
    print("=====", p, "pages:", doc.page_count)
    out = []
    for i, page in enumerate(doc):
        out.append(f"--- page {i+1}\n" + page.get_text())
    txt = "\n".join(out)
    pathlib.Path("scratch/txt").mkdir(exist_ok=True)
    pathlib.Path("scratch/txt/" + pathlib.Path(p).stem + ".txt").write_text(txt, encoding="utf-8")
    print(txt[: int(os.environ.get("N", "6000")) if False else 6000])
