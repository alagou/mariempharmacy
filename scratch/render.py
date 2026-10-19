import sys, pymupdf, pathlib
out = pathlib.Path("scratch/png"); out.mkdir(exist_ok=True)
zoom = float(sys.argv[2]) if len(sys.argv) > 2 else 1.6
pages = sys.argv[3] if len(sys.argv) > 3 else None
p = sys.argv[1]
doc = pymupdf.open(p)
rng = range(doc.page_count)
if pages:
    a, b = pages.split("-"); rng = range(int(a) - 1, min(int(b), doc.page_count))
for i in rng:
    pix = doc[i].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    f = out / f"{pathlib.Path(p).stem}_p{i+1}.png"
    pix.save(f)
    print(f, doc.page_count)
