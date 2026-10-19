"""Generate sample explanation PDFs and check them against the table rows.

python scratch/explain_samples.py  -> scratch/explain_samples/*.pdf + zriba_page1.png
"""
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import pymupdf  # noqa: E402

import analysis as an  # noqa: E402
import explain  # noqa: E402

OUT = ROOT / "scratch" / "explain_samples"
OUT.mkdir(exist_ok=True)
SAMPLES = {"zriba_stay_A": "stay|ZAGHOUAN|ZRIBA|3", "saouaf_B_best": "B|ZAGHOUAN|Saouaf|0",
           "saouaf_B_cautious": "B|ZAGHOUAN|Saouaf|1", "fouchana_A_new": "A_new|BEN AROUS|FOUCHANA|15",
           "fouchana_A_keep2017": "A_keep|BEN AROUS|FOUCHANA|15"}
extra = sys.argv[1:]

d = an.load_data()
s = an.default_settings(d)
alt = an.all_alternatives(d, s)
h = s.horizon
ok = True
for name, uid in list(SAMPLES.items()) + [(x.replace("|", "_"), x) for x in extra]:
    row = alt[alt.ID == uid].iloc[0].to_dict()
    t = time.time()
    c = explain.compute(d, s, row["_spec"], row)
    pdf = explain.build_pdf(d, s, row["_spec"], row, c)
    dt = time.time() - t
    path = OUT / f"{name}.pdf"
    path.write_bytes(pdf)
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    text = "".join(p.get_text() for p in doc)
    sm = c["summ"]
    fy = lambda y: str(y) if y <= h else f"après {h}"  # noqa: E731
    checks = {
        "median": (sm["median_or_later"], row["Année médiane"]),
        "P10": (sm["q10"] or h + 1, row["P10"]), "P90": (sm["q90"] or h + 1, row["P90"]),
        "<=5": (sm["p5"], row["≤ 5 ans"]), "<=10": (sm["p10"], row["≤ 10 ans"]), "horizon": (sm["p_horizon"], row[f"D'ici {h}"]),
    }
    mism = {k: v for k, v in checks.items() if v[0] != v[1]}
    norm = re.sub(r"\s+", " ", text.replace("\u00a0", " "))
    in_pdf = {
        "median": fy(row["Année médiane"]) in norm,
        "P10-P90": f"{fy(row['P10'])} – {fy(row['P90'])}" in norm,
        "<=5": explain.fpct(row["≤ 5 ans"], 1).replace("\u00a0", " ") in norm,
        "<=10": explain.fpct(row["≤ 10 ans"], 1).replace("\u00a0", " ") in norm,
        "horizon": explain.fpct(row[f"D'ici {h}"], 1).replace("\u00a0", " ") in norm,
    }
    good = not mism and all(in_pdf.values()) and 2 <= doc.page_count <= 3
    ok &= good
    print(f"{name:22s} pages={doc.page_count} {dt:.1f}s median={fy(row['Année médiane'])} P10-P90={fy(row['P10'])}-{fy(row['P90'])} "
          f"central={c['det_year']} model-vs-row mismatches={mism} found-in-pdf={in_pdf} -> {'OK' if good else 'CHECK'}")
    if name == "zriba_stay_A":
        doc[0].get_pixmap(dpi=110).save(str(OUT / "zriba_page1.png"))
scen = an.build_scenarios(d, s, "312", ("ZAGHOUAN", "Saouaf"), None, 1)
for sc in scen:
    c = explain.compute(d, s, sc["spec"])
    pdf = explain.build_pdf(d, s, sc["spec"], None, c)
    n = pymupdf.open(stream=pdf, filetype="pdf").page_count
    same = c["summ"] == sc["summary"]
    ok &= same and 2 <= n <= 3
    print(f"scenario {sc['code']:3s} {sc['title'][:45]:45s} pages={n} summary identical to Synthèse: {same}")
print("ALL OK" if ok else "SOME CHECKS FAILED")
