"""Generate the explanation PDF for every option of the table: no exception, 2-3 pages, same numbers as the row."""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
import pymupdf  # noqa: E402

import analysis as an  # noqa: E402
import explain  # noqa: E402

d = an.load_data()
s = an.default_settings(d)
alt = an.all_alternatives(d, s)
h = s.horizon
pages, bad, t0 = Counter(), [], time.time()
for _, r in alt.iterrows():
    row = r.to_dict()
    try:
        c = explain.compute(d, s, row["_spec"], row)
        pdf = explain.build_pdf(d, s, row["_spec"], row, c)
        n = pymupdf.open(stream=pdf, filetype="pdf").page_count
        sm = c["summ"]
        same = (sm["median_or_later"], sm["q10"] or h + 1, sm["q90"] or h + 1, sm["p5"], sm["p10"], sm["p_horizon"]) == \
               (row["Année médiane"], row["P10"], row["P90"], row["≤ 5 ans"], row["≤ 10 ans"], row[f"D'ici {h}"])
        pages[n] += 1
        if not same or not 2 <= n <= 3:
            bad.append((row["ID"], n, same))
    except Exception as exc:  # noqa: BLE001
        bad.append((row["ID"], "exception", repr(exc)[:300]))
print(f"{len(alt)} options in {time.time() - t0:.0f}s; page counts: {dict(pages)}; problems: {len(bad)}")
for b in bad[:30]:
    print("  ", b)
