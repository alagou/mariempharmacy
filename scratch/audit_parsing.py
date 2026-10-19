"""One-off audit: re-parse every archived DPM HTML page and compare with the CSV files."""
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research as r  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
snap = ROOT / "documents/waiting_lists/snapshots/2026-10-19"
wl = pd.read_csv(ROOT / "data/waiting_lists_snapshots.csv", dtype=str)
de = pd.read_csv(ROOT / "data/dpm_delegations_snapshots.csv", dtype=str)
ph = pd.read_csv(ROOT / "data/pharmacies_registry_snapshots.csv", dtype=str)
issues = []
n_rows = 0
for f in sorted(snap.glob("listeatt_*.html")):
    code = f.stem.split("_")[1]
    soup = BeautifulSoup(f.read_bytes(), "html.parser")
    head = r._parse_header(soup)
    rows = r._parse_waiting_table(soup)
    n_rows += len(rows)
    csv = wl[wl.del_code == code].sort_values("order", key=lambda s: s.astype(int))
    if len(csv) != len(rows):
        issues.append((code, "count", len(rows), len(csv)))
    for a, (_, b) in zip(rows, csv.iterrows()):
        if (a["rank"], a["nom"], a["date_inscription"]) != (b["rank"], b["nom"], b["date_inscription"]):
            issues.append((code, "row", a, dict(b)))
    d = de[de.del_code == code]
    if len(d) != 1:
        issues.append((code, "delegation row missing", None, None))
        continue
    d = d.iloc[0]
    for k in ("zone_threshold", "population_dpm", "officines_dpm"):
        if str(head.get(k)) != str(d[k]).split(".")[0]:
            issues.append((code, k, head.get(k), d[k]))
    # dates must be chronological (rank order = date order)
    dates = [x["date_inscription"] for x in rows]
    if dates != sorted(dates):
        issues.append((code, "not chronological", dates, None))
    f2 = snap / f"listedespharm_{code}.html"
    if f2.exists():
        offs = r._parse_officines_table(BeautifulSoup(f2.read_bytes(), "html.parser"))
        n_csv = (ph.del_code == code).sum()
        if len(offs) != n_csv:
            issues.append((code, "registry count", len(offs), n_csv))
        if str(len(offs)) != str(d["officines_dpm"]).split(".")[0]:
            issues.append((code, "registry vs header officines", len(offs), d["officines_dpm"]))
print("waiting rows parsed:", n_rows, "csv rows:", len(wl))
print("issues:", len(issues))
for i in issues[:80]:
    print(i)
