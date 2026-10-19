"""Snapshot of all_alternatives / build_scenarios outputs, to check that refactors leave the numbers unchanged.

python scratch/explain_baseline.py save   -> scratch/explain_samples/baseline.pkl
python scratch/explain_baseline.py check  -> compares the current outputs with the snapshot
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
import analysis as an  # noqa: E402

OUT = Path(__file__).resolve().parent / "explain_samples" / "baseline.pkl"
d = an.load_data()
s = an.default_settings(d)
alt = an.all_alternatives(d, s)
sc = an.build_scenarios(d, s, "312", ("ZAGHOUAN", "Saouaf"), None, 1)
cur = {"alt": alt, "scen": [(x["code"], x["summary"]) for x in sc]}
if sys.argv[1:] == ["save"]:
    OUT.parent.mkdir(exist_ok=True)
    pickle.dump(cur, OUT.open("wb"))
    print("saved", len(alt), [(c, sm["q50"]) for c, sm in cur["scen"]])
else:
    old = pickle.load(OUT.open("rb"))
    cols = [c for c in old["alt"].columns if c in alt.columns]
    key = ["Unité", "Gouvernorat", "Catégorie"]
    a = old["alt"][cols].sort_values(key).reset_index(drop=True)
    b = alt[cols].sort_values(key).reset_index(drop=True)
    diff = [c for c in cols if not a[c].equals(b[c]) and not (a[c].fillna(-1) == b[c].fillna(-1)).all()]
    print("alt rows", len(a), len(b), "columns differing:", diff)
    print("scenarios equal:", old["scen"] == cur["scen"])
