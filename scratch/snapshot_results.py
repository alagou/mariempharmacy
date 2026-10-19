"""Record the dashboard's default scenario results (used for before/after comparisons).

Usage: python scratch/snapshot_results.py <output.json> [horizon] [target delegation] [target commune]
Without a target, the dashboard defaults are used (best other delegation, first B candidate commune).
"""
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as an  # noqa: E402
import forecasting as fc  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
out = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/results.json")
horizon = int(sys.argv[2]) if len(sys.argv) > 2 else fc.DEFAULT_HORIZON
tgt_name = sys.argv[3] if len(sys.argv) > 3 else None
comm_name = sys.argv[4] if len(sys.argv) > 4 else None
d = an.load_data()
if hasattr(an, "default_settings"):
    s = an.default_settings(d, horizon=horizon)
else:
    att = fc.attrition_estimates(d["renewals"], "A")
    s = an.SimSettings(h_mean=att["h_central"], h_sd=max((att["h_high"] - att["h_low"]) / 3.3, 0.003), horizon=horizon)
cmp = an.compare_delegations(d, 100, s)
others = cmp[cmp.delegation != d["profile"]["delegation"]]
tgt = (others[others.delegation == tgt_name].iloc[0] if tgt_name else
       others.sort_values(["sim_median", "position_used", "distance_km"]).iloc[0]).del_code
comm = an.communes_B(d["communes"], s.night_threshold, an.DEFAULT_B_EXISTING)
bc = comm[(comm.distance_km <= 100) & comm.pop_2024.notna() & ~((comm.governorate == "ZAGHOUAN") & (comm.commune_name == "Zriba"))]
bc = bc[bc.commune_name == comm_name].iloc[0] if comm_name else bc.sort_values(["waiting_B", "distance_km"]).iloc[0]
scen = an.build_scenarios(d, s, tgt, (bc.governorate, bc.commune_name), None, 1)
res = {"settings": {f.name: getattr(s, f.name) for f in fields(s) if f.name != "nat_path"}, "scenarios": [], "compare": []}
for x in scen:
    sm = x["summary"]
    res["scenarios"].append({"code": x["code"], "title": x["title"], "position": x["position"], "q10": sm["q10"], "q50": sm["q50"],
                             "q90": sm["q90"], "p10": sm["p10"], "p_horizon": sm["p_horizon"], "band10": sm["band10"],
                             "risk": x["legal_risk"], "confidence": x["confidence"]})
for _, r in cmp.sort_values("sim_median").head(15).iterrows():
    res["compare"].append({"delegation": r.delegation, "km": float(r.distance_km), "position": int(r.position_used),
                           "median": int(r.sim_median), "p10": float(r.p10)})
res["n_before_horizon"] = int((others.sim_median <= s.horizon).sum())
out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
for x in res["scenarios"]:
    print(f"{x['code']:3s} {x['title'][:50]:50s} pos {x['position']}  P10 {x['q10']}  med {x['q50']}  P90 {x['q90']}  "
          f"≤10a {x['p10']:.1%}  ≤{horizon} {x['p_horizon']:.1%}")
print("other delegations with median before horizon:", res["n_before_horizon"],
      [(c["delegation"], c["median"]) for c in res["compare"][:4]])
