import pandas as pd, sys, re, unicodedata, difflib
sys.stdout.reconfigure(encoding="utf-8")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"\[.*?\]|\(.*?\)", " ", s)
    s = re.sub(r"[^A-Z ]", " ", s)
    s = re.sub(r"\b(EL|EZ|ES|ET|ER|ED|EN|DE|LA|LE|DES|COMMUNE|VILLE|DJ|ANCIEN|BHIRA)\b", " ", s)
    s = s.replace("DJ", "J").replace("OU", "O").replace("EE", "E").replace("K", "Q").replace("GH", "G")
    return re.sub(r"\s+", "", s)


d = pd.read_csv("data/dpm_delegations_snapshots.csv")
h = pd.read_excel("documents/statistics/tun_admin_boundaries.xlsx", sheet_name="tun_admin3")
GOV = {"ARIANA": "Ariana", "BEJA": "Béja", "BEN AROUS": "Ben Arous", "BIZERTE": "Bizerte", "GABES": "Gabès", "GAFSA": "Gafsa",
       "JENDOUBA": "Jendouba", "KAIROUAN": "Kairouan", "KASSERINE": "Kasserine", "KEBILI": "Kébili", "LE KEF": "Le Kef", "MAHDIA": "Mahdia",
       "MANOUBA": "Manouba", "MEDENINE": "Médenine", "MONASTIR": "Monastir", "NABEUL": "Nabeul", "SFAX": "Sfax", "SIDI BOUZID": "Sidi Bouzid",
       "SILIANA": "Siliana", "SOUSSE": "Sousse", "TATAOUINE": "Tataouine", "TOZEUR": "Tozeur", "TUNIS": "Tunis", "ZAGHOUAN": "Zaghouan"}
print(sorted(h.adm2_name.unique()))
out = []
for g, grp in d.groupby("governorate"):
    hg = h[h.adm2_name.map(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().upper()) ==
           unicodedata.normalize("NFKD", GOV[g]).encode("ascii", "ignore").decode().upper()]
    cand = {norm(n): (p, n) for p, n in zip(hg.adm3_pcode, hg.adm3_name)}
    for _, r in grp.iterrows():
        k = norm(r.delegation)
        m = difflib.get_close_matches(k, list(cand), n=1, cutoff=0.6)
        out.append((g, r.del_code, r.delegation, cand[m[0]][1] if m else None, cand[m[0]][0] if m else None,
                    difflib.SequenceMatcher(None, k, m[0]).ratio() if m else 0))
o = pd.DataFrame(out, columns=["gov", "del_code", "dpm", "hdx", "pcode", "score"])
print(o[o.score < 0.85].to_string())
print("dups:", o[o.pcode.duplicated(keep=False) & o.pcode.notna()].to_string())
print("unused hdx:", sorted(set(h.adm3_pcode) - set(o.pcode)))
print(h[h.adm3_pcode.isin(set(h.adm3_pcode) - set(o.pcode))][["adm2_name", "adm3_name", "adm3_pcode"]].to_string())
o.to_csv("scratch/xwalk_auto.csv", index=False)
