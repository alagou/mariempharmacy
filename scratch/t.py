import pandas as pd, sys, re, unicodedata
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
rr = pd.read_csv("data/renewal_records.csv", dtype=str).fillna("")


def norm(s):
    return re.sub(r"[^A-Z]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper())


def key(r):
    nom = re.split(r"\bEP\b|\bEPOUSE\b|\bEPSE\b", r.nom.upper())[0]
    return norm(nom) + "|" + norm(r.prenom)[:6] + "|" + r.date_inscription


rr["k"] = rr.apply(key, axis=1)
A = rr[rr.category == "A"]
s24 = A[A.list_type.isin(["renouvele_2024", "renouvele_sans_attestation_2024"])].drop_duplicates("k")
s26 = set(A[A.list_type == "renouvele_2026"].k)
n26 = set(A[A.list_type == "non_renouvele_2026"].k)
s24 = s24.assign(status=s24.k.map(lambda k: "kept" if k in s26 else ("nonren" if k in n26 else "exit")))
print(s24.status.value_counts())
# rank in delegation by date (2024 list)
s24 = s24[s24.delegation != ""]
s24["rank"] = s24.groupby(["governorate", "delegation"]).date_inscription.rank(method="first")
s24["rg"] = pd.cut(s24["rank"], [0, 1, 2, 3, 5, 10, 1000], labels=["1", "2", "3", "4-5", "6-10", "11+"])
print(pd.crosstab(s24.rg, s24.status, normalize="index").round(3), pd.crosstab(s24.rg, s24.status, margins=True))
s24["age"] = 2024 - s24.date_inscription.str[:4].astype(int)
s24["ag"] = pd.cut(s24.age, [-1, 2, 5, 10, 15, 40])
print(pd.crosstab(s24.ag, s24.status, normalize="index").round(3))
z = s24[s24.governorate == "ZAGHOUAN"]
print(z.sort_values(["delegation", "rank"])[["delegation", "rank", "nom", "prenom", "date_inscription", "status"]].to_string())
# 2026 renewed with old dates missing in 2024 (residual mismatch)
new = A[(A.list_type == "renouvele_2026") & ~A.k.isin(set(s24.k)) & (A.date_inscription < "2024-01-01")]
print("residual old-date new in 2026:", len(new))
# 2022 non-renewal => 2024
n22 = set(A[A.list_type.str.startswith("non_renouvele_2022")].k)
s20 = A[A.list_type == "renouvele_2020"].drop_duplicates("k")
all24 = set(A[A.list_type.str.contains("2024")].k)
print("2020 renewed:", len(s20), "present in 2024 lists:", s20.k.isin(all24).sum(), "in renewed 2024:", s20.k.isin(set(s24.k)).sum())
