import pandas as pd, sys, re
sys.stdout.reconfigure(encoding="utf-8")
p = pd.read_csv("data/pharmacies.csv", dtype=str).fillna("")
p = p[p.category == "A"]
imada = p.adresse.str.contains(r"IMAD|imad|Imad|عمادة|SECTEUR|secteur", regex=True)
commune = p.adresse.str.contains(r"COMMUNE|commune|Commune", regex=True)
print("A pharmacies", len(p), "imada addresses", imada.sum(), "explicit 'commune'", commune.sum())
print(p[imada].groupby("governorate").size().sort_values(ascending=False).to_string())
d = pd.read_csv("data/delegations.csv")
print("delegations", len(d))
