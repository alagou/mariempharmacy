"""Data loading, licensing capacity, nearby comparison and stay-vs-switch scenarios."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import forecasting as fc

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z]", "", s)


def load_data() -> dict:
    regs = json.loads((DATA / "regulations.json").read_text(encoding="utf-8"))
    profile = json.loads((DATA / "profile.json").read_text(encoding="utf-8"))
    dele = pd.read_csv(DATA / "delegations.csv", dtype={"del_code": str, "gov_code": str, "ins_code": str})
    special = regs["rules_current"]["A"]["special_communes"]
    miss = dele.zone_threshold.isna()
    dele.loc[miss, "zone_threshold"] = dele.loc[miss, "delegation"].map(special)
    dele["threshold_note"] = np.where(miss & dele.zone_threshold.notna(), "Commune de l'art. 3 (seuil spécifique)", "")
    pop = pd.read_csv(DATA / "population.csv", dtype={"key": str})
    communes = pd.read_csv(DATA / "communes.csv")
    waiting = pd.read_csv(DATA / "waiting_lists.csv", dtype=str).fillna("")
    waiting["order"] = pd.to_numeric(waiting["order"], errors="coerce")
    pharm = pd.read_csv(DATA / "pharmacies.csv", dtype=str).fillna("")
    renewals = pd.read_csv(DATA / "renewal_records.csv", dtype=str).fillna("")
    history = pd.read_csv(DATA / "pharmacy_history.csv")
    sources = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
    geo_path = DATA / "boundaries.geojson"
    geo = json.loads(geo_path.read_text(encoding="utf-8")) if geo_path.exists() else None
    gov = pop[pop.level == "governorate"].pivot_table(index="key", columns="year", values="population")
    gov_growth = {_norm(k): fc.cagr(r[2014], r[2024], 10) for k, r in gov.iterrows()}
    dele["gov_growth"] = dele.governorate.map(lambda g: gov_growth.get(_norm(g)))
    communes["gov_growth"] = communes.governorate.map(lambda g: gov_growth.get(_norm(g)))
    return {"regs": regs, "profile": profile, "delegations": dele, "population": pop, "communes": communes,
            "waiting": waiting, "pharmacies": pharm, "renewals": renewals, "history": history,
            "sources": sources, "geo": geo, "gov_growth": gov_growth}


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------
def growth_for_delegation(row) -> fc.GrowthRates:
    return fc.growth_rates(row.population_dpm, row.pop_2014, row.pop_2004, row.gov_growth, row.governorate)


def capacity_A(dele: pd.DataFrame, pop_mode: str = "census") -> pd.DataFrame:
    d = dele.copy()
    d["allowed_A"] = [fc.allowed_A(p, n) for p, n in zip(d.population_dpm, d.zone_threshold)]
    d["free_A"] = d.allowed_A - d.pharmacies_A
    d["next_pop_A"] = [fc.next_threshold_pop(int(max(e, a)), n, "A") if pd.notna(n) and pd.notna(e) else np.nan
                       for e, a, n in zip(d.pharmacies_A, d.allowed_A, d.zone_threshold)]
    rows = []
    for _, r in d.iterrows():
        if pd.isna(r.population_dpm) or pd.isna(r.next_pop_A):
            rows.append((np.nan, np.nan, None, None, None, ""))
            continue
        g = growth_for_delegation(r)
        yc = fc.year_reaching(r.population_dpm, g.central, r.next_pop_A, mode=pop_mode)
        yh = fc.year_reaching(r.population_dpm, g.high, r.next_pop_A, mode=pop_mode)
        yl = fc.year_reaching(r.population_dpm, g.low, r.next_pop_A, mode=pop_mode)
        rows.append((g.central, g.sd, yc, yh, yl, g.basis))
    d["growth_central"], d["growth_sd"], d["year_next_A_central"], d["year_next_A_high"], d["year_next_A_low"], d["growth_basis"] = zip(*rows)
    d["newcomer_position_A"] = d.waiting_A + 1
    return d


def communes_B(communes: pd.DataFrame, night_threshold: int = 60000, existing_overrides: dict | None = None) -> pd.DataFrame:
    c = communes.copy()
    c["threshold_B"] = night_threshold
    c["allowed_B"] = [fc.allowed_B(p, night_threshold) if pd.notna(p) else np.nan for p in c.pop_2024]
    c["existing_B"] = np.nan
    c["existing_B_source"] = "Inconnu (non publié par la DPM)"
    for (g, name), (val, src) in (existing_overrides or {}).items():
        m = (c.governorate == g) & (c.commune_name == name)
        c.loc[m, "existing_B"] = val
        c.loc[m, "existing_B_source"] = src
    c["newcomer_position_B"] = c.waiting_B + 1
    return c


# Known information on night pharmacies (unofficial) – used as default assumptions only
DEFAULT_B_EXISTING = {
    ("ZAGHOUAN", "Zriba"): (1, "Pharmacie de nuit à Hammam Zriba signalée par med.tn (non officiel)"),
}


# ---------------------------------------------------------------------------
# Waiting lists
# ---------------------------------------------------------------------------
def current_list(waiting: pd.DataFrame, category: str, governorate: str, unit: str) -> pd.DataFrame:
    if category == "A":
        w = waiting[(waiting.category == "A") & (waiting.governorate == governorate) & (waiting.delegation == unit)]
    else:
        w = waiting[(waiting.category == "B") & (waiting.governorate == governorate) & (waiting.commune == unit)]
    return w.sort_values("order")


def list_history(renewals: pd.DataFrame, waiting: pd.DataFrame, category: str, governorate: str, unit: str) -> pd.DataFrame:
    """Presence of each registrant of a list across the official tables 2020-2026."""
    col = "delegation" if category == "A" else "commune"
    r = renewals[(renewals.category == category) & (renewals.governorate == governorate)].copy()
    unit_n = _norm(unit)
    r = r[r[col].map(_norm) == unit_n]
    r["k"] = [fc._key(a, b, c) for a, b, c in zip(r.nom, r.prenom, r.date_inscription)]
    lists = ["renouvele_2020", "non_renouvele_2022_avril", "non_renouvele_2022_aout", "renouvele_2024",
             "renouvele_sans_attestation_2024", "non_renouvele_2024", "renouvele_2026", "non_renouvele_2026"]
    people = r.sort_values("date_inscription").drop_duplicates("k")[["k", "nom", "prenom", "date_inscription"]]
    for lt in lists:
        ks = set(r[r.list_type == lt].k)
        people[lt] = people.k.map(lambda k: "✔" if k in ks else "")
    if category == "A":
        cur = current_list(waiting, "A", governorate, unit)
        ck = {fc._key(a, b, c): rk for a, b, c, rk in zip(cur.nom, cur.prenom, cur.date_inscription, cur["rank"])}
        people["liste_en_ligne_2025"] = people.k.map(lambda k: ck.get(k, ""))
    return people.drop(columns="k").reset_index(drop=True)


def list_flows(renewals: pd.DataFrame, category: str = "A") -> pd.DataFrame:
    """Per unit: number renewed in 2024, still present 2026, not renewed, other exits, new entrants."""
    col = "delegation" if category == "A" else "commune"
    r = renewals[renewals.category == category].copy()
    r["k"] = [fc._key(a, b, c) for a, b, c in zip(r.nom, r.prenom, r.date_inscription)]
    r["unit"] = r[col]
    b24 = r[r.list_type.isin(["renouvele_2024", "renouvele_sans_attestation_2024"])].drop_duplicates("k")
    k26 = set(r[r.list_type == "renouvele_2026"].k)
    n26 = set(r[r.list_type == "non_renouvele_2026"].k)
    b24 = b24.assign(status=np.where(b24.k.isin(k26), "maintenu", np.where(b24.k.isin(n26), "non_renouvele", "sortie_autre")))
    t = b24.pivot_table(index=["governorate", "unit"], columns="status", values="k", aggfunc="count", fill_value=0)
    r26 = r[r.list_type == "renouvele_2026"]
    new = r26[~r26.k.isin(set(b24.k))].groupby(["governorate", "unit"]).size().rename("nouveaux_2026")
    t = t.join(new, how="outer").fillna(0).astype(int).reset_index()
    for c in ["maintenu", "non_renouvele", "sortie_autre"]:
        if c not in t:
            t[c] = 0
    return t


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
@dataclass
class SimSettings:
    pop_mode: str = "census"
    admin_lag: int = 2
    h_mean: float = 0.04
    h_sd: float = 0.01
    lam_exceptional: float = 0.015     # ≥ 83 imada/commune creations since 2007 over 263 units ≈ 0.017 / unit / year
    growth_scenario: str = "central"   # low / central / high
    horizon: int = fc.DEFAULT_HORIZON
    n_sims: int = 4000
    night_threshold: int = 60000
    start_year: int = date.today().year


def _growth_pick(g: fc.GrowthRates, scen: str) -> float:
    return {"low": g.low, "high": g.high}.get(scen, g.central)


def simulate_unit(position: int, existing: int, p0: float, threshold: float, rule: str, g: fc.GrowthRates,
                  s: SimSettings, first_licence_now: bool = False, seed: int = 42, n_sims: int | None = None):
    qp = fc.QueueParams(position=position, existing=int(existing), p0=float(p0), threshold=float(threshold), rule=rule,
                        growth_mean=_growth_pick(g, s.growth_scenario), growth_sd=g.sd, h_mean=s.h_mean, h_sd=s.h_sd,
                        lam_exceptional=s.lam_exceptional, first_licence_now=first_licence_now, pop_mode=s.pop_mode,
                        admin_lag=s.admin_lag, start_year=s.start_year, horizon=s.horizon,
                        n_sims=n_sims or s.n_sims, seed=seed)
    years = fc.simulate_queue(qp)
    return years, fc.summarize(years, s.start_year, s.horizon)


def seniority_position(waiting: pd.DataFrame, category: str, governorate: str, unit: str, reg_date: str) -> int:
    lst = current_list(waiting, category, governorate, unit)
    return int((lst.date_inscription < reg_date).sum()) + 1


def compare_delegations(data: dict, radius_km: float, s: SimSettings, n_sims: int = 600) -> pd.DataFrame:
    """Category A comparison for a newcomer (new registration date) in every delegation within the radius."""
    cap = capacity_A(data["delegations"], s.pop_mode)
    near = cap[(cap.distance_km <= radius_km) & cap.population_dpm.notna() & cap.zone_threshold.notna()].copy()
    prof = data["profile"]
    out = []
    for i, r in near.iterrows():
        g = growth_for_delegation(r)
        is_home = r.delegation == prof["delegation"]
        pos = prof["current_rank"] if is_home else int(r.waiting_A) + 1
        years, summ = simulate_unit(pos, int(r.pharmacies_A), r.population_dpm, r.zone_threshold, "A", g, s,
                                    seed=1000 + i, n_sims=n_sims)
        out.append({"del_code": r.del_code, "position_used": pos, "sim_median": summ["median_or_later"],
                    "sim_q10": summ["q10"], "p5": summ["p5"], "p10": summ["p10"], "band10": summ["band10"],
                    "period": fc.period_label(summ),
                    "seniority_position_hyp": seniority_position(data["waiting"], "A", r.governorate, r.delegation, prof["date_inscription"])})
    res = near.merge(pd.DataFrame(out), on="del_code", how="left")
    return res.sort_values(["sim_median", "distance_km"])


def build_scenarios(data: dict, s: SimSettings, target_del: str | None, target_commune: tuple[str, str] | None,
                    b_existing_target: int | None, b_existing_zriba: int = 1, include_hypothetical_e: bool = True) -> list[dict]:
    prof = data["profile"]
    regs = data["regs"]
    cap = capacity_A(data["delegations"], s.pop_mode)
    home = cap[(cap.governorate == prof["governorate"]) & (cap.delegation == prof["delegation"])].iloc[0]
    g_home = growth_for_delegation(home)
    scen = []

    # A – stay in Zriba, Category A
    yA, sA = simulate_unit(prof["current_rank"], int(home.pharmacies_A), home.population_dpm, home.zone_threshold, "A", g_home, s, seed=1)
    scen.append(_scenario("A", "Rester à Zriba – catégorie A", True, prof["date_inscription"], prof["current_rank"],
                          int(home.waiting_A), sA, yA, "Faible", "Moyenne",
                          "Inscription vérifiée (rang 5/5). Aucune démarche requise hormis le renouvellement de janvier 2028.",
                          ["Ancienneté du 26/07/2017 conservée", "Aucun risque administratif"],
                          [f"Prochaine licence à Zriba quand la population officielle atteindra {int(home.next_pop_A):,} hab.".replace(",", " "),
                           "4 inscrits la précèdent ; la licence revient au premier de la liste"]))

    # B – other delegation, Category A (new registration date)
    if target_del:
        t = cap[cap.del_code == target_del].iloc[0]
        g_t = growth_for_delegation(t)
        pos = int(t.waiting_A) + 1
        yB, sB = simulate_unit(pos, int(t.pharmacies_A), t.population_dpm, t.zone_threshold, "A", g_t, s, seed=2)
        scen.append(_scenario("B", f"Changer pour {t.delegation.title()} – catégorie A", True, "date du nouveau dépôt", pos,
                              int(t.waiting_A), sB, yB, "Élevé", "Moyenne",
                              "Changement de liste = nouvelle demande, classée à la date du nouveau dépôt (arrêté 19/12/2019 art. 5-6).",
                              [f"Distance de Zriba : {t.distance_km:.0f} km", f"Capacité légale actuelle : {int(t.allowed_A)} pour {int(t.pharmacies_A)} officines"],
                              ["Perte de la date du 26/07/2017", f"Placement en fin de liste ({pos}e)", "Irréversible"],
                              unit=t.delegation))

    # C – Zriba, Category B
    cz = data["communes"]
    zr = cz[(cz.governorate == prof["governorate"]) & (cz.commune_name.str.upper() == prof["delegation"])].iloc[0]
    g_cz = fc.growth_rates(zr.pop_2024, zr.pop_2014, zr.pop_2004, zr.gov_growth, zr.governorate)
    posC = int(zr.waiting_B) + 1
    yC, sC = simulate_unit(posC, b_existing_zriba, zr.pop_2024, s.night_threshold, "B", g_cz, s,
                           first_licence_now=(b_existing_zriba == 0), seed=3)
    scen.append(_scenario("C", "Passer en catégorie B (nuit) à Zriba", True, "date du nouveau dépôt", posC, int(zr.waiting_B),
                          sC, yC, "Élevé", "Faible",
                          "Pas de conversion A→B : nouvelle inscription sur la liste B de la commune (décret art. 14 ; arrêté art. 5).",
                          ["Officine de nuit (activité exclusivement nocturne)"],
                          ["Perte de la date du 26/07/2017", f"Officine B existante supposée : {b_existing_zriba}",
                           f"Seconde officine B seulement au-delà de {s.night_threshold:,} hab.".replace(",", " "),
                           "Liste B reconstituée à partir du tableau des renouvellements 2026"], unit="Zriba"))

    # D – other commune, Category B
    if target_commune:
        gname, cname = target_commune
        tc = cz[(cz.governorate == gname) & (cz.commune_name == cname)].iloc[0]
        g_tc = fc.growth_rates(tc.pop_2024, tc.pop_2014, tc.pop_2004, tc.gov_growth, tc.governorate)
        posD = int(tc.waiting_B) + 1
        b_att = fc.attrition_estimates(data["renewals"], "B")
        cutoff = f"{s.start_year - 4}-12-31"
        stalled = cz[(cz.governorate == gname) & (cz.waiting_B > 0) & (cz.first_B_date.fillna("9999") <= cutoff)
                     & (cz.pop_2024 < s.night_threshold)]
        evidence = [f"Listes B : seuls {b_att['rank1_two_year_exit']:.0%} des premiers inscrits ont quitté leur liste entre 2024 et 2026"]
        if len(stalled):
            names = ", ".join(f"{r.commune_name} (depuis {r.first_B_date[:4]})" for _, r in stalled.iterrows())
            evidence.append(f"Dans ce gouvernorat, le premier inscrit B attend depuis plus de 4 ans à : {names} – "
                            "une création B n'est donc pas automatique (officine existante ou procédure lente)")
        for existing, suffix in ([(b_existing_target, "")] if b_existing_target is not None else [(0, " (si aucune officine B)"), (1, " (si une officine B existe)")]):
            if pd.isna(tc.pop_2024):
                continue
            yD, sD = simulate_unit(posD, existing, tc.pop_2024, s.night_threshold, "B", g_tc, s,
                                   first_licence_now=(existing == 0), seed=4 + existing)
            scen.append(_scenario("D" + ("" if not suffix else ("0" if existing == 0 else "1")),
                                  f"Changer pour {cname} – catégorie B{suffix}", True, "date du nouveau dépôt", posD,
                                  int(tc.waiting_B), sD, yD, "Élevé", "Faible",
                                  "Nouvelle inscription sur la liste B de la commune ; art. 6 : une commune sans officine B peut en obtenir une hors numerus clausus.",
                                  [f"Distance de Zriba : {tc.distance_km:.0f} km" if pd.notna(tc.distance_km) else "Distance inconnue",
                                   "Commune sans inscrit B" if tc.waiting_B == 0 else f"{int(tc.waiting_B)} inscrit(s) B"],
                                  ["Perte de la date du 26/07/2017", "Existence d'une officine B non vérifiée",
                                   "Viabilité économique d'une officine de nuit non évaluée", "Irréversible"] + evidence, unit=cname))

    # E – transfer keeping seniority (no legal basis identified)
    if include_hypothetical_e and target_del:
        t = cap[cap.del_code == target_del].iloc[0]
        posE = seniority_position(data["waiting"], "A", t.governorate, t.delegation, prof["date_inscription"])
        g_t = growth_for_delegation(t)
        yE, sE = simulate_unit(posE, int(t.pharmacies_A), t.population_dpm, t.zone_threshold, "A", g_t, s, seed=5)
        scen.append(_scenario("E", f"Transfert vers {t.delegation.title()} en gardant 2017 (hypothèse)", False, prof["date_inscription"],
                              posE, int(t.waiting_A), sE, yE, "Non applicable", "Faible",
                              "Aucune disposition ne permet de transférer l'ancienneté (seulement art. 14-15, transitoires/administratifs). Calcul purement illustratif.",
                              ["Montre l'effet de la date de 2017 sur une autre liste"],
                              ["Non autorisé en l'état des textes – nécessite une confirmation officielle écrite"], unit=t.delegation))

    # comparisons with the stay scenario
    for sc in scen:
        sc["years_vs_stay"] = _years_diff(sA, sc["summary"])
        sc["years"] = sc.pop("_years")
        sc["p_earlier_than_stay"] = float(np.mean(sc["years"] < yA)) if sc["code"] != "A" else None
    _rank(scen)
    return scen


def _scenario(code, title, legal, reg_date, position, list_size, summ, years, risk, confidence, legal_note, pros, cons, unit=None):
    return {"code": code, "title": title, "legally_feasible": legal, "registration_date": reg_date, "position": position,
            "list_size": list_size, "summary": summ, "_years": years, "legal_risk": risk, "confidence": confidence,
            "legal_note": legal_note, "pros": pros, "cons": cons, "unit": unit, "period": fc.period_label(summ)}


def _years_diff(base: dict, other: dict) -> str:
    a, b = base["median_or_later"], other["median_or_later"]
    h = base["horizon"]
    if a > h and b > h:
        return "indéterminé (les deux après l'horizon)"
    if b > h:
        return f"perte ≥ {h + 1 - a} an(s)"
    if a > h:
        return f"gain ≥ {a - b} an(s)"
    d = a - b
    return f"gain de {d} an(s)" if d > 0 else ("équivalent" if d == 0 else f"perte de {-d} an(s)")


RISK_W = {"Faible": 1.0, "Élevé": 0.5, "Non applicable": 0.0}
CONF_W = {"Moyenne": 1.0, "Faible": 0.7}


def _rank(scen: list[dict]) -> None:
    feasible = [s for s in scen if s["legally_feasible"]]
    by_time = sorted(feasible, key=lambda s: (s["summary"]["median_or_later"], -(s["summary"]["p10"])))
    for i, s in enumerate(by_time, 1):
        s["rank_time"] = i
    by_risk = sorted(feasible, key=lambda s: (-RISK_W.get(s["legal_risk"], 0), s["summary"]["median_or_later"]))
    for i, s in enumerate(by_risk, 1):
        s["rank_risk"] = i
    for s in feasible:
        s["overall_score"] = s["summary"]["p10"] * RISK_W.get(s["legal_risk"], 0) * CONF_W.get(s["confidence"], 0.7)
    by_all = sorted(feasible, key=lambda s: (-s["overall_score"], s["summary"]["median_or_later"]))
    for i, s in enumerate(by_all, 1):
        s["rank_overall"] = i
    for s in scen:
        if not s["legally_feasible"]:
            s["rank_time"] = s["rank_risk"] = s["rank_overall"] = None
            s["overall_score"] = None


def scenario_table(scen: list[dict]) -> pd.DataFrame:
    rows = []
    for s in scen:
        sm = s["summary"]
        rows.append({
            "Scénario": f"{s['code']} – {s['title']}",
            "Juridiquement possible": "Oui" if s["legally_feasible"] else "Non (hypothèse)",
            "Date d'inscription retenue": s["registration_date"],
            "Rang estimé": f"{s['position']} / {s['list_size'] + (0 if s['code'] in ('A', 'E') else 1)}",
            "Année plausible au plus tôt (10 %)": str(sm["q10"]) if sm["q10"] else f"> {sm['horizon']}",
            "Période la plus probable": s["period"],
            "≤ 3 ans": sm["band3"], "≤ 5 ans": sm["band5"], "≤ 10 ans": sm["band10"],
            "Écart vs rester": s["years_vs_stay"],
            "Risque juridique": s["legal_risk"], "Confiance": s["confidence"],
            "Rang (délai)": str(s["rank_time"] or "–"), "Rang (risque)": str(s["rank_risk"] or "–"),
            "Rang (global)": str(s["rank_overall"] or "–"),
        })
    return pd.DataFrame(rows)


def recommendation(scen: list[dict]) -> str:
    """Prudent narrative; never recommends withdrawing."""
    a = next(s for s in scen if s["code"] == "A")
    faster = [s for s in scen if s["legally_feasible"] and s["code"] != "A"
              and s["summary"]["median_or_later"] + 5 <= a["summary"]["median_or_later"]]
    txt = [f"Rester à Zriba (catégorie A) : période la plus probable {a['period']} ; probabilité d'autorisation sous 10 ans : {a['summary']['band10']}."]
    if faster:
        best = min(faster, key=lambda s: s["summary"]["median_or_later"])
        txt.append(f"Le scénario « {best['title']} » paraît nettement plus rapide (période {best['period']}), "
                   "mais il implique la perte définitive de l'inscription du 26/07/2017 et repose sur des informations à confirmer.")
        txt.append("Avant toute démarche : obtenir une réponse écrite de la DPM sur la procédure, la position exacte sur la liste visée "
                   "et (pour la catégorie B) l'existence d'une officine de nuit dans la commune. Ne pas retirer l'inscription actuelle avant cette réponse.")
    else:
        txt.append("Aucun changement n'offre un gain suffisamment net et fiable pour justifier la perte de l'ancienneté de 2017.")
    txt.append("Cet outil ne dépose ni ne retire aucune demande ; les décisions restent personnelles et doivent être confirmées auprès de l'administration.")
    return " ".join(txt)
