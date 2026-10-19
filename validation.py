"""Historical verification and calibration of the forecasting model (offline).

Every check re-runs part of the model *as of a past date* with only the information that
was available then and compares it with what actually happened:

* population_backtest – predict the 2014→2024 growth of every unit from 2004-2014 data;
  chooses the projection method and the growth uncertainty used by the simulator.
* attrition_calibration – leave rates of people waiting behind the head of a list, trained
  on 2020→2024 and tested on 2024→2026, split by engagement (renewed in 2020 and 2024).
* nowcast_check / exceptional_rate – do the current DPM convocations ('*') fall where the
  numerus clausus formula says a slot is free? Gives the rate of out-of-formula creations.
* observed_waits – how long people actually waited before being convoked or licensed.
* queue_backtest – the whole queue model run as of 2020 on every 2020 list, compared with
  the licences observed in the 2026 registry (old settings vs calibrated settings).

`python validation.py` prints a summary; the dashboard calls `run_all` (cached).
"""
from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import forecasting as fc

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CENSUS_NAT = "Somme des 24 gouvernorats (INS, RGPH 2004/2014/2024)"


def _n(s) -> str:
    return re.sub(r"[^A-Z]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper())


def _tokens(s) -> set[str]:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return {t for t in re.split(r"[^A-Z]+", s) if len(t) > 1 and t not in {"EP", "EPOUSE", "BEN", "BENT", "EL"}}


def _jeffreys(x: float, n: int, q: float) -> float:
    return fc._beta_ppf(q, x + 0.5, n - x + 0.5)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
def national_census(pop: pd.DataFrame) -> dict[int, float]:
    g = pop[pop.level == "governorate"]
    return g.groupby("year").population.sum().to_dict()


def national_projection_path(proj: pd.DataFrame | None = None, base_year: int = fc.BASE_YEAR,
                             census_base: float | None = None, last_year: int = fc.DEFAULT_HORIZON + 10) -> dict[int, float]:
    """Annual national growth g[y] = N(y)/N(y-1) - 1 from the INS 2026 Flash projection (base RGPH 2024).

    2025-2030 and 2054 are official values, 2031-2053 the documented interpolation of research.py;
    after 2054 the last published rate (0.03 %) is kept.
    """
    proj = proj if proj is not None else pd.read_csv(DATA / "population_projections.csv")
    f = proj[(proj.base == "RGPH 2024") & (proj.level == "national")].set_index("year").population.sort_index()
    if census_base:
        f.loc[base_year] = census_base
    f = f.sort_index()
    g = (f / f.shift(1) - 1).dropna().to_dict()
    last = g[max(g)]
    for y in range(max(g) + 1, last_year + 1):
        g[y] = last
    return g


def population_backtest(dele: pd.DataFrame, pop: pd.DataFrame, proj: pd.DataFrame | None = None) -> dict:
    """Predict each unit's 2014→2024 annual growth with information available in 2014."""
    proj = proj if proj is not None else pd.read_csv(DATA / "population_projections.csv")
    nat = national_census(pop)
    g_nat_0414 = fc.cagr(nat[2004], nat[2014], 10)
    g_nat_1424 = fc.cagr(nat[2014], nat[2024], 10)
    p14 = proj[(proj.base == "RGPH 2014") & (proj.level == "national")].set_index("year").population
    g_ins14 = fc.cagr(p14[2014], p14[2024], 10)
    gov = pop[pop.level == "governorate"].pivot_table(index="key", columns="year", values="population")
    gov_g = {_n(k): fc.cagr(r[2004], r[2014], 10) for k, r in gov.iterrows()}
    d = dele[dele.pop_2004.notna() & dele.pop_2014.notna() & dele.population_dpm.notna()].copy()
    d["r1"] = [fc.cagr(a, b, 10) for a, b in zip(d.pop_2004, d.pop_2014)]
    d["actual"] = [fc.cagr(a, b, 10) for a, b in zip(d.pop_2014, d.population_dpm)]
    d["gov_r1"] = d.governorate.map(lambda g: gov_g.get(_n(g)))
    d = d[d.r1.between(-0.05, 0.08) & d.actual.between(-0.05, 0.08)]
    diff = d.r1 - g_nat_0414
    methods = {
        "Tendance de l'unité 2004-2014": d.r1,
        "Tendance du gouvernorat 2004-2014": d.gov_r1,
        "Tendance nationale 2004-2014": pd.Series(g_nat_0414, index=d.index),
        "Projection INS 2014 (nationale) seule": pd.Series(g_ins14, index=d.index),
    }
    ks = [0.0, 0.25, 0.5, 0.75, 1.0]
    oracle = {k: float(np.sqrt(((g_nat_1424 + k * diff - d.actual) ** 2).mean())) for k in ks}
    k_star = min(oracle, key=oracle.get)
    methods[f"Projection INS 2014 + {k_star:.0%} de l'écart local"] = g_ins14 + k_star * diff
    methods[f"Croissance nationale réelle + {k_star:.0%} de l'écart local (référence)"] = g_nat_1424 + k_star * diff
    rows = []
    for name, pred in methods.items():
        e = (pred - d.actual).dropna()
        rows.append({"Méthode": name, "Unités": len(e), "Biais (pt/an)": e.mean() * 100,
                     "Écart-type (pt/an)": e.std() * 100, "Erreur absolue moy. (pt/an)": e.abs().mean() * 100,
                     "RMSE (pt/an)": float(np.sqrt((e ** 2).mean())) * 100,
                     "Erreur pop. 2024 médiane (%)": float(((1 + pred[e.index]) ** 10 / (1 + d.actual[e.index]) ** 10 - 1).abs().median() * 100)})
    table = pd.DataFrame(rows)
    local_err = (g_nat_1424 + k_star * diff - d.actual)
    # uncertainty of a forward projection = local error + national projection error (INS 2014 missed by g_ins14 - actual)
    growth_sd = float(np.sqrt(local_err.std() ** 2 + (g_ins14 - g_nat_1424) ** 2))
    z = d[d.del_code.astype(str) == "502"]
    return {
        "table": table, "k_local": k_star, "oracle_rmse": oracle, "growth_sd": growth_sd,
        "local_sd": float(local_err.std()), "national": {"g_0414": g_nat_0414, "g_1424": g_nat_1424, "g_ins14_proj": g_ins14,
                                                         "census": nat, "ins14_2024": float(p14[2024])},
        "n_units": len(d),
        "zriba": {"r_0414": float(z.r1.iloc[0]), "r_1424": float(z.actual.iloc[0])} if len(z) else None,
        "source": f"{CENSUS_NAT} ; unités DPM comparables (délégations.csv) ; projection INS base RGPH 2014",
    }


def unit_growth_path(r_1424: float | None, g_nat_1424: float, k_local: float, nat_path: dict[int, float],
                     base_year: int = fc.BASE_YEAR, last_year: int = fc.DEFAULT_HORIZON + 10) -> tuple[float, ...]:
    """growth_path for QueueParams: official national growth + k × (unit − national) observed 2014-2024."""
    diff = (r_1424 - g_nat_1424) if r_1424 is not None and not pd.isna(r_1424) else 0.0
    return tuple(nat_path.get(y, nat_path[max(nat_path)]) + k_local * diff for y in range(base_year + 1, last_year + 1))


# ---------------------------------------------------------------------------
# Waiting lists
# ---------------------------------------------------------------------------
def _renewal_sets(renewals: pd.DataFrame, category: str = "A") -> tuple[pd.DataFrame, dict[str, set]]:
    r = renewals[renewals.category == category].copy()
    r["k"] = [fc._key(a, b, c) for a, b, c in zip(r.nom, r.prenom, r.date_inscription)]
    sets = {
        "R20": set(r[r.list_type == "renouvele_2020"].k),
        "N22": set(r[r.list_type.str.startswith("non_renouvele_2022")].k),
        "R24": set(r[r.list_type.isin(["renouvele_2024", "renouvele_sans_attestation_2024"])].k),
        "N24": set(r[r.list_type == "non_renouvele_2024"].k),
        "R26": set(r[r.list_type == "renouvele_2026"].k),
        "N26": set(r[r.list_type == "non_renouvele_2026"].k),
    }
    return r, sets


def _unit_codes(dele: pd.DataFrame) -> dict[tuple[str, str], str]:
    return {(_n(g), _n(u)): str(c) for g, u, c in zip(dele.governorate, dele.delegation, dele.del_code)}


def registry_matches(people: pd.DataFrame, pharm: pd.DataFrame) -> pd.Series:
    """del_code of the category-A pharmacy whose holder name contains the person's nom + prénom tokens ('' if none)."""
    ph = pharm[pharm.category == "A"]
    by_gov: dict[str, list[tuple[set, str]]] = {}
    for g, nom, code in zip(ph.governorate, ph.nom, ph.del_code):
        by_gov.setdefault(_n(g), []).append((_tokens(nom), str(code)))
    out = []
    for g, nom, pre in zip(people.governorate, people.nom, people.prenom):
        need = _tokens(re.split(r"\bEP\b|\bEPOUSE\b", str(nom).upper())[0]) | _tokens(pre)
        hit = [c for toks, c in by_gov.get(_n(g), []) if need and need <= toks]
        out.append(hit[0] if len(hit) == 1 else ("?" if hit else ""))
    return pd.Series(out, index=people.index)


def _base_table(r: pd.DataFrame, keys: set, dele: pd.DataFrame, list_types: list[str]) -> pd.DataFrame:
    b = r[r.list_type.isin(list_types) & r.k.isin(keys)].drop_duplicates("k").copy()
    codes = _unit_codes(dele)
    b["del_code"] = [codes.get((_n(g), _n(u)), "") for g, u in zip(b.governorate, b.delegation)]
    b = b[b.del_code != ""]
    b["rank"] = b.groupby("del_code").date_inscription.rank(method="first").astype(int)
    return b


def attrition_calibration(renewals: pd.DataFrame, dele: pd.DataFrame, pharm: pd.DataFrame) -> dict:
    """Annual leave rate of people at rank ≥ 3 (not counting licences obtained in their own unit)."""
    r, S = _renewal_sets(renewals)
    regul22 = len(S["N22"] & S["R24"]) / max(len(S["N22"]), 1)
    regul24 = len(S["N24"] & S["R26"]) / max(len(S["N24"]), 1)

    def leave_stats(base, stay_keys, non_keys, regul, years):
        b = base[base["rank"] >= 3].copy()
        b["auth"] = registry_matches(b, pharm) == b.del_code
        b = b[~b.auth]
        stay = b.k.isin(stay_keys)
        non = b.k.isin(non_keys) & ~stay
        exits = float((~stay & ~non).sum() + non.sum() * (1 - regul))
        n = len(b)
        p = exits / max(n, 1)
        return {"n": n, "exits": exits, "p_period": p, "h": 1 - (1 - p) ** (1 / years),
                "h_low": 1 - (1 - _jeffreys(exits, n, 0.05)) ** (1 / years),
                "h_high": 1 - (1 - _jeffreys(exits, n, 0.95)) ** (1 / years)}

    b20 = _base_table(r, S["R20"], dele, ["renouvele_2020"])
    train = leave_stats(b20, S["R24"], S["N24"], regul22, 4)
    b24 = _base_table(r, S["R24"], dele, ["renouvele_2024", "renouvele_sans_attestation_2024"])
    seg = {}
    for name, mask in (("Tous", np.ones(len(b24), bool)), ("Engagés (renouvelés 2020 et 2024)", b24.k.isin(S["R20"]).values),
                       ("Inscrits après 2020 ou absents en 2020", ~b24.k.isin(S["R20"]).values)):
        st = leave_stats(b24[mask], S["R26"], S["N26"], regul24, 2)
        pred = st["n"] * (1 - (1 - train["h"]) ** 2)
        seg[name] = {**st, "predicted_exits_from_train": pred}
    old = fc.attrition_estimates(renewals, "A")
    return {"train": train, "test": seg, "regul_2022": regul22, "regul_2024": regul24,
            "h_all": seg["Tous"]["h"], "h_engaged": seg["Engagés (renouvelés 2020 et 2024)"]["h"],
            "h_engaged_band": (seg["Engagés (renouvelés 2020 et 2024)"]["h_low"], seg["Engagés (renouvelés 2020 et 2024)"]["h_high"]),
            "h_old": old["h_central"],
            "source": "Tableaux DPM des renouvellements 2020, 2022 (non renouvelés), 2024, 2026 ; registre des officines 2026"}


def nowcast_check(dele: pd.DataFrame, waiting: pd.DataFrame) -> dict:
    """Current convocations ('*') vs free slots given by the formula with the RGPH 2024 population."""
    d = dele[dele.population_dpm.notna() & dele.zone_threshold.notna()].copy()
    d["allowed"] = [fc.allowed_A(p, n) for p, n in zip(d.population_dpm, d.zone_threshold)]
    d["free"] = d.allowed - d.pharmacies_A
    w = waiting[waiting.category == "A"]
    stars = w[w["rank"].astype(str).str.strip() == "*"].groupby("del_code").size()
    d["stars"] = d.del_code.astype(str).map(stars).fillna(0).astype(int)
    free, full = d[d.free > 0], d[d.free <= 0]
    return {
        "units": len(d), "free_units": len(free), "full_units": len(full),
        "stars_total": int(d.stars.sum()), "stars_in_free": int(free.stars.sum()), "stars_in_full": int(full.stars.sum()),
        "free_units_with_star": int((free.stars > 0).sum()), "full_units_with_star": int((full.stars > 0).sum()),
        "allowed_total": int(d.allowed.sum()), "existing_total": int(d.pharmacies_A.sum()), "free_slots": int(d.free.clip(lower=0).sum()),
        "waiting_total": int(len(w)),
        "table": d[d.free > 0][["del_code", "governorate", "delegation", "population_dpm", "zone_threshold", "allowed", "pharmacies_A", "free", "stars"]],
    }


def exceptional_rate(now: dict) -> dict:
    """Out-of-formula creations per saturated unit and year (a convocation stays visible ≈ 1 year)."""
    x, n = now["full_units_with_star"], now["full_units"]
    return {"lam": (x + 0.5) / n, "lam_high90": _jeffreys(x, n, 0.90), "x": x, "n": n, "lam_old": 0.015,
            "note": "Estimation de Jeffreys : (convocations hors formule + 0,5) / unités saturées ; "
                    "l'ancienne valeur 0,015 venait des créations par imada/commune 2007-2019 (ancien régime)."}


def observed_waits(waiting: pd.DataFrame, renewals: pd.DataFrame, dele: pd.DataFrame, pharm: pd.DataFrame,
                   snapshot_year: int = 2026) -> dict:
    w = waiting[waiting.category == "A"].copy()
    w["year"] = pd.to_datetime(w.date_inscription, errors="coerce").dt.year + \
        pd.to_datetime(w.date_inscription, errors="coerce").dt.dayofyear / 365.25
    conv = w[w["rank"].astype(str).str.strip() == "*"].year.dropna()
    head = w[w.order == 1].year.dropna()
    r, S = _renewal_sets(renewals)
    b20 = _base_table(r, S["R20"], dele, ["renouvele_2020"])
    b20["match"] = registry_matches(b20, pharm)
    lic = b20[b20.match == b20.del_code]
    ly = pd.to_datetime(lic.date_inscription, errors="coerce").dt.year
    nc = nowcast_check(dele, waiting)
    full_codes = set(dele.del_code.astype(str)) - set(nc["table"].del_code.astype(str))
    deep = b20[b20.del_code.isin(full_codes) & (b20["rank"] >= 4)]
    deep_auth = int((deep.match == deep.del_code).sum())

    def q(s):
        return {"n": int(len(s)), "p10": float(s.quantile(0.1)), "median": float(s.median()), "p90": float(s.quantile(0.9))} if len(s) else None
    return {
        "convoked_reg_year": q(conv), "convoked_wait": q(snapshot_year - conv),
        "head_reg_year": q(head), "head_wait": q(snapshot_year - head),
        "licensed_2020_2026": {"n": int(len(lic)), "reg_year": q(ly.dropna()),
                               "rank_2020": q(lic["rank"].astype(float))},
        "saturated_rank4plus": {"n": int(len(deep)), "licensed": deep_auth},
        "source": "Liste d'attente DPM arrêtée au 31/12/2025 ; tableau des renouvellements 2020 ; registre des officines 2026",
    }


def b_first_licence(communes: pd.DataFrame, snapshot: float = 2026.8, max_pop: int = 20000) -> dict:
    """Small communes with a category-B list: how long has the head of the list been waiting?

    Used for 'first B licence of the commune' scenarios: an exponential delay whose median equals the
    current median wait (for a steady flow, the elapsed wait of the current heads has the same median
    as the full wait under an exponential law)."""
    s = communes[(communes.waiting_B > 0) & (communes.pop_2024 < max_pop)]
    dt = pd.to_datetime(s.first_B_date, errors="coerce")
    w = (snapshot - (dt.dt.year + dt.dt.dayofyear / 365.25)).dropna()
    med = float(w.median())
    return {"n": int(len(w)), "median_wait": med, "p90_wait": float(w.quantile(0.9)), "hazard": 1 - 0.5 ** (1 / med),
            "source": "Listes B reconstituées (tableau des renouvellements 2026), communes < 20 000 hab."}


# ---------------------------------------------------------------------------
# Queue backtest (as of 2020)
# ---------------------------------------------------------------------------
LIST_REVISIONS_2020_2025 = [(2021, 2020), (2022, 2021), (2023, 2022), (2024, 2023), (2026, 2024)]


def queue_backtest(dele: pd.DataFrame, pop: pd.DataFrame, renewals: pd.DataFrame, pharm: pd.DataFrame,
                   settings: dict, n_sims: int = 300, seed: int = 7) -> dict:
    """Run the queue model on every 2020 category-A list and compare with licences seen in the 2026 registry.

    Information used for the forecast: 2020 list order, 2004/2014 censuses (population projected with the
    unit's 2004-2014 trend), and the number of pharmacies in 2020. The latter is not published, so it is
    reconstructed as (pharmacies 2026 − licences observed for 2020-list members): the test is therefore
    conditional on the observed number of licences per unit for the capacity part and mainly measures
    *who* gets them (order, attrition, timing) and how many in total.
    """
    r, S = _renewal_sets(renewals)
    b20 = _base_table(r, S["R20"], dele, ["renouvele_2020"])
    b20["match"] = registry_matches(b20, pharm)
    b20["licensed"] = b20.match == b20.del_code
    gov = pop[pop.level == "governorate"].pivot_table(index="key", columns="year", values="population")
    gov_g = {_n(k): fc.cagr(r_[2004], r_[2014], 10) for k, r_ in gov.iterrows()}
    rng_seed = seed
    rows, unit_rows = [], []
    for code, grp in b20.groupby("del_code"):
        u = dele[dele.del_code.astype(str) == code]
        if u.empty:
            continue
        u = u.iloc[0]
        if pd.isna(u.pop_2014) or pd.isna(u.zone_threshold) or pd.isna(u.pharmacies_A):
            continue
        g = fc.cagr(u.pop_2004, u.pop_2014, 10) if pd.notna(u.pop_2004) else None
        g = g if g is not None and -0.03 < g < 0.06 else gov_g.get(_n(u.governorate), 0.01)
        grp = grp.sort_values("rank")
        n_auth = int(grp.licensed.sum())
        e2020 = int(u.pharmacies_A) - n_auth
        if settings["mode"] == "census":
            pop_by = {2015: float(u.pop_2014), 2025: float(u.pop_2014) * (1 + g) ** 11}
            events = [(2026, 2025)]
        else:
            pop_by = {y: float(u.pop_2014) * (1 + g) ** (y - 2014.5) for y in range(2020, 2025)}
            events = LIST_REVISIONS_2020_2025
        h_pos = None
        res = fc.simulate_positions(len(grp), e2020, float(u.zone_threshold), pop_by, range(2021, 2027), events,
                                    settings["h"], settings["h_sd"], settings["lam"], settings["pending_delay"],
                                    settings["award_delay"], settings.get("growth_sd", 0.0), n_sims=n_sims,
                                    seed=rng_seed, h_by_pos=h_pos, max_per_year=settings.get("max_per_year", np.inf))
        rng_seed += 1
        auth = np.isfinite(res) & (res > 0)
        p = auth.mean(axis=0)
        cnt = auth.sum(axis=1)
        unit_rows.append({"del_code": code, "delegation": u.delegation, "n_list": len(grp), "observed": n_auth,
                          "pred_mean": float(cnt.mean()), "p10": float(np.quantile(cnt, 0.1)), "p90": float(np.quantile(cnt, 0.9))})
        for (_, pr), pi in zip(grp.iterrows(), p):
            rows.append({"del_code": code, "rank": int(pr["rank"]), "p": float(pi), "obs": int(pr.licensed)})
    people, units = pd.DataFrame(rows), pd.DataFrame(unit_rows)
    brier = float(((people.p - people.obs) ** 2).mean())
    base = float(((people.obs.mean() - people.obs) ** 2).mean())
    cover = float(((units.observed >= units.p10) & (units.observed <= units.p90)).mean())
    people["bin"] = pd.cut(people.p, [-0.01, 0.05, 0.2, 0.5, 0.8, 1.01], labels=["< 5 %", "5–20 %", "20–50 %", "50–80 %", "> 80 %"])
    calib = people.groupby("bin", observed=False).agg(personnes=("obs", "size"), prob_moyenne=("p", "mean"), taux_observe=("obs", "mean")).reset_index()
    return {"people": people, "units": units, "brier": brier, "brier_climatology": base,
            "pred_total": float(units.pred_mean.sum()), "obs_total": int(units.observed.sum()),
            "coverage_p10_p90": cover, "bias_per_unit": float((units.pred_mean - units.observed).mean()),
            "mae_per_unit": float((units.pred_mean - units.observed).abs().mean()), "calibration": calib,
            "n_people": len(people), "n_units": len(units), "settings": settings}


# ---------------------------------------------------------------------------
# Calibrated parameters and full run
# ---------------------------------------------------------------------------
def load() -> dict:
    dele = pd.read_csv(DATA / "delegations.csv", dtype={"del_code": str})
    return {"delegations": dele, "population": pd.read_csv(DATA / "population.csv", dtype={"key": str}),
            "renewals": pd.read_csv(DATA / "renewal_records.csv", dtype=str).fillna(""),
            "waiting": pd.read_csv(DATA / "waiting_lists.csv", dtype=str).fillna("").assign(
                order=lambda w: pd.to_numeric(w["order"], errors="coerce")),
            "pharmacies": pd.read_csv(DATA / "pharmacies.csv", dtype=str).fillna(""),
            "projections": pd.read_csv(DATA / "population_projections.csv"),
            "dossiers": pd.read_csv(DATA / "officine_dossiers.csv"),
            "communes": pd.read_csv(DATA / "communes.csv"),
            "updates": pd.read_csv(DATA / "dpm_list_updates.csv")}


def calibrate(d: dict) -> dict:
    """Parameters used by default in the dashboard, each with its evidence."""
    popb = population_backtest(d["delegations"], d["population"], d["projections"])
    att = attrition_calibration(d["renewals"], d["delegations"], d["pharmacies"])
    now = nowcast_check(d["delegations"], d["waiting"])
    lam = exceptional_rate(now)
    nat = popb["national"]
    bfirst = b_first_licence(d["communes"])
    return {
        "b_first_hazard": bfirst["hazard"], "b_first": bfirst,
        "k_local": popb["k_local"], "growth_sd": popb["growth_sd"], "g_nat_1424": nat["g_1424"],
        "nat_path": national_projection_path(d["projections"], census_base=nat["census"][2024]),
        "h_engaged": att["h_engaged"], "h_all": att["h_all"], "h_sd": max((att["h_engaged_band"][1] - att["h_engaged_band"][0]) / 3.3, 0.004),
        "lam": round(lam["lam"], 4), "award_delay": 0.5,
        "population": popb, "attrition": att, "nowcast": now, "exceptional": lam,
    }


OLD_BACKTEST = {"mode": "census", "h": 0.0425, "h_sd": 0.01, "lam": 0.015, "pending_delay": 1.0, "award_delay": 0.0}


def processing_delays(dossiers: pd.DataFrame, updates: pd.DataFrame, pharm: pd.DataFrame) -> dict:
    """Category-A creation dossiers: time from the list revision to the request and to the licence (arrêté),
    and share of licensed holders still found in the 2026 registry (measures the registry-matching recall)."""
    d = dossiers[(dossiers.objet == "CREATION") & dossiers.delegation.notna()].drop_duplicates("nom").copy()
    for c in ("date_demande", "date_arrete"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    lists = np.sort(pd.to_datetime(updates.list_date).values)

    def prev(t):
        k = np.searchsorted(lists, np.datetime64(t)) - 1 if pd.notna(t) else -1
        return pd.Timestamp(lists[k]) if k >= 0 else pd.NaT
    d["list"] = d.date_demande.map(prev)
    to_req = ((d.date_demande - d.list).dt.days / 365.25).dropna()
    to_lic = ((d.date_arrete - d.list).dt.days / 365.25).dropna()
    lic = d[d.date_arrete.notna()].copy()
    parts = lic.nom.str.split()
    lic["prenom"], lic["nom"] = parts.str[-1], parts.str[:-1].str.join(" ")
    recall = float((registry_matches(lic, pharm) != "").mean()) if len(lic) else float("nan")
    q = lambda s: {"n": int(len(s)), "p10": float(s.quantile(0.1)), "median": float(s.median()), "p90": float(s.quantile(0.9))}
    return {"list_to_request": q(to_req), "list_to_licence": q(to_lic), "registry_recall": recall, "n_licensed": int(len(lic)),
            "source": "Pages DPM « suivi des dossiers des officines » archivées 2018-2024 (data/officine_dossiers.csv)"}


CAP_GRID = (1.0, 2.0, np.inf)
DELAY_GRID = (0.5, 1.0, 1.5)


def run_all(d: dict | None = None, n_sims: int = 300) -> dict:
    """Calibration + queue backtests.

    The award delay comes from the licence dossiers (list revision → arrêté); the number of licences
    processed per unit and year is chosen on the 2020→2026 backtest (Brier score). The full grid is kept
    for transparency: longer delays score slightly better there, partly because licences granted to people
    whose name is not found in the registry count as 'not licensed' (see registry_recall).
    """
    d = d or load()
    cal = calibrate(d)
    proc = processing_delays(d["dossiers"], d["updates"], d["pharmacies"])
    # licences open the year after the revision + Poisson(delay): 0.5 gives 61 % the same year, 91 % within
    # one more year, consistent with list → arrêté P50 ≈ 0.6 yr / P90 ≈ 1.2 yr
    delay0 = 0.5 if proc["list_to_licence"]["median"] < 1 else 1.0
    args = (d["delegations"], d["population"], d["renewals"], d["pharmacies"])
    qb_old = queue_backtest(*args, OLD_BACKTEST, n_sims)
    grid = []
    for delay in DELAY_GRID:
        for cap in CAP_GRID:
            st = {"mode": "estimates", "h": cal["h_all"], "h_sd": cal["h_sd"], "lam": cal["lam"], "pending_delay": delay,
                  "award_delay": delay, "growth_sd": cal["growth_sd"], "max_per_year": cap}
            q = queue_backtest(*args, st, n_sims)
            grid.append((q["brier"], delay, cap, q))
    best = min((g for g in grid if g[1] == delay0), key=lambda t: t[0])
    cal["award_delay"], cal["max_per_year"], cal["processing"] = best[1], best[2], proc
    grid_table = pd.DataFrame([{"Délai moyen (ans)": dl, "Licences / unité / an": ("illimité" if math.isinf(c) else str(int(c))),
                                "Brier": b, "Prévu": q["pred_total"], "Observé": q["obs_total"],
                                "Couverture P10-P90": q["coverage_p10_p90"]} for b, dl, c, q in grid])
    waits = observed_waits(d["waiting"], d["renewals"], d["delegations"], d["pharmacies"])
    return {"calibration": cal, "queue_old": qb_old, "queue_new": best[3], "process_grid": grid_table, "waits": waits}


RESULTS = DATA / "validation.json"
PARAM_KEYS = ("k_local", "growth_sd", "g_nat_1424", "h_engaged", "h_all", "h_sd", "lam", "award_delay", "max_per_year", "b_first_hazard")


def _plain(o):
    if isinstance(o, pd.DataFrame):
        return [_plain(r) for r in o.to_dict("records")]
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if math.isnan(o) else ("inf" if math.isinf(o) else float(o))
    return o


def save(res: dict, path: Path = RESULTS) -> dict:
    """Compact, offline copy of the calibration and backtests used by the dashboard."""
    import datetime as dt
    import json
    cal = res["calibration"]
    out = {
        "generated": dt.date.today().isoformat(),
        "params": {k: cal[k] for k in PARAM_KEYS},
        "nat_path": cal["nat_path"],
        "population": {k: cal["population"][k] for k in ("table", "k_local", "oracle_rmse", "growth_sd", "local_sd", "national", "n_units", "zriba", "source")},
        "attrition": cal["attrition"], "exceptional": cal["exceptional"], "b_first": cal["b_first"], "processing": cal["processing"],
        "nowcast": {k: v for k, v in cal["nowcast"].items()},
        "waits": res["waits"], "process_grid": res["process_grid"],
        "queue": {name: {k: res[name][k] for k in ("brier", "brier_climatology", "pred_total", "obs_total", "coverage_p10_p90",
                                                   "bias_per_unit", "mae_per_unit", "calibration", "n_people", "n_units", "settings")}
                  for name in ("queue_old", "queue_new")},
    }
    out = _plain(out)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load_saved(path: Path = RESULTS) -> dict | None:
    import json
    if not path.exists():
        return None
    v = json.loads(path.read_text(encoding="utf-8"))
    v["params"] = {k: (math.inf if x == "inf" else x) for k, x in v["params"].items()}
    v["nat_path"] = {int(k): x for k, x in v["nat_path"].items()}
    return v


def _fmt_pct(x):
    return f"{x:.2%}"


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 200)
    res = run_all()
    cal = res["calibration"]
    pb = cal["population"]
    print("== Backtest population 2014→2024 ==", pb["n_units"], "unités")
    print(pb["table"].round(2).to_string(index=False))
    print("k_local", pb["k_local"], "oracle RMSE", {k: round(v * 100, 3) for k, v in pb["oracle_rmse"].items()},
          "growth_sd", _fmt_pct(pb["growth_sd"]), "national", {k: v for k, v in pb["national"].items() if k != "census"})
    a = cal["attrition"]
    print("\n== Attrition ==  train 2020→2024", {k: round(v, 4) for k, v in a["train"].items()},
          "regul22", round(a["regul_2022"], 3), "regul24", round(a["regul_2024"], 3))
    for k, v in a["test"].items():
        print(" ", k, {kk: round(vv, 4) for kk, vv in v.items()})
    print("  h_old", round(a["h_old"], 4))
    nc = cal["nowcast"]
    print("\n== Nowcast ==", {k: v for k, v in nc.items() if k != "table"})
    print("λ", cal["exceptional"])
    w = res["waits"]
    print("\n== Attentes observées ==")
    for k, v in w.items():
        print(" ", k, v)
    for name in ("queue_old", "queue_new"):
        q = res[name]
        print(f"\n== Backtest file d'attente 2020→2026 ({name}) ==", q["settings"])
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()
               if k in ("brier", "brier_climatology", "pred_total", "obs_total", "coverage_p10_p90", "bias_per_unit", "mae_per_unit", "n_people", "n_units")})
        print(q["calibration"].round(3).to_string(index=False))
    print(res["process_grid"].round(4).to_string(index=False))
    print("processing", cal["processing"], "-> award_delay", cal["award_delay"], "max_per_year", cal["max_per_year"])
    print("B first licence", cal["b_first"])
    save(res)
    print("→", RESULTS.relative_to(ROOT))
