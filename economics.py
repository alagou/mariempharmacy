"""Purchasing power and pharmacy income.

Official series (censuses, INS regional expenditure, the 2024 officinal market, the
legal zone threshold) stay untouched. Everything else is labelled as a scenario:

* future industrial jobs move the resident population only when a figure is published;
* one new job is not one new inhabitant (commuters are already counted where they sleep);
* pharmacy turnover is an equal share of the national medicine market, scaled by the
  legal threshold, not a set of accounts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as an
import forecasting as fc

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load() -> dict:
    return json.loads((DATA / "economics.json").read_text(encoding="utf-8"))


def national_spend_2024(eco: dict | None = None) -> float:
    eco = eco or load()
    return eco["market"]["public_price_tnd"] / eco["population_national_2024"]


def threshold_reference(delegations: pd.DataFrame) -> float:
    """Population-weighted harmonic mean of the legal thresholds.

    An index of ``reference / threshold`` then averages to 1, so reallocating the
    national per-capita medicine spend by this index still sums to the national market
    on the units that have a threshold.
    """
    d = delegations.dropna(subset=["population_dpm", "zone_threshold"])
    d = d[d.zone_threshold > 0]
    return float(d.population_dpm.sum() / (d.population_dpm / d.zone_threshold).sum())


def medicine_index(threshold: float, reference: float) -> float:
    if threshold is None or pd.isna(threshold) or threshold <= 0:
        return float("nan")
    return reference / float(threshold)


def region_of(eco: dict, governorate: str) -> dict:
    name = eco["governorate_region"].get(governorate, "")
    row = dict(eco["regions"].get(name, {}))
    row["name"] = name
    return row


def poverty(eco: dict, governorate: str, delegation: str) -> tuple[float, str]:
    key = f"{governorate}|{delegation}"
    if key in eco["poverty_delegation_2015"]:
        return eco["poverty_delegation_2015"][key], "délégation, carte INS 2015"
    if governorate in eco["poverty_governorate_2015"]:
        return eco["poverty_governorate_2015"][governorate], "gouvernorat, enquête 2015"
    return float("nan"), "non publié"


def consumption_proxy(eco: dict, governorate: str, delegation: str) -> dict:
    """Regional DPA (official) and a local proxy (not an INS delegation figure)."""
    reg = region_of(eco, governorate)
    p, level = poverty(eco, governorate, delegation)
    dpa = reg.get("dpa_2021")
    p_reg = reg.get("poverty_2015")
    proxy = None
    if dpa is not None and p_reg is not None and p == p and p_reg < 1:
        proxy = dpa * (1 - p) / (1 - p_reg)
    return {"region": reg.get("name", ""), "dpa_2021": dpa, "poverty_2021": reg.get("poverty_2021"),
            "poverty": p, "poverty_level": level, "dpa_proxy": proxy}


def central_population(p0: float, g: fc.GrowthRates, settings: an.SimSettings, last_year: int) -> pd.Series:
    """Central official population at 1 January of base+1 … last_year. Same path as the queue model."""
    gm, _, path = an.growth_inputs(g, settings)
    years = np.arange(fc.BASE_YEAR + 1, last_year + 1)
    pops = []
    seq = list(path) if path is not None else None
    for y in years:
        k = int(y - (fc.BASE_YEAR + 1))
        if seq is not None:
            use = seq if k < len(seq) else seq + [seq[-1]] * (k + 1 - len(seq))
            prod = 1.0
            for x in use[1:k + 1]:
                prod *= 1 + x
            pops.append(float(p0) * prod)
        else:
            pops.append(float(p0) * (1 + gm) ** k)
    return pd.Series(pops, index=[int(y) for y in years])


def _mode_params(eco: dict, mode: str) -> tuple[str, float]:
    key = {"low": "jobs_new_low", "high": "jobs_new_high"}.get(mode, "jobs_new_central")
    coef = eco["settlement_per_job"][{"low": "low", "high": "high"}.get(mode, "central")]
    return key, coef


def events_touching(eco: dict, governorate: str, delegation: str) -> list[dict]:
    out = []
    for e in eco["events"]:
        places = [(e["governorate"], e["delegation"])] + [tuple(x.values()) for x in e.get("also", [])]
        if (governorate, delegation) in places:
            out.append(e)
    return out


def extra_residents(eco: dict, governorate: str, delegation: str, year: int, mode: str = "central") -> float:
    """Residents added by projects that are not already in the 2024 census."""
    job_key, coef = _mode_params(eco, mode)
    total = 0.0
    for e in events_touching(eco, governorate, delegation):
        jobs = e.get(job_key) or 0
        start = e.get("start_year")
        if e.get("in_census_2024") or not jobs or not start:
            continue
        ramp = max(int(e.get("ramp_years") or 1), 1)
        if year < start:
            continue
        weight = min(1.0, (year - start + 1) / ramp)
        total += jobs * coef * weight
    return total


def spend_per_person(eco: dict, year: int, threshold: float, reference: float, price_growth: float) -> float:
    base = national_spend_2024(eco) * medicine_index(threshold, reference)
    return base * (1 + price_growth) ** (year - eco["market"]["year"])


def ramp_factor(eco: dict, age: int, enabled: bool) -> float:
    if not enabled:
        return 1.0
    ramp = eco["ramp"]
    return ramp[age] if age < len(ramp) else ramp[-1]


def sharing(existing: int, population: float, threshold: float, opened: bool) -> int:
    """Pharmacies sharing the delegation once she is open (every legal slot is assumed filled)."""
    allowed = fc.allowed_A(population, threshold)
    if not opened:
        return max(int(existing), 1)
    return max(int(existing) + 1, allowed)


def annual_accounts(population: float, threshold: float, existing: int, opened: bool, spend: float,
                    eco: dict, ramp_f: float = 1.0) -> dict:
    n = sharing(existing, population, threshold, opened)
    ca = 0.0 if not opened else population * spend / n * ramp_f
    m = eco["margins"]
    return {"pharmacies": n, "ca": ca, "gross": ca * m["gross_central"],
            "net_low": ca * m["net_low"], "net": ca * m["net_central"], "net_high": ca * m["net_high"]}


def capacity_today(population: float, threshold: float, existing: int, eco: dict, reference: float) -> dict:
    """Equal share of the pharmacies already open, at 2024 prices. She is not in the count."""
    spend = national_spend_2024(eco) * medicine_index(threshold, reference)
    n = max(int(existing), 1)
    ca = population * spend / n
    m = eco["margins"]
    return {"spend_per_person": spend, "pharmacies": n, "ca": ca, "gross": ca * m["gross_central"],
            "net_low": ca * m["net_low"], "net": ca * m["net_central"], "net_high": ca * m["net_high"]}


def yearly_income(p0: float, g: fc.GrowthRates, settings: an.SimSettings, governorate: str, delegation: str,
                  threshold: float, existing: int, open_year: int | None, price_growth: float,
                  ramp: bool, mode: str = "central", eco: dict | None = None, reference: float | None = None) -> pd.DataFrame:
    eco = eco or load()
    if reference is None:
        raise ValueError("reference")
    pop = central_population(p0, g, settings, settings.horizon)
    rows = []
    cumul = 0.0
    for year in range(settings.start_year, settings.horizon + 1):
        residents = float(pop.loc[year]) + extra_residents(eco, governorate, delegation, year, mode)
        opened = open_year is not None and year >= open_year
        age = (year - open_year) if opened else 0
        spend = spend_per_person(eco, year, threshold, reference, price_growth)
        acc = annual_accounts(residents, threshold, existing, opened, spend, eco, ramp_factor(eco, age, ramp and opened))
        if opened:
            cumul += acc["net"]
        rows.append({"Année": year, "Habitants": residents, "Officines en partage": acc["pharmacies"],
                     "CA médicaments": acc["ca"], "Marge brute": acc["gross"],
                     "Résultat net bas": acc["net_low"], "Résultat net": acc["net"],
                     "Résultat net haut": acc["net_high"], "Cumul net": cumul if opened or cumul else 0.0})
    return pd.DataFrame(rows)


def crossing_year(pop: pd.Series, extra_at, target: float) -> int | None:
    if target is None or pd.isna(target):
        return None
    for year, base in pop.items():
        if base + extra_at(int(year)) >= target:
            return int(year)
    return None


def historical_zaghouan(delegations: pd.DataFrame, g_nat: float) -> dict:
    z = delegations[delegations.governorate == "ZAGHOUAN"]
    p14 = float(z.pop_2014.sum())
    p24 = float(z.population_dpm.sum())
    g = fc.cagr(p14, p24, 10)
    extra = p14 * ((1 + g) ** 10 - (1 + g_nat) ** 10)
    return {"pop_2014": p14, "pop_2024": p24, "cagr": g, "national_cagr": g_nat,
            "excess_residents": extra}


def zone_table(delegations: pd.DataFrame, compare: pd.DataFrame, settings: an.SimSettings,
               price_growth: float, ramp: bool, mode: str = "central") -> pd.DataFrame:
    eco = load()
    ref = threshold_reference(delegations)
    rows = []
    merged = compare.dropna(subset=["population_dpm", "zone_threshold"]).copy()
    for r in merged.itertuples(index=False):
        cons = consumption_proxy(eco, r.governorate, r.delegation)
        cap = capacity_today(r.population_dpm, r.zone_threshold, int(r.pharmacies_A), eco, ref)
        g = an.growth_for_delegation(r)
        open_m = int(r.sim_median) if pd.notna(r.sim_median) and int(r.sim_median) <= settings.horizon else None
        open_e = int(r.sim_q10) if pd.notna(r.sim_q10) and int(r.sim_q10) <= settings.horizon else None
        y_m = yearly_income(r.population_dpm, g, settings, r.governorate, r.delegation, r.zone_threshold,
                            int(r.pharmacies_A), open_m, price_growth, ramp, mode, eco, ref)
        y_e = yearly_income(r.population_dpm, g, settings, r.governorate, r.delegation, r.zone_threshold,
                            int(r.pharmacies_A), open_e, price_growth, ramp, mode, eco, ref)
        pop = central_population(r.population_dpm, g, settings, settings.horizon)
        extra_2035 = extra_residents(eco, r.governorate, r.delegation, 2035, mode)
        base_cross = crossing_year(pop, lambda _y: 0.0, r.next_pop_A)
        evt_cross = crossing_year(pop, lambda y, gov=r.governorate, dele=r.delegation: extra_residents(eco, gov, dele, y, mode), r.next_pop_A)
        rows.append({
            "del_code": r.del_code, "Délégation": str(r.delegation).title(), "Gouvernorat": str(r.governorate).title(),
            "km": r.distance_km, "Zone N": int(r.zone_threshold),
            "Indice médicament": medicine_index(r.zone_threshold, ref),
            "Région": cons["region"], "DPA région 2021": cons["dpa_2021"],
            "Proxy DPA": cons["dpa_proxy"], "Pauvreté 2015": cons["poverty"],
            "Niveau pauvreté": cons["poverty_level"],
            "DT / hab. médicament": cap["spend_per_person"],
            "CA officine en place": cap["ca"], "Net officine en place": cap["net"],
            "Ouverture médiane": open_m, "Ouverture tôt (P10)": open_e,
            "Cumul net médiane": float(y_m["Cumul net"].iloc[-1]),
            "Cumul net tôt": float(y_e["Cumul net"].iloc[-1]),
            "Habitants en plus (2035)": extra_2035,
            "Seuil franchi (démographie)": base_cross, "Seuil franchi (événements)": evt_cross,
        })
    return pd.DataFrame(rows)


def mature_net(row_pop: float, threshold: float, existing: int, year: int, price_growth: float,
               eco: dict, reference: float) -> float:
    """Net income of an extra pharmacy in `year`, at full ramp, on the demographic population."""
    spend = spend_per_person(eco, year, threshold, reference, price_growth)
    acc = annual_accounts(row_pop, threshold, existing, True, spend, eco, 1.0)
    return acc["net"]
