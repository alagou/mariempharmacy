"""Population projections, licensing thresholds and waiting-list (queue) forecasts.

Every function is deterministic for a given seed so that results are reproducible.
Observed values (censuses, official lists) are never modified here: projections are
returned as separate, explicitly labelled series.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BASE_YEAR = 2024          # last census (RGPH 2024) used by the DPM
NEXT_CENSUSES = (2034, 2044, 2054)
DEFAULT_HORIZON = 2050       # queue forecasts; population projections are shown to 2045 by default
PROJECTION_END = 2045


# ---------------------------------------------------------------------------
# Licensing rules (décret 2019-1013)
# ---------------------------------------------------------------------------
def allowed_A(pop: float, threshold: float) -> int:
    """Category A: one pharmacy per *semi-integer* tranche (art. 5)."""
    if pop is None or threshold is None or pd.isna(pop) or pd.isna(threshold) or threshold <= 0:
        return 0
    return int(math.floor(pop / threshold + 0.5))


def allowed_B(pop: float, threshold: float) -> int:
    """Category B: one pharmacy per *incomplete* tranche (art. 11)."""
    if pop is None or threshold is None or pd.isna(pop) or pd.isna(threshold) or threshold <= 0:
        return 0
    return int(math.ceil(pop / threshold))


def next_threshold_pop(existing: int, threshold: float, rule: str = "A") -> float:
    """Population needed for one more licence than `existing`."""
    if rule == "A":
        return (existing + 0.5) * threshold
    return existing * threshold + 1


# ---------------------------------------------------------------------------
# Population growth
# ---------------------------------------------------------------------------
def cagr(p0: float, p1: float, years: float) -> float | None:
    if p0 is None or p1 is None or pd.isna(p0) or pd.isna(p1) or p0 <= 0 or years <= 0:
        return None
    return (p1 / p0) ** (1 / years) - 1


@dataclass
class GrowthRates:
    central: float
    low: float
    high: float
    sd: float
    linear_slope: float | None
    basis: str
    observed: dict = field(default_factory=dict)


def growth_rates(p2024: float, p2014: float | None, p2004: float | None,
                 fallback: float | None = None, fallback_label: str = "") -> GrowthRates:
    """Low/central/high annual growth from census observations.

    central = CAGR 2014-2024; low/high = min/max of the CAGRs 2004-2014 and 2014-2024
    (±0.5 point when only one period is observed). When the unit has no comparable
    history, the governorate growth (fallback) is used and the band is widened.
    """
    r2 = cagr(p2014, p2024, 10)
    r1 = cagr(p2004, p2014, 10)
    observed = {y: v for y, v in ((2004, p2004), (2014, p2014), (2024, p2024)) if v is not None and not pd.isna(v)}
    slope = None
    if len(observed) >= 2:
        xs, ys = np.array(list(observed)), np.array(list(observed.values()), dtype=float)
        slope = float(np.polyfit(xs, ys, 1)[0])
    if r2 is not None and r1 is not None:
        lo, hi = min(r1, r2), max(r1, r2)
        if hi - lo < 0.002:
            lo, hi = r2 - 0.002, r2 + 0.002
        return GrowthRates(r2, lo, hi, max((hi - lo) / 2, 0.002), slope, "Recensements 2004-2014-2024 de l'unité", observed)
    if r2 is not None:
        return GrowthRates(r2, r2 - 0.005, r2 + 0.005, 0.004, slope, "Recensements 2014-2024 de l'unité", observed)
    fb = fallback if fallback is not None else 0.008
    return GrowthRates(fb, fb - 0.007, fb + 0.007, 0.005, None,
                       f"Pas d'historique comparable : croissance du gouvernorat {fallback_label} (estimation)", observed)


def project_population(p0: float, rates: GrowthRates, start: int = BASE_YEAR, end: int = PROJECTION_END) -> pd.DataFrame:
    years = np.arange(start, end + 1)
    t = years - start
    out = pd.DataFrame({"year": years})
    out["Faible"] = p0 * (1 + rates.low) ** t
    out["Central (TCAM)"] = p0 * (1 + rates.central) ** t
    out["Élevé"] = p0 * (1 + rates.high) ** t
    if rates.linear_slope is not None:
        out["Linéaire"] = p0 + rates.linear_slope * t
    return out


def year_reaching(p0: float, rate: float, target: float, start: int = BASE_YEAR,
                  mode: str = "census", horizon: int = 2060) -> int | None:
    """First year in which the *official* population reaches `target`.

    mode='annual': official figure updated every year (INS estimates).
    mode='census': official figure only changes at censuses (2034, 2044, ...).
    """
    if p0 >= target:
        return start
    if rate <= 0:
        return None
    if mode == "annual":
        y = start + math.log(target / p0) / math.log(1 + rate)
        y = math.ceil(y)
        return y if y <= horizon else None
    for c in NEXT_CENSUSES:
        if c > horizon:
            break
        if p0 * (1 + rate) ** (c - start) >= target:
            return c
    return None


# ---------------------------------------------------------------------------
# Waiting-list attrition from successive official renewal tables
# ---------------------------------------------------------------------------
def _key(nom: str, prenom: str, date: str) -> str:
    def norm(s):
        return re.sub(r"[^A-Z]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper())
    nom = re.split(r"\bEP\b|\bEPOUSE\b", str(nom).upper())[0]
    return f"{norm(nom)}|{norm(prenom)[:6]}|{date}"


def attrition_estimates(renewals: pd.DataFrame, category: str = "A") -> dict:
    """Annual probability that a registrant who is NOT first in line leaves the list.

    Uses the 2024 renewed table vs the 2026 renewed / not-renewed tables. A share of the
    'not renewed' people regularise later (measured with 2024 -> 2026); that share is
    removed from the exits. Rank-1/2 people are excluded because their exits are mostly
    authorisations rather than attrition.
    """
    r = renewals[renewals.category == category].copy()
    r["k"] = [_key(a, b, c) for a, b, c in zip(r.nom, r.prenom, r.date_inscription)]
    base = r[r.list_type.isin(["renouvele_2024", "renouvele_sans_attestation_2024"])].drop_duplicates("k")
    base = base[base.delegation.fillna("") != ""] if category == "A" else base[base.commune.fillna("") != ""]
    unit = "delegation" if category == "A" else "commune"
    base = base.assign(rank=base.groupby(["governorate", unit]).date_inscription.rank(method="first"))
    kept26 = set(r[r.list_type == "renouvele_2026"].k)
    non26 = set(r[r.list_type == "non_renouvele_2026"].k)
    non24 = set(r[r.list_type == "non_renouvele_2024"].k)
    regul = len(non24 & kept26) / max(len(non24), 1)       # share of non-renewed who came back
    sub = base[base["rank"] >= 3]
    n = len(sub)
    exits_other = int((~sub.k.isin(kept26) & ~sub.k.isin(non26)).sum())
    nonren = int(sub.k.isin(non26).sum())
    eff_exits = exits_other + nonren * (1 - regul)
    p2 = eff_exits / max(n, 1)
    h = 1 - (1 - p2) ** 0.5
    # Beta posterior (Jeffreys) for the 2-year exit probability -> annual band
    a, b = eff_exits + 0.5, n - eff_exits + 0.5
    lo2, hi2 = _beta_ppf(0.05, a, b), _beta_ppf(0.95, a, b)
    top = base[base["rank"] == 1]
    top_exit = float((~top.k.isin(kept26) & ~top.k.isin(non26)).mean()) if len(top) else float("nan")
    return {
        "category": category, "n_at_risk": n, "exits_other": exits_other, "non_renewed": nonren,
        "regularisation_share": regul, "two_year_exit": p2, "h_central": h,
        "h_low": 1 - (1 - lo2) ** 0.5, "h_high": 1 - (1 - hi2) ** 0.5,
        "rank1_two_year_exit": top_exit, "n_rank1": int(len(top)),
        "source": "Tableaux DPM des renouvellements 2024 et 2026 (renouvelés / non renouvelés)",
    }


def _beta_ppf(q: float, a: float, b: float) -> float:
    try:
        from scipy.stats import beta
        return float(beta.ppf(q, a, b))
    except Exception:  # noqa: BLE001 - normal approximation fallback
        m = a / (a + b)
        s = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        z = {0.05: -1.645, 0.95: 1.645}.get(q, 0)
        return min(max(m + z * s, 0.0), 1.0)


def backtest_attrition(renewals: pd.DataFrame) -> dict:
    """Out-of-sample check: hazard estimated on 2020 -> 2024, compared with 2024 -> 2026."""
    r = renewals[renewals.category == "A"].copy()
    r["k"] = [_key(a, b, c) for a, b, c in zip(r.nom, r.prenom, r.date_inscription)]
    s20 = r[r.list_type == "renouvele_2020"].drop_duplicates("k")
    s20 = s20[s20.delegation.fillna("") != ""]
    s20 = s20.assign(rank=s20.groupby(["governorate", "delegation"]).date_inscription.rank(method="first"))
    s20 = s20[s20["rank"] >= 3]
    in24 = set(r[r.list_type.str.contains("2024") & ~r.list_type.str.startswith("non_")].k)
    stay4 = s20.k.isin(in24).mean() if len(s20) else float("nan")
    h_train = 1 - stay4 ** 0.25 if stay4 == stay4 else float("nan")
    test = attrition_estimates(renewals, "A")
    predicted_2y_stay = (1 - h_train) ** 2
    observed_2y_stay = 1 - test["two_year_exit"]
    return {
        "train_period": "2020 → 2024", "test_period": "2024 → 2026",
        "n_train": int(len(s20)), "h_train": h_train,
        "predicted_two_year_stay": predicted_2y_stay, "observed_two_year_stay": observed_2y_stay,
        "abs_error_points": abs(predicted_2y_stay - observed_2y_stay) * 100,
        "caveat": "La liste 2020 publiée en juin 2020 était incomplète (renouvellements encore en cours) : "
                  "les sorties 2020-2024 sont donc surestimées. Résultat indicatif.",
    }


def backtest_capacity(delegations: pd.DataFrame) -> dict:
    """How well does the decree formula reproduce today's number of Category A pharmacies?"""
    d = delegations.dropna(subset=["zone_threshold", "population_dpm", "pharmacies_A"]).copy()
    d["allowed"] = [allowed_A(p, n) for p, n in zip(d.population_dpm, d.zone_threshold)]
    diff = d.pharmacies_A - d.allowed
    return {
        "n_units": int(len(d)),
        "share_equal": float((diff == 0).mean()),
        "share_within_1": float((diff.abs() <= 1).mean()),
        "units_above": int((diff > 0).sum()),
        "units_below": int((diff < 0).sum()),
        "note": "Écart positif = officines créées sous les anciennes règles ou par dérogation (art. 6) ; "
                "écart négatif = créations en attente après application du RGPH 2024.",
    }


# ---------------------------------------------------------------------------
# Queue simulation (Monte Carlo)
# ---------------------------------------------------------------------------
@dataclass
class QueueParams:
    position: int                 # 1 = first in line
    existing: int                 # licences already granted in the unit (same category)
    p0: float                     # official population (base year)
    threshold: float              # numerus clausus N
    rule: str = "A"               # "A" (semi-integer) or "B" (incomplete tranche)
    growth_mean: float = 0.01
    growth_sd: float = 0.003
    h_mean: float = 0.04          # annual attrition of people ahead
    h_sd: float = 0.01
    lam_exceptional: float = 0.01 # extra licences per year (art. 6, closures...)
    first_licence_now: bool = False  # e.g. commune without any category-B pharmacy
    pop_mode: str = "census"      # "census" or "annual"
    admin_lag: int = 2            # years between an official population update and the licence
    open_slot_delay_extra: float = 1.0  # mean extra years (Poisson) to process an already-open slot
    start_year: int = 2026
    horizon: int = DEFAULT_HORIZON
    n_sims: int = 4000
    seed: int = 42


def simulate_queue(p: QueueParams) -> np.ndarray:
    """Return simulated authorisation years (np.inf = not before the horizon)."""
    rng = np.random.default_rng(p.seed)
    n = p.n_sims
    years = np.arange(p.start_year, p.horizon + 1)
    ahead = np.full(n, max(p.position - 1, 0), dtype=float)
    result = np.full(n, np.inf)
    g = rng.normal(p.growth_mean, p.growth_sd, n)
    a_h, b_h = _beta_params(p.h_mean, p.h_sd)
    h = rng.beta(a_h, b_h, n) if p.h_mean > 0 else np.zeros(n)
    granted = np.full(n, float(p.existing))
    rule_fn = allowed_A if p.rule == "A" else allowed_B
    if p.rule == "A":
        def vec_allowed(pop):
            return np.floor(pop / p.threshold + 0.5)
    else:
        def vec_allowed(pop):
            return np.ceil(pop / p.threshold)
    # licences already available today (population allows more than existing)
    today_allowed = rule_fn(p.p0, p.threshold)
    pending = np.full(n, float(max(today_allowed - p.existing, 0)))
    if p.first_licence_now and p.existing == 0:
        pending = np.maximum(pending, 1.0)
    # open slots: convocation, 30+30 days for the file, CNOPT opinion, ministerial decision
    pending_year = p.start_year + 1 + rng.poisson(p.open_slot_delay_extra, n)
    for y in years:
        active = np.isinf(result)
        # attrition of people ahead (non-renewal, withdrawal, death, other list...)
        if p.h_mean > 0:
            leave = rng.binomial(ahead.astype(int), np.clip(h, 0, 1))
            ahead = np.where(active, ahead - leave, ahead)
        # new licences this year
        slots = np.where(pending_year == y, pending, 0.0)
        eff_year = y - p.admin_lag
        if p.pop_mode == "annual":
            update = eff_year >= BASE_YEAR + 1
        else:
            update = eff_year in NEXT_CENSUSES
        if update:
            pop = p.p0 * (1 + g) ** (eff_year - BASE_YEAR)
            allowed = vec_allowed(pop)
            new = np.maximum(allowed - (granted + pending), 0)
            slots += new
            granted += new
        if p.lam_exceptional > 0:
            exc = rng.poisson(p.lam_exceptional, n)
            slots += exc
            granted += exc  # art. 7: counted within the numerus clausus
        # allocate slots in list order
        got = active & (slots > ahead)
        result[got] = y
        ahead = np.where(active & ~got, np.maximum(ahead - slots, 0), ahead)
    return result


def _beta_params(mean: float, sd: float) -> tuple[float, float]:
    mean = min(max(mean, 1e-4), 0.999)
    var = min(sd ** 2, mean * (1 - mean) * 0.99)
    k = mean * (1 - mean) / var - 1
    return max(mean * k, 1e-3), max((1 - mean) * k, 1e-3)


PROB_BANDS = [(0.05, "< 5 %"), (0.20, "5 – 20 %"), (0.50, "20 – 50 %"), (0.80, "50 – 80 %"), (1.01, "> 80 %")]


def prob_band(p: float) -> str:
    for lim, lab in PROB_BANDS:
        if p < lim:
            return lab
    return "> 80 %"


def summarize(years: np.ndarray, start_year: int, horizon: int) -> dict:
    finite = years[np.isfinite(years)]
    share = len(finite) / len(years)

    def within(k):
        return float((years <= start_year - 1 + k).mean())

    capped = np.where(np.isfinite(years), years, 1e9)

    def q(x):
        v = np.quantile(capped, x, method="lower")
        return int(v) if v < 1e9 else None

    return {
        "p3": within(3), "p5": within(5), "p10": within(10), "p_horizon": share,
        "q10": q(0.10), "q25": q(0.25), "q50": q(0.50), "q75": q(0.75), "q90": q(0.90),
        "band3": prob_band(within(3)), "band5": prob_band(within(5)), "band10": prob_band(within(10)),
        "band_h": prob_band(share), "horizon": horizon,
        "median_or_later": q(0.50) if q(0.50) is not None else horizon + 1,
    }


def period_label(s: dict) -> str:
    """Human-readable 'most likely period' (interquartile range), honest about the horizon."""
    h = s["horizon"]
    if s["q25"] is None:
        return f"après {h}"
    if s["q75"] is None:
        return f"{s['q25']} – après {h}"
    if s["q25"] == s["q75"]:
        return str(s["q25"])
    return f"{s['q25']} – {s['q75']}"
