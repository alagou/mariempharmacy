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
NEXT_CENSUSES = (2034, 2044, 2054, 2064)
DEFAULT_HORIZON = 2070       # single source of truth for queue forecasts, population charts and labels
PROJECTION_END = DEFAULT_HORIZON
# Last year covered by the official INS projection (Flash 2026, base RGPH 2024). Beyond it the national
# growth is held at the last published rate and the per-simulation growth uncertainty widens.
OFFICIAL_PROJECTION_END = 2054
SD_WIDEN_PER_DECADE = 0.5    # growth sd × (1 + 0.5 per decade beyond OFFICIAL_PROJECTION_END)


def growth_sd_factor(years) -> np.ndarray:
    return 1 + SD_WIDEN_PER_DECADE * np.maximum(np.asarray(years, dtype=float) - OFFICIAL_PROJECTION_END, 0) / 10


def after_horizon(horizon: int = DEFAULT_HORIZON) -> str:
    return f"après {horizon}"


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


def year_reaching_path(p0: float, path, target: float, base_year: int = BASE_YEAR, horizon: int = DEFAULT_HORIZON) -> int | None:
    """First year whose official population at 1 January reaches `target` (pop(1/1/base+1) = p0,
    path[k] = growth during the year ending 1/1/(base+1+k), as in QueueParams.growth_path)."""
    if p0 >= target:
        return base_year
    pop = p0
    for k, g in enumerate(path[1:], start=1):
        y = base_year + 1 + k
        if y > horizon:
            break
        pop *= 1 + g
        if pop >= target:
            return y
    return None


def year_reaching(p0: float, rate: float, target: float, start: int = BASE_YEAR,
                  mode: str = "census", horizon: int = DEFAULT_HORIZON) -> int | None:
    """First year in which the *official* population reaches `target`.

    mode='annual' / 'estimates': official figure updated every year (INS estimates).
    mode='census': official figure only changes at censuses (2034, 2044, ...).
    """
    if p0 >= target:
        return start
    if rate <= 0:
        return None
    if mode in ("annual", "estimates"):
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
# DPM list revisions 2017-2025 (data/dpm_list_updates.csv): 11 lists, at least one per year except 2024
# (waiting for RGPH 2024), each based on the latest INS estimate at 1 January of the same or previous year.
FIRST_LIST_REVISION = 2026
REVISION_STEP = 1


@dataclass
class QueueParams:
    position: int                 # 1 = first in line
    existing: int                 # licences already granted in the unit (same category)
    p0: float                     # official population at the base date (RGPH 2024 ≈ 1/1/2025)
    threshold: float              # numerus clausus N
    rule: str = "A"               # "A" (semi-integer) or "B" (incomplete tranche)
    growth_mean: float = 0.01     # constant central growth (used when growth_path is None)
    growth_sd: float = 0.003      # sd of the per-simulation growth shift (calibrated on the 2014→2024 backtest)
    growth_path: tuple | None = None  # central annual growth for each year from base_year (index 0)
    h_mean: float = 0.04          # annual attrition of people ahead
    h_sd: float = 0.01
    lam_exceptional: float = 0.01 # extra licences per year (art. 6, closures...)
    first_licence_now: bool = False  # e.g. commune without any category-B pharmacy
    pop_mode: str = "estimates"   # "estimates" (DPM practice since 2017), "census" or "annual"
    admin_lag: int = 2            # census/annual modes: years between the population reference and the licence
    award_delay: float = 0.5      # estimates mode: mean extra years (Poisson) after the year following a list revision
    open_slot_delay_extra: float = 1.0  # mean extra years (Poisson) to process an already-open slot
    max_per_year: float = np.inf  # licences processed per unit and year (DPM convokes one candidate at a time)
    pending_hazard: float | None = None  # annual probability that an already-open slot is awarded (replaces the Poisson delay)
    revision_years: tuple | None = None  # estimates mode: list revision years (population at 1 January)
    base_year: int = BASE_YEAR
    start_year: int = 2026
    horizon: int = DEFAULT_HORIZON
    n_sims: int = 4000
    seed: int = 42
    # central path: every random draw replaced by its central value (mean growth and attrition, median delays,
    # exceptional licences only once their expected cumulative number reaches 1); used to explain one option
    deterministic: bool = False


def population_paths(p: QueueParams, rng: np.random.Generator, last_year: int,
                     z: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulated official population at 1 January of base_year+1 … last_year (one row per simulation).

    z: optional growth shift in standard deviations per simulation (default: drawn from N(0, 1))."""
    yrs = np.arange(p.base_year + 1, last_year + 1)
    k = len(yrs)
    if p.growth_path is not None:
        path = np.asarray(p.growth_path, dtype=float)
        path = np.concatenate([path, np.full(max(k - len(path), 0), path[-1] if len(path) else p.growth_mean)])[:k]
    else:
        path = np.full(k, p.growth_mean)
    zz = rng.normal(0.0, 1.0, p.n_sims) if z is None else np.asarray(z, dtype=float)
    shift = (zz * p.growth_sd)[:, None] * growth_sd_factor(yrs)[None, :]
    g = np.maximum(path[None, :] + shift, -0.05)
    pop = p.p0 * np.cumprod(1 + g, axis=1) / (1 + g[:, :1])   # pop at 1/1/base+1 = p0
    return yrs, pop


def _update_events(p: QueueParams) -> list[tuple[int, int]]:
    """(year the new licences become available, reference year of the official population)."""
    if p.pop_mode == "estimates":
        revs = p.revision_years or tuple(range(FIRST_LIST_REVISION, p.horizon + 1, REVISION_STEP))
        return [(r + 1, r) for r in revs if r + 1 <= p.horizon]
    # population at 1 January of ref; a census held in year c is taken as the population of 1/1/(c+1)
    if p.pop_mode == "annual":
        return [(y, y - p.admin_lag + 1) for y in range(p.start_year, p.horizon + 1) if y - p.admin_lag >= p.base_year + 1]
    return [(c + p.admin_lag, c + 1) for c in NEXT_CENSUSES if c + p.admin_lag <= p.horizon]


def poisson_median(lam: float) -> int:
    k, term = 0, math.exp(-lam)
    cdf = term
    while cdf < 0.5:
        k += 1
        term *= lam / k
        cdf += term
    return k


def geometric0_median(p: float) -> int:
    """Median of (Geometric(p) - 1), i.e. number of full years before the event."""
    return max(math.ceil(math.log(0.5) / math.log(1 - p)) - 1, 0)


def simulate_queue(p: QueueParams, trace: dict | None = None) -> np.ndarray:
    """Return simulated authorisation years (np.inf = not before the horizon).

    trace: if given, filled with the initial state and one record per year for simulation 0
    (meant for p.deterministic with n_sims=1, see central_path)."""
    rng = np.random.default_rng(p.seed)
    det = p.deterministic
    n = p.n_sims
    years = np.arange(p.start_year, p.horizon + 1)
    ahead = np.full(n, max(p.position - 1, 0), dtype=float)
    result = np.full(n, np.inf)
    a_h, b_h = _beta_params(p.h_mean, p.h_sd)
    if det:
        h = np.full(n, max(p.h_mean, 0.0))
    else:
        h = rng.beta(a_h, b_h, n) if p.h_mean > 0 else np.zeros(n)
    granted = np.full(n, float(p.existing))
    rule_fn = allowed_A if p.rule == "A" else allowed_B
    if p.rule == "A":
        def vec_allowed(pop):
            return np.floor(pop / p.threshold + 0.5)
    else:
        def vec_allowed(pop):
            return np.ceil(pop / p.threshold)
    pyears, pop = population_paths(p, rng, p.horizon + 2, z=np.zeros(n) if det else None)
    # licences already available today (population allows more than existing)
    today_allowed = rule_fn(p.p0, p.threshold)
    pending = float(max(today_allowed - p.existing, 0))
    pending = np.full(n, pending)
    if p.first_licence_now and p.existing == 0:
        pending = np.maximum(pending, 1.0)
    granted += pending
    # schedule[sim, year index] = licences becoming available that year
    schedule = np.zeros((n, len(years) + 60))
    # open slots: convocation, 30+30 days for the file, CNOPT opinion, ministerial decision
    if det:
        k0 = geometric0_median(p.pending_hazard) if p.pending_hazard else poisson_median(p.open_slot_delay_extra)
        pend_idx = np.full(n, min(k0, schedule.shape[1] - 1))
    elif p.pending_hazard:
        pend_idx = np.minimum(rng.geometric(p.pending_hazard, n) - 1, schedule.shape[1] - 1)
    else:
        pend_idx = np.minimum(rng.poisson(p.open_slot_delay_extra, n), schedule.shape[1] - 1)
    np.add.at(schedule, (np.arange(n), pend_idx), pending)
    events = {avail: ref for avail, ref in _update_events(p) if avail >= p.start_year}
    backlog = np.zeros(n)
    exc_expected = 0.0
    if trace is not None:
        trace.update(init={"today_allowed": today_allowed, "pending": float(pending[0]), "pending_year": int(years[0] + pend_idx[0]),
                           "h": float(h[0]), "ahead": float(ahead[0]), "existing": p.existing},
                     pyears=pyears, pop=pop[0].copy(), events=events, years=[])
    for i, y in enumerate(years):
        active = np.isinf(result)
        ahead_start = ahead.copy()
        leave = np.zeros(n)
        # attrition of people ahead (non-renewal, withdrawal, death, other list...)
        if p.h_mean > 0:
            leave = ahead * np.clip(h, 0, 1) if det else rng.binomial(ahead.astype(int), np.clip(h, 0, 1))
            ahead = np.where(active, ahead - leave, ahead)
        new = np.zeros(n)
        if y in events:
            j = np.searchsorted(pyears, events[y])
            if j < len(pyears):
                new = np.maximum(vec_allowed(pop[:, j]) - granted, 0)
                granted += new
                if det:
                    d = np.full(n, poisson_median(p.award_delay) if p.pop_mode == "estimates" else 0)
                else:
                    d = rng.poisson(p.award_delay, n) if p.pop_mode == "estimates" else np.zeros(n, dtype=int)
                np.add.at(schedule, (np.arange(n), np.minimum(i + d, schedule.shape[1] - 1)), new)
        backlog += schedule[:, i]
        exc = np.zeros(n)
        if p.lam_exceptional > 0:
            if det:
                exc_expected += p.lam_exceptional
                exc = np.full(n, float(exc_expected >= 1))
                exc_expected -= exc[0]
            else:
                exc = rng.poisson(p.lam_exceptional, n)
            backlog += exc
            granted += exc  # art. 7: counted within the numerus clausus
        slots = np.minimum(backlog, p.max_per_year)
        # allocate slots in list order (central path: expected number ahead rounded to whole people)
        ahead_n = np.floor(ahead + 0.5) if det else ahead
        got = active & (slots > ahead_n)
        result[got] = y
        served = np.where(active, np.minimum(slots, ahead_n + 1), slots)
        if trace is not None:
            trace["years"].append({
                "year": int(y), "ref": events.get(int(y)), "new": float(new[0]), "granted": float(granted[0]),
                "exc": float(exc[0]), "available": float(backlog[0]), "slots": float(slots[0]),
                "ahead_start": float(ahead_start[0]), "leave": float(leave[0]) if active[0] else 0.0,
                "served_ahead": float(min(slots[0], ahead_n[0])) if active[0] else 0.0, "got": bool(got[0]),
                "active": bool(active[0])})
        backlog -= served
        ahead = np.where(active & ~got, np.maximum(ahead - slots, 0), ahead)
        if trace is not None:
            trace["years"][-1]["ahead_end"] = float(ahead[0]) if active[0] and not got[0] else 0.0
    return result


def central_path(p: QueueParams) -> tuple[float, dict]:
    """Deterministic run of simulate_queue (same code, central values) with its year-by-year trace."""
    from dataclasses import replace
    tr: dict = {}
    res = simulate_queue(replace(p, deterministic=True, n_sims=1), trace=tr)
    return float(res[0]), tr


def simulate_positions(n_queue: int, existing: int, threshold: float, pop_by_year: dict, years: range,
                       events: list[tuple[int, int]], h_mean: float, h_sd: float, lam: float,
                       pending_delay: float, award_delay: float, growth_sd: float = 0.0,
                       n_sims: int = 400, seed: int = 0, h_by_pos: np.ndarray | None = None,
                       max_per_year: float = np.inf) -> np.ndarray:
    """Backtest engine: authorisation year of every queue position (sims × n_queue; inf = none).

    pop_by_year: official population per reference year (deterministic central values; an optional
    per-simulation multiplicative error with sd growth_sd·(years since first reference) is applied).
    max_per_year: licences actually processed per unit and year (open slots beyond it wait).
    """
    rng = np.random.default_rng(seed)
    yrs = list(years)
    res = np.full((n_sims, n_queue), np.inf)
    alive = np.ones((n_sims, n_queue), dtype=bool)
    a_h, b_h = _beta_params(max(h_mean, 1e-4), max(h_sd, 1e-4))
    hs = rng.beta(a_h, b_h, n_sims)[:, None]
    if h_by_pos is not None:
        hs = np.clip(hs / max(h_mean, 1e-4) * h_by_pos[None, :], 0, 1)
    sched = np.zeros((n_sims, len(yrs) + 30))
    ref0 = min(pop_by_year)
    err = rng.normal(0, growth_sd, n_sims)
    granted = np.full(n_sims, float(existing))
    allowed0 = math.floor(pop_by_year[ref0] / threshold + 0.5)
    pend = max(allowed0 - existing, 0)
    granted += pend
    idx = np.minimum(rng.poisson(pending_delay, n_sims), sched.shape[1] - 1)
    np.add.at(sched, (np.arange(n_sims), idx), pend)
    for avail, ref in events:
        if ref not in pop_by_year or avail < yrs[0]:
            continue
        pop = pop_by_year[ref] * np.exp(err * (ref - ref0))
        new = np.maximum(np.floor(pop / threshold + 0.5) - granted, 0)
        granted += new
        idx = np.minimum(avail - yrs[0] + rng.poisson(award_delay, n_sims), sched.shape[1] - 1)
        np.add.at(sched, (np.arange(n_sims), idx), new)
    backlog = np.zeros(n_sims)
    for i, y in enumerate(yrs):
        waiting = alive & np.isinf(res)
        alive &= ~(waiting & (rng.random((n_sims, n_queue)) < hs))
        backlog += sched[:, i] + (rng.poisson(lam, n_sims) if lam > 0 else 0)
        slots = np.minimum(backlog, max_per_year)
        waiting = alive & np.isinf(res)
        order = np.cumsum(waiting, axis=1)
        got = waiting & (order <= slots[:, None])
        res[got] = y
        backlog -= np.minimum(slots, waiting.sum(axis=1))
    res[~alive & np.isinf(res)] = -1   # left the list before being authorised
    return res


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
