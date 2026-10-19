"""Detailed, fully traced explanation of one option (2–3 page PDF, in memory).

Every number comes from the model itself: the option spec stored by analysis.all_alternatives /
build_scenarios is re-simulated with the same seed (identical Monte Carlo result) and the year-by-year
sequence is the deterministic run of the same forecasting.simulate_queue (forecasting.central_path).
"""
from __future__ import annotations

import io
import math
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

import analysis as an
import forecasting as fc
from report import _esc, _fonts

Z90 = 1.2816  # standard-normal quantile used for the P10 / P90 growth paths
KEY_YEARS = (2026, 2030, 2035, 2045, 2050, 2054, 2060, 2070)
KIND_LABEL = {"stay": "Rester sur la liste actuelle (référence)", "A_new": "Nouvelle inscription en catégorie A (jour)",
              "A_keep": "Transfert en gardant la date de 2017 (hypothèse non prévue par les textes)",
              "B": "Nouvelle inscription en catégorie B (nuit)"}
B_CASE = {"best": "hypothèse favorable : aucune officine de nuit dans la commune",
          "cautious": "hypothèse prudente : officines de nuit déjà au complet",
          "known": "nombre d'officines de nuit retenu"}


# ---------------------------------------------------------------------------
# Formatting (French)
# ---------------------------------------------------------------------------
def fint(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{int(round(x)):,}".replace(",", "\u00a0")


def fnum(x, d: int = 1) -> str:
    return f"{x:,.{d}f}".replace(",", "\u00a0").replace(".", ",")


def fpct(x, d: int = 2) -> str:
    return f"{x * 100:.{d}f}".replace(".", ",") + "\u00a0%"


def fdate(s) -> str:
    try:
        return pd.Timestamp(s).strftime("%d/%m/%Y")
    except Exception:  # noqa: BLE001
        return str(s)


def fyear(y, h: int) -> str:
    return f"après {h}" if y is None or y > h else str(int(y))


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
def _meta(spec: dict, row: dict | None) -> dict:
    """Labels shown in the table (same mapping as analysis.all_alternatives)."""
    k = spec["kind"]
    m = {"legal": k != "A_keep", "loses": k in ("A_new", "B"),
         "risk": {"stay": "Faible", "A_keep": "Non applicable"}.get(k, "Élevé"),
         "conf": "Moyenne" if k in ("stay", "A_new") else "Faible"}
    if row:
        m.update({"legal": bool(row.get("Juridiquement possible", m["legal"])), "risk": row.get("Risque juridique", m["risk"]),
                  "conf": row.get("Confiance", m["conf"])})
    return m


def _national_status(proj: pd.DataFrame) -> dict[int, str]:
    f = proj[(proj.base == "RGPH 2024") & (proj.level == "national")].set_index("year").kind.to_dict()
    out = {}
    for y in range(fc.BASE_YEAR + 1, fc.DEFAULT_HORIZON + 11):
        a, b = f.get(y - 1, "projection officielle" if y - 1 == fc.BASE_YEAR else None), f.get(y)
        if b is None:
            out[y] = "prolongée"
        elif a == b == "projection officielle":
            out[y] = "officielle"
        else:
            out[y] = "interpolée"
    return out


def _unit_list(data: dict, spec: dict, unit_row) -> pd.DataFrame:
    if spec["level"] == "delegation":
        return an.current_list(data["waiting"], "A", spec["governorate"], spec["unit"])
    name = unit_row.get("commune") if isinstance(unit_row.get("commune"), str) else None
    if not name:
        return data["waiting"].iloc[0:0]
    return an.current_list(data["waiting"], "B", spec["governorate"], name)


def compute(data: dict, s: an.SimSettings, spec: dict, row: dict | None = None) -> dict:
    """All numbers of the explanation (also used by the tests)."""
    unit, g, p0 = an.unit_inputs(data, s, spec)
    qp = an.spec_params(data, s, spec)
    years = fc.simulate_queue(qp)
    summ = fc.summarize(years, s.start_year, s.horizon)
    det_year, tr = fc.central_path(qp)
    # population paths: central and P10 / P90 growth (same population_paths as the simulator)
    pyears, pops = fc.population_paths(replace(qp, n_sims=3), np.random.default_rng(0), s.horizon + 2, z=np.array([-Z90, 0.0, Z90]))
    pop = {int(y): (float(pops[0, j]), float(pops[1, j]), float(pops[2, j])) for j, y in enumerate(pyears)}
    rule, N, existing = spec["rule"], spec["threshold"], spec["existing"]
    allowed_now = fc.allowed_A(p0, N) if rule == "A" else fc.allowed_B(p0, N)
    base = max(existing, allowed_now)
    if spec["first_licence_now"] and existing == 0:
        base = max(base, 1)
    target = fc.next_threshold_pop(base, N, rule)
    reach = {}
    for k, lab in enumerate(("bas", "central", "haut")):
        y = next((yy for yy in sorted(pop) if pop[yy][k] >= target), None)
        reach[lab] = y
    lst = _unit_list(data, spec, unit)
    reg = data["profile"]["date_inscription"]
    if spec["kind"] == "stay":
        ahead = lst[lst["order"] < spec["position"]] if "order" in lst else lst.iloc[: spec["position"] - 1]
    elif spec["kind"] == "A_keep":
        ahead = lst[lst.date_inscription < reg]
    else:
        ahead = lst
    nat_status = _national_status(pd.read_csv(an.DATA / "population_projections.csv"))
    v = data.get("validation") or {}
    g_nat = s.g_nat_1424
    diff = g.central - g_nat
    return {"spec": spec, "meta": _meta(spec, row), "unit": unit, "g": g, "p0": p0, "qp": qp, "summ": summ, "years": years,
            "det_year": det_year, "trace": tr, "pop": pop, "allowed_now": allowed_now, "target": target, "reach": reach,
            "base_licences": base, "list": lst, "ahead": ahead.reset_index(drop=True), "nat_status": nat_status,
            "g_nat": g_nat, "diff": diff, "local_add": s.k_local * diff, "validation": v, "s": s, "row": row or {}}


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
class _Doc:
    def __init__(self):
        self.font, self.bold = _fonts()
        self.ttf = self.font != "Helvetica"
        self.body = ParagraphStyle("b", fontName=self.font, fontSize=8, leading=10, spaceAfter=1.5)
        self.small = ParagraphStyle("s", parent=self.body, fontSize=6.8, leading=8.2, spaceAfter=0)
        self.cell = ParagraphStyle("c", parent=self.body, fontSize=6.6, leading=7.8, spaceAfter=0)
        self.cellb = ParagraphStyle("cb", parent=self.cell, fontName=self.bold)
        self.h1 = ParagraphStyle("h1", parent=self.body, fontName=self.bold, fontSize=13, leading=16, spaceAfter=2,
                                 textColor=colors.HexColor("#123a63"))
        self.h2 = ParagraphStyle("h2", parent=self.body, fontName=self.bold, fontSize=9.5, leading=12, spaceBefore=5, spaceAfter=2,
                                 textColor=colors.HexColor("#123a63"), keepWithNext=1)
        self.lead = ParagraphStyle("l", parent=self.body, fontSize=6.8, leading=8.2, spaceAfter=0, keepWithNext=1)
        self.note = ParagraphStyle("n", parent=self.small, textColor=colors.HexColor("#555555"))
        self.el = []

    def t(self, s) -> str:
        s = str(s)
        if not self.ttf:
            for a, b in (("≤", "<="), ("≥", ">="), ("→", "->"), ("★", "*"), ("≈", "~"), ("−", "-"), ("×", "x"), ("\u00a0", " ")):
                s = s.replace(a, b)
        return s

    def p(self, html: str, style=None):
        self.el.append(Paragraph(self.t(html), style or self.body))

    def h(self, txt: str):
        self.el.append(Paragraph(self.t(_esc(txt)), self.h2))

    def table(self, rows: list[list], widths: list[float], header: bool = True, zebra: bool = True, bold_first_col=False,
              highlight: list[int] | None = None):
        data = []
        for i, r in enumerate(rows):
            data.append([Paragraph(self.t(c if isinstance(c, str) and c.startswith("<") else _esc("" if c is None else c)),
                                   self.cellb if (header and i == 0) or (bold_first_col and j == 0) else self.cell)
                         for j, c in enumerate(r)])
        t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c2cf")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
              ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5)]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3ebf5")))
        if zebra:
            for i in range(1 if header else 0, len(rows)):
                if i % 2 == 0:
                    st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f8fb")))
        for i in highlight or []:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fff2c6")))
        t.setStyle(TableStyle(st))
        self.el.append(t)


def _unit_title(c: dict) -> str:
    sp = c["spec"]
    u = sp["unit"].title() if sp["level"] == "delegation" else sp["unit"]
    return f"{u} ({sp['governorate'].title()})"


def _sources_pop(data: dict, c: dict) -> list[list]:
    sp, u = c["spec"], c["unit"]
    pop = data["population"]
    rows = [["Année", "Population", "Nature", "Source"]]
    if sp["level"] == "delegation":
        rows.append(["2024", fint(c["p0"]), "recensement (RGPH 2024)",
                     "DPM, listes d'attente arrêtées au 31/12/2025 (population RGPH 2024 appliquée par la DPM)"])
        ins = pop[(pop.level == "delegation_ins") & (pop.key.astype(str) == str(u.get("ins_code")))]
        for y in (2014, 2004):
            r = ins[ins.year == y]
            if len(r):
                rows.append([str(y), fint(r.population.iloc[0]), f"recensement (RGPH {y})", r.source.iloc[0]])
    else:
        key = f"{sp['governorate'].title()}|{sp['unit']}"
        com = pop[(pop.level == "commune") & (pop.key == key)]
        for y in (2024, 2014, 2004):
            r = com[com.year == y]
            val = r.population.iloc[0] if len(r) else u.get(f"pop_{y}")
            src = r.source.iloc[0] if len(r) else (u.get("population_note") or "communes.csv")
            if val is not None and not pd.isna(val):
                rows.append([str(y), fint(val), f"recensement (RGPH {y})", src])
    return rows


def _seq_rows(c: dict) -> tuple[list[list], list[int], int]:
    tr, sp, s = c["trace"], c["spec"], c["s"]
    h = s.horizon
    det = c["det_year"]
    last = int(det) if math.isfinite(det) else h
    ahead = c["ahead"]
    recs = [r for r in tr["years"] if r["year"] <= last]
    rule_fn = fc.allowed_A if sp["rule"] == "A" else fc.allowed_B
    pyears = list(tr["pyears"])
    served = 0
    leave_acc = 0.0
    out, hl = [], []
    hdr = ["Année", "Pop. officielle utilisée (1er janv.)", "Officines autorisées (formule)", "Licences comptées",
           "Places ouvertes / disponibles", "Convocation (qui ?)", "Désist. attendus devant (depuis la ligne préc.)",
           "Inscrits devant elle (fin d'année)"]
    keep_idx = []
    for i, r in enumerate(recs):
        ev = r["new"] > 0 or r["served_ahead"] > 0 or r["got"] or r["exc"] > 0
        if i == 0 or ev or r["year"] % 5 == 0 or r["year"] == last:
            keep_idx.append((i, ev))
    if len(keep_idx) > 24:
        keep_idx = [(i, ev) for i, ev in keep_idx if ev or i == 0 or recs[i]["year"] == last]
    keep = {i for i, _ in keep_idx}
    for i, r in enumerate(recs):
        y = r["year"]
        conv = []
        k = int(round(r["served_ahead"]))
        for _ in range(k):
            served += 1
            if served <= len(ahead):
                a = ahead.iloc[served - 1]
                conv.append(f"rang {served} actuel (inscrit le {fdate(a.date_inscription)})")
            else:
                conv.append(f"{served}e candidat devant elle")
        if r["got"]:
            conv.append("<b>ELLE – licence</b>")
        leave_acc += r["leave"] if r["active"] else 0.0
        if i not in keep:
            continue
        ref = r["ref"]
        if ref is None:
            popref, allowed = c["p0"], c["allowed_now"]
            plab = f"{fint(popref)} (RGPH 2024, liste du 31/12/2025)"
        else:
            j = pyears.index(ref)
            popref = float(tr["pop"][j])
            allowed = rule_fn(popref, sp["threshold"])
            plab = f"{fint(popref)} (au 1/1/{ref})"
        opened = r["new"] + r["exc"] + (tr["init"]["pending"] if y == tr["init"]["pending_year"] else 0)
        open_txt = f"{fint(opened)} / {fint(r['available'])}" + (" (exceptionnelle)" if r["exc"] else "")
        out.append([str(y), plab, str(int(allowed)), fint(r["granted"]), open_txt, "<br/>".join(conv) if conv else "—",
                    fnum(leave_acc, 2) if r["active"] else "—", (fnum(r["ahead_end"], 1) if not r["got"] else "0 (servie)")])
        leave_acc = 0.0
        if r["got"]:
            hl.append(len(out))
    return [hdr] + out, hl, served


def build_pdf(data: dict, s: an.SimSettings, spec: dict, row: dict | None = None, c: dict | None = None) -> bytes:
    c = c or compute(data, s, spec, row)
    d = _Doc()
    sp, sm, meta, g, u = c["spec"], c["summ"], c["meta"], c["g"], c["unit"]
    h, start = s.horizon, s.start_year
    prof = data["profile"]
    rule, N = sp["rule"], sp["threshold"]
    unit_t = _unit_title(c)
    regdate = fdate(prof["date_inscription"])
    pos, ahead_n = sp["position"], sp["position"] - 1
    q = {k: (sm[k] if sm[k] is not None else h + 1) for k in ("q10", "q50", "q90")}
    # ---- title + a. résumé ---------------------------------------------------------------------
    d.p(_esc(f"Comment est calculée l'option « {unit_t} – catégorie {rule} »"), d.h1)
    d.p(_esc(f"{KIND_LABEL[sp['kind']]}" + (f" – {B_CASE[sp['b_case']]}" if sp.get("b_case") else "")
             + f". Document généré le {datetime.now():%d/%m/%Y à %H:%M} ; mêmes données, mêmes paramètres et même graine "
             f"aléatoire que l'écran du tableau de bord d'où il a été demandé (identifiant : {sp['id']}, graine {sp['seed']})."), d.note)
    d.h("a. Résumé de l'option")
    dist = u.get("distance_km")
    legal_txt = ("Oui" if meta["legal"] else "Non – aucun texte ne permet de transférer l'ancienneté (calcul illustratif)")
    legal_ref = {"stay": f"inscription existante (arrêté du 19/12/2019, art. 7 : renouvellement ; prochaine échéance "
                         f"{fdate(prof.get('next_renewal_deadline', '2028-01-31'))})",
                 "A_new": "nouvelle demande classée à la date du dépôt (arrêté du 19/12/2019, art. 5 et 6)",
                 "A_keep": "aucune base : les art. 14-15 de l'arrêté ne visent que les transitions de 2019 et les découpages",
                 "B": "nouvelle licence obligatoire pour changer de catégorie (décret 2019-1013, art. 14) ; liste B de la commune "
                      "(arrêté, art. 2 et 5)"}[sp["kind"]]
    loses = "Non" if not meta["loses"] else f"Oui – l'inscription du {regdate} est perdue définitivement"
    pcat = "A – officine de jour" if rule == "A" else "B – officine de nuit"
    rows = [["Lieu", f"{unit_t} – {'délégation' if sp['level'] == 'delegation' else 'commune'}",
             "Distance de Zriba", f"{fnum(dist, 0)} km" if dist is not None and not pd.isna(dist) else "—"],
            ["Catégorie", pcat, "Rang retenu", f"{pos} (soit {ahead_n} inscrit(s) devant elle)"],
            ["Juridiquement possible", legal_txt, "Perte de l'ancienneté 2017", loses],
            ["Base juridique", legal_ref, "Risque juridique / confiance", f"{meta['risk']} / {meta['conf']}"],
            ["Année médiane d'autorisation", f"<b>{fyear(q['q50'], h)}</b>", "Intervalle P10 – P90",
             f"{fyear(q['q10'], h)} – {fyear(q['q90'], h)}"],
            ["Probabilité dans 5 ans (≤ " + str(start + 4) + ")", fpct(sm["p5"], 1), "Dans 10 ans (≤ " + str(start + 9) + ")",
             fpct(sm["p10"], 1)],
            [f"Probabilité d'ici {h}", fpct(sm["p_horizon"], 1), "Trajectoire centrale (section e)",
             fyear(c["det_year"] if math.isfinite(c["det_year"]) else None, h)]]
    d.table([[f"<b>{_esc(r[0])}</b>", r[1] if str(r[1]).startswith("<") else _esc(r[1]), f"<b>{_esc(r[2])}</b>",
              r[3] if str(r[3]).startswith("<") else _esc(r[3])] for r in rows], [3.3, 6.0, 3.3, 5.6], header=False)

    # ---- b. population ---------------------------------------------------------------------------
    d.h("b. Population : données, approximations et projection")
    d.table(_sources_pop(data, c), [1.1, 1.7, 3.0, 12.4])
    gsrc = ("croissance annuelle moyenne observée entre les recensements 2014 et 2024"
            if g.basis.startswith("Recensements") else g.basis)
    n14, n24 = (c["validation"].get("population", {}).get("national", {}).get("census", {}).get(k) for k in ("2014", "2024"))
    d.p(_esc(f"Point de départ : {fint(c['p0'])} habitants (RGPH 2024). Le modèle considère que ce chiffre est la population "
             f"officielle au 1er janvier 2025 (le recensement a eu lieu fin 2024), conformément à l'art. 18 du décret 2019-1013 : "
             f"seule la population communiquée par l'INS compte."))
    d.p(_esc(f"Taux de croissance local observé : {fpct(g.central)} par an ({gsrc}). Taux national sur la même période : "
             f"{fpct(c['g_nat'])} par an (somme des 24 gouvernorats : {fint(n14)} → {fint(n24)} hab.). Écart local : "
             f"{fpct(c['diff'])} ; la rétro-validation 2014→2024 montre qu'il faut n'en garder que {fpct(s.k_local, 0)} "
             f"(ces écarts s'atténuent avec le temps) : on ajoute donc {fpct(c['local_add'])} par an au taux national officiel."))
    d.p(_esc("Taux national officiel = évolution annuelle de la projection INS « Flash » 2025-2054 (base RGPH 2024, mai 2026) : "
             "2025-2030 et 2054 sont des valeurs publiées ; 2031-2053 une interpolation documentée entre les valeurs publiées ; "
             f"après 2054, le dernier taux publié est prolongé. Formule : Pop(1/1/y) = Pop(1/1/y-1) × (1 + taux national(y) + "
             f"{fpct(c['local_add'])}). Incertitude : ± {fpct(s.growth_sd)} par an d'écart de croissance (erreur mesurée en "
             f"rétro-validation), élargie de 50 % par décennie après 2054 ; P10 / P90 = ± 1,28 écart-type."))
    pop, st_ = c["pop"], c["nat_status"]
    nat = dict(zip(range(fc.BASE_YEAR + 1, fc.BASE_YEAR + 1 + len(s.nat_path)), s.nat_path))
    prow = [["Année (1er janv.)", "Taux national", "Statut national", "Taux retenu (central)", "Facteur cumulé depuis 2025",
             "Population centrale", "P10 (croiss. basse)", "P90 (croiss. haute)"]]
    for y in KEY_YEARS:
        if y not in pop or y > h:
            continue
        gn = nat.get(y, s.nat_path[-1])
        prow.append([str(y), fpct(gn), st_.get(y, "prolongée"), fpct(gn + c["local_add"]), fnum(pop[y][1] / c["p0"], 4),
                     fint(pop[y][1]), fint(pop[y][0]), fint(pop[y][2])])
    d.table(prow, [2.0, 1.8, 2.0, 2.2, 2.5, 2.4, 2.6, 2.7])
    y_ex = 2030 if 2030 in pop else max(pop)
    factors = " × ".join(f"(1 + {fpct(nat.get(y, s.nat_path[-1]) + c['local_add'])})" for y in range(fc.BASE_YEAR + 2, y_ex + 1))
    d.p(_esc(f"Exemple : Pop(1/1/{y_ex}) = {fint(c['p0'])} × {factors} = {fint(pop[y_ex][1])} habitants."), d.small)

    # ---- c. cadre juridique et seuils ------------------------------------------------------------
    d.h("c. Cadre juridique et seuils")
    if rule == "A":
        law = (f"Loi n° 73-55 (professions pharmaceutiques) et décret gouvernemental 2019-1013 : la catégorie A est fixée par délégation "
               f"(art. 3) ; {sp['unit'].title()} est en zone {int(u.zone) if pd.notna(u.get('zone')) else '—'}, soit une officine "
               f"par tranche semi-entière de N = {fint(N)} habitants (art. 4-5, annexe 1). Formule : officines autorisées = "
               f"partie entière(population / N + 0,5) ; une nouvelle officine est due dès que la population dépasse la dernière "
               f"tranche complète de 50 % de N. Les créations exceptionnelles (délégation sans officine, imada, art. 6) sont comptées "
               f"dans ce nombre (art. 7). Licences attribuées dans l'ordre de la liste (art. 20), elle-même classée selon la date "
               f"du cachet de la poste (arrêté du 19/12/2019, art. 6).")
    else:
        law = (f"Décret 2019-1013 : la catégorie B (nuit) est fixée par commune (art. 10-11) : une officine par tranche non entière "
               f"de N = {fint(N)} habitants (zone 2 présumée ; l'annexe 2 n'a pas pu être archivée – pour une commune de moins de "
               f"50 000 hab. le résultat est le même). Formule : officines B autorisées = arrondi supérieur(population / N), donc "
               f"au moins 1 dès qu'il y a des habitants. Une commune sans officine B est prioritaire (art. 6 et 12). Changer de "
               f"catégorie exige une nouvelle licence (art. 14) ; dès qu'une officine B est autorisée, la garde de nuit des "
               f"officines A cesse dans la commune (art. 17).")
    d.p(_esc(law))
    ex_src = ("liste des officines en ligne de la DPM, instantané du " + str(u.get("snapshot_date", ""))
              if rule == "A" else ("non publié par la DPM : " + B_CASE.get(sp["b_case"], "")))
    free_now = c["allowed_now"] - sp["existing"]
    thr = [["Élément", "Valeur", "Calcul / justification"],
           ["Officines existantes", fint(sp["existing"]), ex_src],
           ["Autorisées aujourd'hui", fint(c["allowed_now"]),
            (f"partie entière({fint(c['p0'])} / {fint(N)} + 0,5) = partie entière({fnum(c['p0'] / N + 0.5, 2)})" if rule == "A"
             else f"arrondi supérieur({fint(c['p0'])} / {fint(N)}) = arrondi supérieur({fnum(c['p0'] / N, 2)})")],
           ["Places libres aujourd'hui", fint(max(free_now, 0)) + (" (+1 première officine B, art. 6)" if sp["first_licence_now"] and sp["existing"] == 0 and free_now <= 0 else ""),
            f"{fint(c['allowed_now'])} − {fint(sp['existing'])}" + (" (au-dessus de la formule : places à résorber)" if free_now < 0 else "")],
           [f"Population nécessaire pour la {fint(c['base_licences'] + 1)}e officine {rule}", fint(c["target"]),
            (f"({fint(c['base_licences'])} + 0,5) × {fint(N)}" if rule == "A" else f"{fint(c['base_licences'])} × {fint(N)} + 1")
            + f" ; soit {fpct(c['target'] / c['p0'] - 1, 1)} de plus qu'aujourd'hui"]]
    rc = c["reach"]
    paths = (("croissance centrale", "central"), ("croissance haute (P90)", "haut"), ("croissance basse (P10)", "bas"))
    if len({rc[k] if rc[k] and rc[k] <= h else None for _, k in paths}) == 1 and not (rc["central"] and rc["central"] <= h):
        paths = (("croissance centrale, haute (P90) et basse (P10)", "central"),)
    for lab, k in paths:
        y = rc[k]
        thr.append([f"Année où ce seuil est atteint – {lab}",
                    fyear(y, h), (f"population au 1/1/{y} ≥ {fint(c['target'])} → révision de liste de {y} → place ouverte en "
                                  f"{y + 1}, puis convocation (30 + 30 jours, art. 9 de l'arrêté), avis CNOPT et arrêté"
                                  if y and y <= h else f"seuil non atteint avant {h + 1}, même avec la croissance haute"
                                  if len(paths) == 1 else f"seuil non atteint avant {h + 1}")])
    d.table(thr, [4.6, 2.2, 11.4])

    # ---- d. historique de la liste -------------------------------------------------------------
    d.h("d. Historique de la liste d'attente")
    upd = pd.read_csv(an.DATA / "dpm_list_updates.csv")
    d.p(_esc(f"Révisions des listes par la DPM (pages archivées) : {len(upd)} listes de {upd.list_date.min()[:4]} à "
             f"{upd.list_date.max()[:4]} – " + ", ".join(fdate(x) for x in upd.list_date) +
             ". Chacune applique la dernière estimation INS au 1er janvier (RGPH 2024 pour la liste du 31/12/2025). "
             "Le modèle suppose donc une révision par an à partir de 2026 (arrêté, art. 8 : mise à jour trimestrielle)."), d.small)
    lst, ahead = c["list"], c["ahead"]
    src_l = ("liste DPM en ligne arrêtée au 31/12/2025 (consultée le " + str(lst.snapshot_date.iloc[0]) + ")" if len(lst) and rule == "A"
             else "liste B reconstituée depuis le tableau DPM des renouvellements 2026 (rangs = ordre chronologique, à confirmer)")
    if len(ahead):
        shown = ", ".join(f"rang {i + 1} : {fdate(a.date_inscription)}" for i, a in ahead.head(12).iterrows())
        more = f" … (+{len(ahead) - 12} autres)" if len(ahead) > 12 else ""
        head_wait = start - pd.Timestamp(ahead.date_inscription.iloc[0]).year
        txt = (f"Liste actuelle : {len(lst)} inscrit(s) ({src_l}). Devant elle : {len(ahead)} personne(s), inscrites le {shown}{more}. "
               f"La première attend depuis {head_wait} ans environ.")
    else:
        txt = f"Liste actuelle : {len(lst)} inscrit(s) ({src_l}) ; personne devant elle."
    if sp["kind"] == "stay":
        rn = "; ".join(fdate(r["date"]) if len(r["date"]) > 4 else r["date"] for r in prof.get("renewals", []))
        txt += (f" Elle-même : inscrite le {regdate}, renouvelée {rn} ; rang {pos} sur {prof['list_size']} "
                f"({prof['rank_source']}).")
    elif sp["kind"] == "A_keep":
        txt += f" Rang calculé en insérant la date du {regdate} parmi les inscrits (hypothèse)."
    else:
        txt += " Une nouvelle demande déposée aujourd'hui serait classée en dernière position (arrêté, art. 6)."
    d.p(_esc(txt), d.small)
    hist = an.list_history(data["renewals"], data["waiting"], rule, sp["governorate"],
                           sp["unit"] if rule == "A" else (u.get("commune") if isinstance(u.get("commune"), str) else sp["unit"]))
    cnt = {lt: int((hist[lt] == "✔").sum()) for lt in ("renouvele_2020", "renouvele_2024", "non_renouvele_2024", "renouvele_2026",
                                                        "non_renouvele_2026") if lt in hist}
    od = pd.read_csv(an.DATA / "officine_dossiers.csv", dtype=str).fillna("")
    um = an._norm(sp["unit"])
    od = od[(od.governorate.map(an._norm) == an._norm(sp["governorate"])) & ((od.delegation.map(an._norm) == um) | (od.commune.map(an._norm) == um))]
    cre = od[od.objet == "CREATION"]
    hrow = [["Tableau officiel", "Renouvelés 2020", "Renouvelés 2024", "Non renouvelés 2024", "Renouvelés 2026", "Non renouvelés 2026",
             "Dossiers DPM 2017-2024 (suivi archivé)"],
            ["Inscrits de cette liste", *(str(cnt.get(k, "—")) for k in ("renouvele_2020", "renouvele_2024", "non_renouvele_2024",
                                                                           "renouvele_2026", "non_renouvele_2026")),
             f"{len(od)} dossier(s) dont {len(cre)} création(s)"
             + (" : " + ", ".join(f"demande {fdate(x)}" for x in cre.date_demande.head(4) if x) if len(cre) else "")]]
    d.table(hrow, [3.0, 1.8, 1.8, 2.1, 1.8, 2.1, 5.6])
    att = c["validation"].get("attrition", {})
    w = c["validation"].get("waits", {})
    d.p(_esc(f"Vitesse historique : au niveau national, les candidats convoqués en 2026 attendaient depuis "
             f"{fnum(w.get('convoked_wait', {}).get('median', float('nan')), 0)} ans en médiane ; les sorties de liste (non-renouvellement, "
             f"désistement) ont été de {fpct(att.get('h_engaged', float('nan')), 1)} par an pour les inscrits « engagés » et "
             f"{fpct(att.get('h_all', float('nan')), 1)} pour l'ensemble (tableaux 2024 → 2026). Taux utilisé ici pour les personnes "
             f"devant elle : {fpct(c['qp'].h_mean, 2)} par an"
             + (" (engagées : toutes renouvelées en 2020, 2024 et 2026)." if sp["engaged_ahead"] else ".")), d.small)

    # ---- e. séquence chronologique -------------------------------------------------------------
    d.h("e. Séquence chronologique (trajectoire centrale)")
    seq, hl, served = _seq_rows(c)
    tr = c["trace"]
    pend = tr["init"]
    intro = (f"Chaque ligne applique les règles ci-dessus avec les valeurs centrales : croissance centrale, {fpct(c['qp'].h_mean, 2)} "
             f"de désistements par an parmi les personnes devant elle (moyenne, d'où des fractions de personne), délais médians, "
             f"1 convocation au plus par {'délégation' if rule == 'A' else 'commune'} et par an (aucune délégation n'a eu deux "
             f"convocations simultanées). Révision de l'année y−1 (population au 1/1/y−1) → places ouvertes l'année y. "
             f"« Licences comptées » = officines existantes + places déjà ouvertes (la formule n'en crée que si elle donne plus).")
    if pend["pending"]:
        intro += (f" Places déjà dues aujourd'hui : {fint(pend['pending'])}, traitées à partir de {pend['pending_year']}"
                  + (f" (délai médian de première licence B : {pend['pending_year'] - start} an(s), probabilité "
                     f"{fpct(s.b_first_hazard, 1)} par an calibrée sur les listes B)" if c["qp"].pending_hazard else "") + ".")
    d.p(_esc(intro), d.lead)
    d.table(seq, [1.0, 3.3, 1.6, 1.4, 2.1, 4.7, 1.6, 2.5], highlight=hl)
    det = c["det_year"]
    if math.isfinite(det):
        end = f"Selon la trajectoire centrale, elle obtient la licence en {int(det)}."
    else:
        last = tr["years"][-1]
        end = (f"Selon la trajectoire centrale, aucune licence avant {h + 1} : en {h}, il reste en moyenne "
               f"{fnum(last['ahead_end'], 1)} inscrit(s) devant elle et la population ({fint(c['pop'][h][1]) if h in c['pop'] else '—'}) "
               f"n'a pas atteint {fint(c['target'])}.")
    med = q["q50"]
    if (math.isfinite(det) and med == int(det)) or (not math.isfinite(det) and med > h):
        rel = f"La médiane Monte Carlo ({fyear(med, h)}) est identique."
    else:
        ys = np.asarray(c["years"])
        if math.isfinite(det) and med < det:
            why = (f"{fpct(float((ys <= med).mean()), 1)} des simulations obtiennent la licence en {med} ou avant, contre 50 % "
                   f"requis pour la médiane")
        elif math.isfinite(det):
            why = f"seulement {fpct(float((ys <= det).mean()), 1)} des simulations l'obtiennent en {int(det)} ou avant"
        else:
            why = f"{fpct(float((ys <= h).mean()), 1)} des simulations l'obtiennent avant {h + 1}"
        cause = ("le délai de première licence B est très dispersé (loi géométrique) et la trajectoire centrale prend son délai "
                 "médian en années pleines" if c["qp"].pending_hazard else
                 "les incertitudes ne sont pas symétriques : une croissance plus forte, plus de désistements ou une création "
                 "exceptionnelle avancent la date, alors que la trajectoire centrale arrondit les désistements moyens à la "
                 "personne entière")
        rel = (f"La médiane Monte Carlo est {fyear(med, h)} : {why}. L'écart vient de ce que {cause}. Les chiffres de la "
               f"section a sont ceux du Monte Carlo.")
    d.p(_esc(end + " " + rel + " « Rang k actuel » = le k-ième inscrit de la liste d'aujourd'hui s'il n'a pas quitté la liste "
             "entre-temps ; sinon c'est le suivant qui est convoqué."), d.small)

    # ---- f. probabilités ------------------------------------------------------------------------
    d.h("f. Comment les probabilités sont calculées")
    v = c["validation"]
    qn = v.get("queue", {}).get("queue_new", {})
    exc = v.get("exceptional", {})
    pb = v.get("population", {})
    best = min(pb.get("table", [{}]), key=lambda r: r.get("RMSE (pt/an)", 9) if "réelle" not in r.get("Méthode", "") else 9) if pb else {}
    n = c["qp"].n_sims
    d.p(_esc(f"Le même calcul est rejoué {fint(n)} fois (simulation de Monte Carlo, graine {sp['seed']}). D'une simulation à "
             f"l'autre varient : la croissance (écart tiré selon une loi normale d'écart-type {fpct(s.growth_sd)} par an) ; le taux "
             f"de désistement (loi bêta de moyenne {fpct(c['qp'].h_mean, 2)}, écart-type {fpct(s.h_sd, 2)}) puis les désistements "
             f"eux-mêmes (tirage personne par personne) ; les créations exceptionnelles (loi de Poisson de {fnum(s.lam_exceptional, 4)} "
             f"par unité et par an : aucune convocation hors formule observée dans {exc.get('n', '—')} délégations saturées, "
             f"estimation de Jeffreys) ; le délai entre révision et attribution (Poisson de moyenne {fnum(s.award_delay, 1)} an)."))
    d.p(_esc(f"Résultat : pour chaque simulation, l'année d'obtention (ou « pas avant {h + 1} »). P10 = année atteinte par les 10 % "
             f"de simulations les plus rapides, médiane = 50 %, P90 = 90 % ; {h + 1} signifie « après {h} ». Ici : P10 "
             f"{fyear(q['q10'], h)}, médiane {fyear(q['q50'], h)}, P90 {fyear(q['q90'], h)} ; part des simulations avec licence "
             f"au plus tard en {start + 4} : {fpct(sm['p5'], 1)}, en {start + 9} : {fpct(sm['p10'], 1)}, en {h} : "
             f"{fpct(sm['p_horizon'], 1)}. Risque juridique « {meta['risk']} » : "
             + {"Faible": "l'inscription actuelle est conservée", "Élevé": "l'ancienneté de 2017 est perdue et la démarche est irréversible",
                "Non applicable": "option non prévue par les textes"}.get(meta["risk"], "")
             + f" ; confiance « {meta['conf']} » : "
             + ("liste et population officielles publiées" if meta["conf"] == "Moyenne"
                else "données incomplètes (liste reconstituée, officines B non publiées ou hypothèse juridique)") + "."))
    n_years = h - start + 1
    p_exc = 1 - math.exp(-s.lam_exceptional * n_years)
    if not math.isfinite(c["det_year"]) and sm["p_horizon"] > 0:
        d.p(_esc(f"Pourquoi une probabilité non nulle alors que la trajectoire centrale n'aboutit pas ? Parce que certaines simulations "
                 f"ont une croissance nettement plus forte ou davantage de désistements, et parce qu'une création exceptionnelle peut "
                 f"survenir : au moins une en {n_years} ans avec probabilité 1 − exp(−{fnum(s.lam_exceptional, 4)} × {n_years}) = "
                 f"{fpct(p_exc, 1)} (elle irait au premier inscrit encore présent)."
                 + (f" Ici personne n'est devant elle et aucune place ne peut s'ouvrir par la population : cette voie explique "
                    f"à elle seule les {fpct(sm['p_horizon'], 1)} obtenus en Monte Carlo (écart dû au tirage aléatoire de "
                    f"{fint(n)} simulations)." if ahead_n == 0 and c["reach"]["haut"] is None else "")), d.small)
    if qn:
        d.p(_esc(f"Fiabilité vérifiée sur le passé (python validation.py, {v.get('generated', '')}) : file d'attente rejouée depuis 2020 "
                 f"pour {fint(qn.get('n_people'))} inscrits de {qn.get('n_units')} délégations – score de Brier {fnum(qn.get('brier', 0), 3)} "
                 f"(référence sans modèle {fnum(qn.get('brier_climatology', 0), 3)}), {fpct(qn.get('coverage_p10_p90', 0), 0)} des "
                 f"délégations dans leur intervalle P10–P90, {fnum(qn.get('pred_total', 0), 0)} licences prévues pour "
                 f"{qn.get('obs_total')} retrouvées dans le registre (qui n'en retrouve qu'environ "
                 f"{fpct(v.get('processing', {}).get('registry_recall', 0), 0)}). Population 2014→2024 : erreur médiane "
                 f"{fnum(best.get('Erreur pop. 2024 médiane (%)', float('nan')), 1)} % avec la méthode retenue."), d.small)

    # ---- g. hypothèses et limites ---------------------------------------------------------------
    d.h("g. Hypothèses et limites propres à cette option")
    lim = ["Projection INS publiée jusqu'en 2030 puis en 2054 seulement : années 2031-2053 interpolées, au-delà de 2054 prolongées "
           "(incertitude élargie) ; toute date après 2030 dépend de cette approximation.",
           "La DPM est supposée continuer à réviser les listes chaque année avec la dernière estimation INS, et à convoquer un "
           "candidat à la fois par unité."]
    if rule == "B":
        lim.append("Nombre d'officines de nuit par commune non publié : " + B_CASE.get(sp["b_case"], "") + " ; les deux hypothèses "
                   "(favorable / prudente) encadrent la réalité. Population communale tirée d'une compilation des résultats INS "
                   "(citypopulation.de, source secondaire) ; liste B reconstituée.")
        if sp["first_licence_now"]:
            lim.append(f"Première officine B : délai calibré sur {v.get('b_first', {}).get('n', '—')} communes où le premier inscrit "
                       f"attend depuis {fnum(v.get('b_first', {}).get('median_wait', float('nan')), 0)} ans en médiane ; viabilité "
                       f"économique d'une officine de nuit non évaluée.")
    if sp["kind"] == "A_new":
        lim.append("Rang = dernier de la liste actuelle ; les inscriptions postérieures se placent derrière elle. Retirer l'inscription "
                   f"du {regdate} est irréversible (arrêté, art. 5 et 10).")
    if sp["kind"] == "A_keep":
        lim.append("Calcul purement illustratif : aucun texte ne permet de conserver la date de 2017 en changeant de liste.")
    if sp["kind"] == "stay":
        lim.append("Condition : renouveler l'inscription avant fin janvier des années paires (prochaine échéance "
                   f"{fdate(prof.get('next_renewal_deadline', '2028-01-31'))}), sinon radiation (arrêté, art. 7).")
    if rule == "A":
        lim.append("L'exception « une officine par commune » des délégations à plusieurs communes (art. 6) n'est pas modélisée "
                   "faute de données commune par commune sur les officines A.")
    lim.append("Prévision statistique, pas un avis juridique : toute démarche doit être confirmée par écrit auprès de la DPM.")
    for x in lim:
        d.p("• " + _esc(x), d.small)

    buf = io.BytesIO()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(d.font, 6.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(1.3 * cm, 0.8 * cm, d.t(f"Option {sp['id']} · horizon {h} · {fint(n)} simulations · données locales"))
        canvas.drawRightString(A4[0] - 1.3 * cm, 0.8 * cm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.3 * cm, rightMargin=1.3 * cm, topMargin=1.1 * cm, bottomMargin=1.2 * cm,
                            title=f"Explication – {unit_t} catégorie {rule}", author="Outil local")
    doc.build(d.el, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
