"""Prévision d'autorisation d'officine – application locale (Streamlit).

Lancement : streamlit run app.py   →   http://localhost:8501 (lié à 127.0.0.1 uniquement)
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st

import analysis as an
import forecasting as fc

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(page_title="Prévision officine – Zriba", page_icon="💊", layout="wide")

# ---------------------------------------------------------------------------
# Interface languages (French default)
# ---------------------------------------------------------------------------
T = {
    "fr": {
        "title": "Prévision d'autorisation d'officine", "lang": "Langue",
        "nav": ["Vue d'ensemble", "Carte", "Comparer les délégations", "Catégorie A vs B", "Population",
                "Historique des listes", "Simulateur rester / changer", "Réglementation", "Rapport", "Données & sources", "Vérification / Fiabilité", "Explorer toutes les options"],
        "settings": "Hypothèses", "radius": "Rayon autour de Zriba (km)", "pop_mode": "Mise à jour de la population officielle",
        "estimates": "Estimations INS à chaque révision des listes (pratique DPM 2017-2025)", "census": "Recensements seulement (hypothèse pessimiste)", "annual": "Estimations annuelles INS avec délai réglable", "gmodel": "Modèle de croissance", "official": "Projection INS 2026 + écart local (calibré)", "trend": "Tendance 2014-2024 constante",
        "growth": "Scénario de croissance", "low": "Faible", "central": "Central", "high": "Élevé",
        "attr": "Attrition annuelle des inscrits devant elle", "exc": "Créations exceptionnelles / unité / an (art. 6)",
        "lag": "Délai administratif après mise à jour (ans)", "horizon": "Horizon des prévisions", "night": "Seuil catégorie B (annexe 2)",
        "refresh": "Actualiser les données", "rank": "Rang actuel", "registered": "Inscrite le",
    },
    "en": {
        "title": "Pharmacy licence forecast", "lang": "Language",
        "nav": ["Overview", "Map", "Compare delegations", "Category A vs B", "Population",
                "Waiting-list history", "Stay / switch simulator", "Regulations", "Report", "Data & sources", "Verification / reliability", "Explore all options"],
        "settings": "Assumptions", "radius": "Radius around Zriba (km)", "pop_mode": "Official population updates",
        "estimates": "INS estimates at each list revision (DPM practice 2017-2025)", "census": "Censuses only (pessimistic)", "annual": "Annual INS estimates with adjustable lag", "gmodel": "Growth model", "official": "INS 2026 projection + local gap (calibrated)", "trend": "Constant 2014-2024 trend",
        "growth": "Growth scenario", "low": "Low", "central": "Central", "high": "High",
        "attr": "Annual attrition of people ahead", "exc": "Exceptional licences / unit / year (art. 6)",
        "lag": "Administrative lag after update (years)", "horizon": "Forecast horizon", "night": "Category B threshold (annex 2)",
        "refresh": "Refresh data", "rank": "Current rank", "registered": "Registered on",
    },
    "ar": {
        "title": "توقع الحصول على رخصة صيدلية", "lang": "اللغة",
        "nav": ["نظرة عامة", "الخريطة", "مقارنة المعتمديات", "الصنف أ مقابل ب", "السكان",
                "تاريخ قائمات الانتظار", "محاكي البقاء / التغيير", "التشريع", "التقرير", "البيانات والمصادر", "التحقق / الموثوقية", "استكشاف كل الخيارات"],
        "settings": "الفرضيات", "radius": "الشعاع حول الزريبة (كم)", "pop_mode": "تحيين عدد السكان الرسمي",
        "estimates": "تقديرات المعهد عند كل مراجعة للقائمات (ممارسة 2017-2025)", "census": "التعدادات فقط (فرضية متشائمة)", "annual": "تقديرات سنوية مع أجل قابل للتعديل", "gmodel": "نموذج النمو", "official": "إسقاطات المعهد 2026 + الفارق المحلي", "trend": "اتجاه 2014-2024 ثابت",
        "growth": "سيناريو النمو", "low": "منخفض", "central": "مركزي", "high": "مرتفع",
        "attr": "نسبة الانسحاب السنوية للمسجلين قبلها", "exc": "إحداثات استثنائية / وحدة / سنة (الفصل 6)",
        "lag": "الأجل الإداري بعد التحيين (سنوات)", "horizon": "أفق التوقعات", "night": "عتبة الصنف ب (الملحق 2)",
        "refresh": "تحيين البيانات", "rank": "الترتيب الحالي", "registered": "تاريخ التسجيل",
    },
}

lang = st.sidebar.selectbox("Langue / Language / اللغة", ["fr", "en", "ar"],
                            format_func=lambda x: {"fr": "Français", "en": "English", "ar": "العربية"}[x])
L = T[lang]
if lang == "ar":
    st.markdown("<style>.main .block-container, section[data-testid='stSidebar'] {direction: rtl; text-align: right;}</style>",
                unsafe_allow_html=True)
st.markdown("""<style>
.badge {display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.8rem;margin-right:4px}
.ok {background:#e3f4e8;color:#1d6b35}.warn {background:#fff4d6;color:#7a5a00}.bad {background:#fde2e2;color:#8a1c1c}
.small {font-size:0.85rem;color:#555}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data (cached; reloaded automatically when a data file changes)
# ---------------------------------------------------------------------------
def data_key() -> float:
    return max(p.stat().st_mtime for p in DATA.glob("*") if p.is_file())


@st.cache_data(show_spinner=False)
def get_data(key: float) -> dict:
    return an.load_data()


@st.cache_data(show_spinner=False)
def get_attrition(key: float) -> dict:
    d = get_data(key)
    return {"A": fc.attrition_estimates(d["renewals"], "A"), "B": fc.attrition_estimates(d["renewals"], "B"),
            "bt_attr": fc.backtest_attrition(d["renewals"]), "bt_cap": fc.backtest_capacity(d["delegations"])}


@st.cache_data(show_spinner="Simulation des délégations…")
def get_compare(key: float, radius: float, settings: tuple) -> pd.DataFrame:
    return an.compare_delegations(get_data(key), radius, an.SimSettings(*settings))


@st.cache_data(show_spinner="Simulation des scénarios…")
def get_scenarios(key: float, settings: tuple, target_del, target_commune, b_target, b_zriba) -> list[dict]:
    return an.build_scenarios(get_data(key), an.SimSettings(*settings), target_del, target_commune, b_target, b_zriba)


@st.cache_data(show_spinner="Calcul de toutes les alternatives (≈ 10 s, une seule fois)…")
def get_alternatives(key: float, settings: tuple) -> pd.DataFrame:
    return an.all_alternatives(get_data(key), an.SimSettings(*settings))


KEY = data_key()
D = get_data(KEY)
ATT = get_attrition(KEY)
PROFILE = D["profile"]
VAL = D.get("validation")
DEF = an.default_settings(D)

# ---------------------------------------------------------------------------
# Sidebar: navigation + assumptions + manual refresh
# ---------------------------------------------------------------------------
st.sidebar.title(L["title"])
NAV_ORDER = [0, 2, 11, 6, 1, 3, 4, 5, 10, 7, 8, 9]   # display order; page indices stay stable
page_idx = st.sidebar.radio("Navigation", NAV_ORDER, format_func=lambda i: L["nav"][i], label_visibility="collapsed")

with st.sidebar.expander(L["settings"], expanded=False):
    radius = st.slider(L["radius"], 10, 150, 100, 5)
    pop_mode = st.radio(L["pop_mode"], ["estimates", "census", "annual"], format_func=lambda x: L[x],
                        help="Historique DPM 2017-2025 : 11 listes, chacune fondée sur la dernière estimation INS au 1er janvier "
                             "(RGPH 2024 pour la liste du 31/12/2025).")
    gmodel = st.radio(L["gmodel"], ["official", "trend"], format_func=lambda x: L[x],
                      disabled=VAL is None, help="Choisi par rétro-validation 2014→2024 (page Vérification).")
    growth = st.select_slider(L["growth"], ["low", "central", "high"], value="central", format_func=lambda x: L[x])
    h_mean = st.slider(L["attr"] + " (%)", 0.0, 10.0, round(DEF.h_mean * 100, 1), 0.1,
                       help="Inscrits engagés (renouvelés en 2020 et 2024), mesuré 2024→2026"
                            + (f" : {DEF.h_mean:.1%} (IC 90 % {VAL['attrition']['h_engaged_band'][0]:.1%}–"
                               f"{VAL['attrition']['h_engaged_band'][1]:.1%}). Autres listes : × {DEF.h_other_ratio:.2f}." if VAL else ".")) / 100
    lam = st.slider(L["exc"], 0.0, 0.03, float(round(DEF.lam_exceptional, 3)), 0.001, format="%.3f",
                    help="Aucune convocation hors formule dans les unités saturées (liste du 31/12/2025) → estimation de Jeffreys ; "
                         "l'ancienne valeur 0,015 reposait sur les créations 2007-2019 (ancien régime).")
    lag = st.slider(L["lag"], 1, 4, 2, help="Utilisé seulement pour les modes « recensements » et « annuel ».")
    horizon = st.slider(L["horizon"], 2035, 2080, fc.DEFAULT_HORIZON, 5)
    night = st.radio(L["night"], [60000, 50000], format_func=lambda x: f"{x:,} hab. (zone {'2 – présumée' if x == 60000 else '1'})".replace(",", " "))

SETTINGS = an.default_settings(D, pop_mode=pop_mode, admin_lag=lag, h_mean=h_mean, lam_exceptional=lam,
                               growth_scenario=growth, horizon=horizon, night_threshold=night,
                               **({"growth_model": gmodel} if VAL else {}))
AFTER = fc.after_horizon(horizon)
STUPLE = tuple(asdict(SETTINGS).values())

with st.sidebar.expander("🔄 " + L["refresh"]):
    st.caption("Télécharge manuellement les documents officiels et un nouvel instantané des listes DPM. "
               "Les anciennes versions sont conservées (aucun écrasement, provenance enregistrée dans sources.json). "
               "Nécessite Internet ; durée ≈ 5 à 10 minutes.")
    ok = st.checkbox("Je confirme le téléchargement")
    if st.button(L["refresh"], disabled=not ok):
        import research
        with st.status("Actualisation…", expanded=True) as status:
            try:
                research.refresh_all(log=st.write)
                status.update(label="Données actualisées", state="complete")
            except Exception as exc:  # noqa: BLE001
                status.update(label=f"Échec : {exc}", state="error")
        st.cache_data.clear()

st.sidebar.caption(f"Instantané DPM : {D['delegations'].snapshot_date.max()} · données locales · aucune transmission")

# ---------------------------------------------------------------------------
# Shared computations
# ---------------------------------------------------------------------------
CAP = an.capacity_A(D["delegations"], pop_mode, SETTINGS)
HOME = CAP[(CAP.governorate == PROFILE["governorate"]) & (CAP.delegation == PROFILE["delegation"])].iloc[0]
COMM = an.communes_B(D["communes"], night, an.DEFAULT_B_EXISTING)
ZR_C = COMM[(COMM.governorate == "ZAGHOUAN") & (COMM.commune_name == "Zriba")].iloc[0]


def fmt(n) -> str:
    return "—" if n is None or (isinstance(n, float) and np.isnan(n)) else f"{int(round(n)):,}".replace(",", " ")


def badge(text: str, kind: str = "warn") -> str:
    return f"<span class='badge {kind}'>{text}</span>"


def pct(p: float) -> str:
    return f"{p:.0%}".replace("%", " %")


def year_txt(s: dict, short: bool = False) -> str:
    sm = s["summary"]
    return str(sm["q50"]) if sm["q50"] else (f"> {sm['horizon']}" if short else f"après {sm['horizon']}")


def span_txt(s: dict) -> str:
    sm = s["summary"]
    lo, hi = (str(sm[k]) if sm[k] else f"après {sm['horizon']}" for k in ("q10", "q90"))
    return lo if lo == hi else f"{lo} – {hi}"


def default_target_del(cmp: pd.DataFrame) -> str | None:
    c = cmp[cmp.delegation != PROFILE["delegation"]]
    return None if c.empty else c.sort_values(["sim_median", "position_used", "distance_km"]).iloc[0].del_code


def b_candidates() -> pd.DataFrame:
    c = COMM[(COMM.distance_km <= radius) & COMM.pop_2024.notna() & ~((COMM.governorate == "ZAGHOUAN") & (COMM.commune_name == "Zriba"))]
    return c.sort_values(["waiting_B", "distance_km"])


def synthesis(scen: list[dict], cmp: pd.DataFrame) -> None:
    """Compact top-of-page summary; same scenarios, ranking and 'faster' rule as an.recommendation."""
    a = next(s for s in scen if s["code"] == "A")
    alts = sorted((s for s in scen if s["legally_feasible"] and s["code"] != "A"), key=lambda s: s["rank_overall"])
    h = a["summary"]["horizon"]
    with st.container(border=True):
        st.markdown("**Synthèse** <span class='small'>· année médiane d'autorisation (P10 – P90), probabilité sous 10 ans, "
                    "risque juridique · confiance</span>", unsafe_allow_html=True)
        for col, s in zip(st.columns(1 + len(alts)), [a] + alts):
            tag = {"A": " (rester)", "D0": " sans officine B", "D1": " avec officine B"}.get(s["code"], "")
            col.metric(f"{s['code']} · {(s['unit'] or PROFILE['delegation']).title()} "
                       f"{'B' if 'catégorie B' in s['title'] else 'A'}{tag}", year_txt(s, short=True),
                       help=f"{s['title']} · classement global : {s['rank_overall']} · {s['legal_note']}")
            col.caption(f"{span_txt(s)} · ≤ 10 ans : {s['summary']['band10']}  \n"
                        f"Risque {s['legal_risk'].lower()} · confiance {s['confidence'].lower()}")
        faster = [s for s in alts if s["summary"]["median_or_later"] + 5 <= a["summary"]["median_or_later"]]
        txt = [f"Rester à Zriba (rang {a['position']}/{a['list_size']}) mène à une autorisation médiane {year_txt(a)} "
               f"(P10 – P90 : {span_txt(a)}, {pct(a['summary']['p_horizon'])} de chances d'ici {h}) avec un risque juridique "
               f"{a['legal_risk'].lower()}."]
        best = min(faster, key=lambda s: s["summary"]["median_or_later"]) if faster else None
        twin = next((s for s in alts if best and s is not best and s["code"][0] == best["code"][0] == "D"), None)
        if best:
            txt.append(f"L'option la plus rapide, « {best['title']} », donnerait {year_txt(best)} ({span_txt(best)}, "
                       f"{best['summary']['band10']} sous 10 ans), au prix d'un risque {best['legal_risk'].lower()} "
                       f"(perte définitive de l'inscription du {pd.Timestamp(PROFILE['date_inscription']):%d/%m/%Y}) "
                       f"et d'une confiance {best['confidence'].lower()}"
                       + (f" ; dans l'autre cas ({twin['title'].split('(')[-1].rstrip(')')}), elle recule à {year_txt(twin)} "
                          f"({pct(twin['summary']['p_horizon'])} d'ici {h})" if twin else "") + ".")
        rest = [s for s in alts if s is not best and s is not twin]
        new = cmp[(cmp.delegation != PROFILE["delegation"]) & (cmp.sim_median <= h)]
        txt.append(("Les autres options restent tardives (" + " ; ".join(
            f"{s['code']} : {year_txt(s)}, {pct(s['summary']['p_horizon'])} d'ici {h}" for s in rest) + ") et " if rest else "")
            + (f"aucune des {len(cmp) - 1} autres délégations à ≤ {radius} km n'offre une médiane avant {h + 1} pour un nouveau dépôt."
               if new.empty else f"{len(new)} des {len(cmp) - 1} autres délégations à ≤ {radius} km offrent une médiane avant "
                                 f"{h + 1} pour un nouveau dépôt (meilleure : {new.iloc[0].delegation.title()}, {int(new.iloc[0].sim_median)})."))
        if best:
            check = f"l'absence d'officine de nuit à {best['unit']} et " if best["code"][0] in "CD" else ""
            txt.append(f"**Recommandation** : conserver l'inscription de Zriba et obtenir d'abord de la DPM une confirmation écrite de "
                       f"{check}de la position sur la liste visée ; ne changer que si l'hypothèse favorable est confirmée.")
        else:
            txt.append("**Recommandation** : rester à Zriba – aucun changement n'offre un gain assez net et fiable pour justifier "
                       "la perte de l'ancienneté.")
        st.markdown(" ".join(txt))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview():
    st.title(L["title"])
    st.markdown(f"**{PROFILE['name']}** · Zaghouan › Zriba · catégorie A (jour) "
                + badge("Inscription vérifiée", "ok") + badge("Données locales", "ok"), unsafe_allow_html=True)
    cmp = get_compare(KEY, radius, STUPLE)
    scen = get_scenarios(KEY, STUPLE, default_target_del(cmp), _default_commune(), None, 1)
    synthesis(scen, cmp)
    c = st.columns(5)
    c[0].metric(L["rank"], f"{PROFILE['current_rank']} / {PROFILE['list_size']}")
    c[1].metric(L["registered"], "26/07/2017")
    c[2].metric("Population officielle Zriba", fmt(HOME.population_dpm), help="RGPH 2024 retenu par la DPM")
    c[3].metric("Officines A : existantes / autorisées", f"{int(HOME.pharmacies_A)} / {int(HOME.allowed_A)}")
    c[4].metric("Population requise pour la 4e licence", fmt(HOME.next_pop_A))

    a = next(s for s in scen if s["code"] == "A")
    st.subheader("En bref")
    yc = HOME.year_next_A_central
    st.markdown(
        f"- **Zriba, catégorie A** : la 4e officine ne sera possible que lorsque la population *officielle* atteindra "
        f"{fmt(HOME.next_pop_A)} habitants (seuil zone 9 = 10 000, règle de la tranche semi-entière). "
        f"Croissance retenue : {HOME.growth_central:.2%}/an vers 2035 ({HOME.growth_basis}) ; "
        f"seuil atteint, scénario central : **{int(yc) if pd.notna(yc) else AFTER}**"
        + (f" (croissance haute : {int(HOME.year_next_A_high)})" if pd.notna(HOME.year_next_A_high) else "")
        + ". Il faut ensuite être première de la liste : la licence revient au 1er inscrit, et 4 inscrits la précèdent.\n"
        f"- **Scénario « rester »** : période la plus probable **{a['period']}** ; autorisation sous 5 ans : {a['summary']['band5']}, "
        f"sous 10 ans : {a['summary']['band10']}.\n"
        f"- La liste de Zriba est **inchangée depuis 2020** (mêmes 5 personnes, tous renouvelés en 2020, 2024 et 2026).\n"
        f"- **Catégorie B à Zriba** : 1 inscrite depuis 2014 ; une officine de nuit existerait déjà (source non officielle) ; "
        f"une seconde exigerait plus de {fmt(night)} habitants.")
    st.info(an.recommendation(scen))
    st.warning("Ne jamais retirer l'inscription actuelle sans réponse écrite de la DPM. Conséquences d'un retrait : "
               + " ".join("• " + x for x in D["regs"]["withdrawal_consequences"]))
    with st.expander("Qualité des données et limites"):
        st.markdown(
            "- **Vérifié (officiel)** : listes d'attente catégorie A et liste des officines (DPM, arrêtées au 31/12/2025) ; tableaux des "
            "renouvellements 2020-2026 ; populations RGPH 2014 et 2024 (INS) ; décret 2019-1013 et arrêté du 19/12/2019.\n"
            "- **Reconstitué** : listes catégorie B (ordre chronologique du tableau 2026 – la DPM ne publie pas de liste B en ligne exploitable).\n"
            "- **Source secondaire** : populations communales et valeurs 2004 par délégation (citypopulation.de, compilation INS).\n"
            "- **Inconnu** : nombre d'officines B par commune ; annexe 2 (zones B) ; intentions des autres inscrits.\n"
            "- La liste du 31/12/2025 est arrêtée sur le RGPH 2024, mais les listes de 2017 à 2023 l'ont été sur les *estimations INS au "
            "1er janvier* (pages DPM archivées) : le modèle suppose donc une révision par an sur la dernière estimation INS.\n"
            f"- Au-delà de {fc.OFFICIAL_PROJECTION_END} (fin de la projection officielle INS), la croissance est prolongée au dernier "
            "taux publié et l'incertitude est élargie ; résultats indicatifs jusqu'à l'horizon choisi.")


def _default_commune():
    c = b_candidates()
    return None if c.empty else (c.iloc[0].governorate, c.iloc[0].commune_name)


def _circle(lat, lon, r_km, n=120):
    lat1, lon1 = math.radians(lat), math.radians(lon)
    d = r_km / 6371.0
    pts = []
    for i in range(n + 1):
        b = 2 * math.pi * i / n
        lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b))
        lon2 = lon1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat2))
        pts.append((math.degrees(lat2), math.degrees(lon2)))
    return zip(*pts)


def page_map():
    st.title(L["nav"][1])
    st.caption("Carte hors ligne : limites des délégations (HDX/OCHA, simplifiées) dessinées localement, sans fond de carte en ligne.")
    cmp = get_compare(KEY, radius, STUPLE)
    metrics = {"Places libres catégorie A (autorisées − existantes)": "free_A", "Inscrits catégorie A": "waiting_A",
               "Année médiane simulée (nouveau dépôt)": "sim_median", "Population 2024": "population_dpm",
               "Croissance annuelle 2014-2024": "growth_central"}
    mlab = st.selectbox("Indicateur", list(metrics))
    col = metrics[mlab]
    df = cmp.copy()
    geo = D["geo"]
    rings = {}
    for f in geo["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        rings[f["properties"]["adm3_pcode"]] = [p[0] for p in polys]
    vals = df[col].astype(float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    scale = "RdYlGn" if col in ("free_A", "growth_central") else "RdYlGn_r"
    fig = go.Figure()
    for _, r in df.iterrows():
        if r.location_quality not in ("exact", "auto", "commune") or not isinstance(r.pcodes, str):
            continue
        v = r[col]
        color = "#cccccc" if pd.isna(v) else pc.sample_colorscale(scale, [(v - vmin) / (vmax - vmin) if vmax > vmin else 0.5])[0]
        xs, ys = [], []
        for p in r.pcodes.split(";"):
            for ring in rings.get(p, []):
                xs += [pt[0] for pt in ring] + [None]
                ys += [pt[1] for pt in ring] + [None]
        hover = (f"<b>{r.delegation.title()}</b> ({r.governorate.title()})<br>Distance : {r.distance_km:.0f} km<br>"
                 f"Pop. 2024 : {fmt(r.population_dpm)} · zone {'' if pd.isna(r.zone) else int(r.zone)} (N={fmt(r.zone_threshold)})<br>"
                 f"Officines A : {int(r.pharmacies_A)} / autorisées {int(r.allowed_A)}<br>Inscrits A : {int(r.waiting_A)}<br>"
                 f"Nouveau dépôt : rang {int(r.position_used)} → {r.period}")
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=color, mode="lines", line=dict(color="#555", width=0.6),
                                 hoverinfo="text", text=hover, hoveron="fills", showlegend=False))
    cy, cx = _circle(HOME.lat, HOME.lon, radius)
    fig.add_trace(go.Scatter(x=list(cx), y=list(cy), mode="lines", line=dict(color="#1f6f8b", dash="dash"), name=f"{radius} km", hoverinfo="skip"))
    bc = COMM[(COMM.distance_km <= radius) & COMM.lat.notna()]
    fig.add_trace(go.Scatter(x=bc.lon + 0.01, y=bc.lat + 0.01, mode="markers",
                             marker=dict(symbol="diamond", size=8, color=np.where(bc.waiting_B == 0, "#7b2cbf", "#c77dff"), line=dict(width=0.5, color="#333")),
                             text=[f"Commune {r.commune_name} – inscrits B : {r.waiting_B} – pop. {fmt(r.pop_2024)}" for _, r in bc.iterrows()],
                             hoverinfo="text", name="Communes (catégorie B)"))
    fig.add_trace(go.Scatter(x=[HOME.lon], y=[HOME.lat], mode="markers+text", text=["Zriba"], textposition="top center",
                             marker=dict(symbol="star", size=16, color="#d00000"), name="Zriba"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1 / math.cos(math.radians(HOME.lat)), visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(height=720, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#eaf2f8",
                      legend=dict(orientation="h", y=-0.02))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Couleur : {mlab} (vert = plus favorable). Losanges violets foncés : communes sans aucun inscrit en catégorie B. "
               "Les nouvelles délégations sans limites propres ne sont pas coloriées (voir le tableau de comparaison).")


def page_compare():
    st.title(L["nav"][2])
    cmp = get_compare(KEY, radius, STUPLE)
    st.markdown(f"**Catégorie A – {len(cmp)} délégations à moins de {radius} km.** Pour Zriba, le rang actuel (5) est utilisé ; "
                "ailleurs, le rang d'un **nouveau dépôt** (fin de liste, nouvelle date). La colonne « rang si 2017 conservée » est "
                "purement hypothétique (aucune base juridique identifiée).")
    st.caption(f"Toutes les options A et B de toute la Tunisie, filtrables et triables : page « {L['nav'][11]} ».")
    show = cmp[["governorate", "delegation", "distance_km", "zone", "zone_threshold", "population_dpm", "pop_2014", "growth_central",
                "pharmacies_A", "allowed_A", "free_A", "next_pop_A", "year_next_A_central", "waiting_A", "position_used",
                "sim_q10", "period", "band10", "seniority_position_hyp", "location_quality"]].rename(columns={
        "governorate": "Gouvernorat", "delegation": "Délégation", "distance_km": "Distance (km)", "zone": "Zone", "zone_threshold": "Seuil N",
        "population_dpm": "Pop. 2024", "pop_2014": "Pop. 2014", "growth_central": "Croissance/an", "pharmacies_A": "Officines A",
        "allowed_A": "Autorisées", "free_A": "Places libres", "next_pop_A": "Pop. pour +1", "year_next_A_central": "Année +1 (central)",
        "waiting_A": "Inscrits", "position_used": "Rang retenu", "sim_q10": "Au plus tôt (10 %)", "period": "Période probable",
        "band10": "≤ 10 ans", "seniority_position_hyp": "Rang si 2017 conservée (hyp.)", "location_quality": "Localisation"})
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "Croissance/an": st.column_config.NumberColumn(format="percent"),
        "Pop. 2024": st.column_config.NumberColumn(format="%d"), "Pop. 2014": st.column_config.NumberColumn(format="%d"),
        "Pop. pour +1": st.column_config.NumberColumn(format="%d"), "Année +1 (central)": st.column_config.NumberColumn(format="%d"),
        "Au plus tôt (10 %)": st.column_config.NumberColumn(format="%d")})
    st.download_button("Télécharger (CSV)", show.to_csv(index=False).encode("utf-8-sig"), "comparaison_delegations.csv", "text/csv")
    st.subheader("Catégorie B – communes dans le rayon")
    cb = COMM[COMM.distance_km <= radius].sort_values("distance_km")
    st.dataframe(cb[["governorate", "commune_name", "located_on_delegation", "distance_km", "pop_2014", "pop_2024", "allowed_B",
                     "existing_B", "existing_B_source", "waiting_B", "first_B_date", "night_zone", "population_note"]].rename(columns={
        "governorate": "Gouvernorat", "commune_name": "Commune", "located_on_delegation": "Délégation (localisation)",
        "distance_km": "Distance (km)", "pop_2014": "Pop. 2014", "pop_2024": "Pop. 2024", "allowed_B": "Officines B autorisées",
        "existing_B": "Officines B existantes", "existing_B_source": "Source (existantes)", "waiting_B": "Inscrits B",
        "first_B_date": "1re inscription B", "night_zone": "Zone B", "population_note": "Population"}),
        width="stretch", hide_index=True)
    st.caption("Le nombre d'officines B par commune n'est pas publié : il doit être confirmé auprès de la DPM avant toute décision.")


def page_ab():
    st.title(L["nav"][3])
    ra, rb = D["regs"]["rules_current"]["A"], D["regs"]["rules_current"]["B"]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Catégorie A – jour")
        st.markdown(f"- Base : population de la **délégation** (art. 3)\n- Formule : {ra['formula']}\n"
                    f"- Distance minimale : {ra['distance_min_m']} m (imada : {ra['distance_min_imada_m']} m)\n- {ra['transfer']}\n"
                    + "\n".join(f"- {e['text']} *({e['ref']})*" for e in ra["exceptions"]))
        st.markdown(f"**Zriba** : zone 9, N = 10 000 → autorisées = ⌊27 849/10 000 + 0,5⌋ = **{int(HOME.allowed_A)}** ; "
                    f"existantes {int(HOME.pharmacies_A)} ; inscrits {int(HOME.waiting_A)} ; +1 à {fmt(HOME.next_pop_A)} hab.")
    with c2:
        st.subheader("Catégorie B – nuit")
        st.markdown(f"- Base : population de la **commune** (art. 10-11)\n- Formule : {rb['formula']}\n"
                    f"- Seuils : zone 1 = 50 000, zone 2 = 60 000 ({rb['zone_note']})\n- Distance minimale : {rb['distance_min_m']} m ; {rb['distance_A_B']}\n"
                    f"- {rb['night_activity']}\n" + "\n".join(f"- {e['text']} *({e['ref']})*" for e in rb["exceptions"]))
        st.markdown(f"**Commune de Zriba** : pop. {fmt(ZR_C.pop_2024)} → autorisées = ⌈{fmt(ZR_C.pop_2024)}/{fmt(night)}⌉ = **{int(ZR_C.allowed_B)}** ; "
                    f"existante(s) : 1 *(non officiel)* ; inscrits B : {int(ZR_C.waiting_B)} (depuis {ZR_C.first_B_date}).")
    st.info("Garde : ce n'est pas une catégorie de licence. Sans officine B dans la commune, les officines A assurent la garde ; "
            "elle cesse dès qu'une officine B y est autorisée (art. 15 et 17). Changer de catégorie exige une nouvelle licence (art. 14).")
    st.subheader("Listes catégorie A et B – communes et délégations proches (≤ 40 km)")
    near_d = CAP[CAP.distance_km <= 40][["delegation", "distance_km", "population_dpm", "pharmacies_A", "allowed_A", "waiting_A"]]
    near_c = COMM[COMM.distance_km <= 40][["commune_name", "located_on_delegation", "pop_2024", "allowed_B", "waiting_B", "first_B_date"]]
    c1, c2 = st.columns(2)
    c1.dataframe(near_d.sort_values("distance_km").rename(columns={"delegation": "Délégation", "distance_km": "km", "population_dpm": "Pop.",
                 "pharmacies_A": "Officines A", "allowed_A": "Autorisées A", "waiting_A": "Inscrits A"}), hide_index=True, width="stretch")
    c2.dataframe(near_c.rename(columns={"commune_name": "Commune", "located_on_delegation": "Délégation", "pop_2024": "Pop.",
                 "allowed_B": "Autorisées B", "waiting_B": "Inscrits B", "first_B_date": "1re inscription"}), hide_index=True, width="stretch")
    st.subheader("Évolution nationale du nombre d'officines (jour / nuit)")
    h = D["history"]
    hn = h[h.level == "national"]
    fig = go.Figure()
    for cat, name in [("A", "Jour (A)"), ("B", "Nuit (B)")]:
        x = hn[hn.category == cat]
        fig.add_trace(go.Scatter(x=x.year, y=x["count"], mode="lines+markers", name=name))
    cur_a = int(D["delegations"].pharmacies_A.sum())
    fig.add_trace(go.Scatter(x=[2025], y=[cur_a], mode="markers", marker=dict(size=11, symbol="diamond"), name="Jour (A) – liste DPM 2025"))
    fig.update_layout(height=380, margin=dict(t=10), yaxis_title="Officines", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption("1990-2013 : valeurs lues sur les graphiques statistiques officiels de la DPM ; 2025 : somme des officines A des listes DPM. "
               "Aucun total officiel récent n'est publié pour la catégorie B.")
    st.subheader("Historique des règles (numerus clausus)")
    st.dataframe(pd.DataFrame(D["regs"]["history_numerus_clausus"]).rename(columns={"from": "De", "to": "À", "source": "Source"}),
                 hide_index=True, width="stretch")
    g13 = h[(h.level == "governorate") & (h.name == "ZAGHOUAN")]
    st.caption(f"Gouvernorat de Zaghouan en 2013 : {int(g13[g13.category == 'A']['count'].iloc[0])} officines de jour et "
               f"{int(g13[g13.category == 'B']['count'].iloc[0])} de nuit (DPM) ; aujourd'hui "
               f"{int(CAP[CAP.governorate == 'ZAGHOUAN'].pharmacies_A.sum())} officines de jour (liste DPM).")


def page_population():
    st.title(L["nav"][4])
    opts = CAP[CAP.distance_km <= radius].sort_values("distance_km")
    sel = st.selectbox("Délégation", opts.del_code, format_func=lambda c: f"{opts.set_index('del_code').loc[c, 'delegation'].title()} "
                                                                          f"({opts.set_index('del_code').loc[c, 'distance_km']:.0f} km)")
    r = CAP[CAP.del_code == sel].iloc[0]
    g = an.growth_for_delegation(r)
    proj = fc.project_population(r.population_dpm, g, end=max(fc.PROJECTION_END, horizon))
    fig = go.Figure()
    _, gsd, gpath = an.growth_inputs(g, an.SimSettings(**{**asdict(SETTINGS), "growth_scenario": "central"}))
    if gpath is not None:
        yrs = np.arange(fc.BASE_YEAR + 1, proj.year.max() + 1)
        gp = np.asarray(gpath[:len(yrs)])
        wid = fc.growth_sd_factor(yrs)
        for lab, z, dash in [("basse (−1 σ)", -1, "dot"), ("centrale", 0, "solid"), ("haute (+1 σ)", 1, "dot")]:
            gg = gp + z * gsd * wid
            pop = r.population_dpm * np.cumprod(np.r_[1.0, 1 + gg[1:]])
            fig.add_trace(go.Scatter(x=yrs, y=pop, name=f"INS 2026 + écart local – {lab}", line=dict(dash=dash, color="#1f6f8b")))
        fig.add_vrect(x0=fc.OFFICIAL_PROJECTION_END, x1=proj.year.max(), fillcolor="#eee", opacity=0.5, line_width=0,
                      annotation_text="au-delà de la projection INS", annotation_position="top left")
    for col, dash in [("Faible", "dot"), ("Central (TCAM)", "solid"), ("Élevé", "dot"), ("Linéaire", "dash")]:
        if col in proj:
            fig.add_trace(go.Scatter(x=proj.year, y=proj[col], name=f"Tendance 2014-2024 – {col}", line=dict(dash=dash),
                                     visible=True if gpath is None else "legendonly"))
    obs = [(2004, r.pop_2004, "Recensement 2004 (source secondaire)"), (2014, r.pop_2014, "Recensement 2014 (INS)"),
           (2024, r.population_dpm, "Recensement 2024 (INS/DPM)")]
    obs = [o for o in obs if pd.notna(o[1])]
    fig.add_trace(go.Scatter(x=[o[0] for o in obs], y=[o[1] for o in obs], mode="markers", marker=dict(size=12, color="black"),
                             name="Observé", text=[o[2] for o in obs], hoverinfo="text+y"))
    if pd.notna(r.next_pop_A):
        fig.add_hline(y=r.next_pop_A, line_dash="dash", line_color="red", annotation_text=f"Seuil +1 officine A : {fmt(r.next_pop_A)}")
        fig.add_hline(y=r.next_pop_A + r.zone_threshold, line_dash="dot", line_color="#d77", annotation_text="Seuil +2")
    for c in fc.NEXT_CENSUSES:
        if c <= proj.year.max():
            fig.add_vline(x=c, line_dash="dot", line_color="#999")
    fig.update_layout(height=430, margin=dict(t=10), yaxis_title="Habitants", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption((f"Courbes bleues : projection nationale INS 2026 (base RGPH 2024) + {SETTINGS.k_local:.0%} de l'écart local observé "
                f"2014-2024 ; bande ± 1 écart-type ({gsd:.2%}/an, élargie après {fc.OFFICIAL_PROJECTION_END}). " if gpath is not None else "")
               + f"Tendance : {g.basis}. Lignes verticales : prochains recensements. Les projections sont des estimations, "
               "distinctes des valeurs observées.")
    rows = []
    for k in (1, 2, 3):
        target = r.next_pop_A + (k - 1) * r.zone_threshold
        rows.append({"Licence supplémentaire": f"+{k}", "Population requise": fmt(target),
                     **{f"{lab} – {mode}": str(fc.year_reaching(r.population_dpm, rate, target, mode=m, horizon=horizon) or f"> {horizon}")
                        for lab, rate in [("Faible", g.low), ("Central", g.central), ("Élevé", g.high)]
                        for m, mode in [("annual", "annuel"), ("census", "recensement")]}})
    st.subheader("Années de franchissement des seuils (catégorie A)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(f"Taux annuels : faible {g.low:.2%} · central {g.central:.2%} · élevé {g.high:.2%}"
                + (f" · linéaire {g.linear_slope:+.0f} hab./an" if g.linear_slope else ""))
    st.subheader("Communes de la délégation (catégorie B)")
    cc = COMM[COMM.located_on_delegation == r.delegation]
    if len(cc):
        st.dataframe(cc[["commune_name", "pop_2004", "pop_2014", "pop_2024", "allowed_B", "waiting_B", "population_note"]],
                     hide_index=True, width="stretch")
    st.subheader("Projections nationales INS")
    pj = pd.read_csv(DATA / "population_projections.csv")
    pj = pj[pj.level == "national"]
    fig = go.Figure()
    for (base, kind), x in pj.groupby(["base", "kind"]):
        fig.add_trace(go.Scatter(x=x.year, y=x.population, name=f"{base} – {kind}",
                                 mode="lines+markers" if kind == "projection officielle" else "lines",
                                 line=dict(dash="dot" if kind != "projection officielle" else "solid")))
    nat = an.val.national_census(D["population"])
    fig.add_trace(go.Scatter(x=list(nat), y=list(nat.values()), mode="markers", marker=dict(size=10, color="black"), name="Recensements"))
    fig.update_layout(height=300, margin=dict(t=10), yaxis_title="Habitants", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption("La projection base 2014 surestimait 2024 (12,34 M contre 11,97 M recensés). Projection 2026 : valeurs officielles 2025-2030 "
               "et 2054 (Flash INS, mai 2026) ; 2031-2053 reconstitué à partir du texte (≈ 0,5 %/an vers 2035, 0,03 % en 2054).")


def page_history():
    st.title(L["nav"][5])
    cat = st.radio("Catégorie", ["A", "B"], horizontal=True, format_func=lambda c: "A – jour" if c == "A" else "B – nuit (reconstituée)")
    if cat == "A":
        opts = CAP[CAP.distance_km <= radius].sort_values("distance_km")
        unit = st.selectbox("Délégation", opts.delegation, format_func=str.title)
        gov = opts[opts.delegation == unit].governorate.iloc[0]
    else:
        opts = COMM[(COMM.distance_km <= radius) & (COMM.commune.notna()) & (COMM.waiting_B > 0)].sort_values("distance_km")
        unit = st.selectbox("Commune", opts.commune, format_func=str.title)
        gov = opts[opts.commune == unit].governorate.iloc[0]
    cur = an.current_list(D["waiting"], cat, gov, unit)
    st.subheader("Liste actuelle")
    st.caption(cur.source.iloc[0] if len(cur) else "Aucun inscrit.")
    st.dataframe(cur[["rank", "nom", "prenom", "date_inscription"]].rename(columns={"rank": "Rang", "nom": "Nom", "prenom": "Prénom",
                 "date_inscription": "Inscription"}), hide_index=True, width="stretch")
    st.subheader("Présence dans les tableaux officiels successifs")
    st.dataframe(an.list_history(D["renewals"], D["waiting"], cat, gov, unit), hide_index=True, width="stretch")
    st.caption("✔ = présent dans le tableau. Les tableaux « non renouvelé » d'avril/août 2022 et de 2024/2026 signalent des inscrits "
               "en défaut de renouvellement (25 % des non-renouvelés de 2024 ont ensuite régularisé). « * » dans la liste en ligne = "
               "probablement convoqué pour une création (à confirmer).")
    st.subheader("Mouvements 2024 → 2026 dans le rayon")
    flows = an.list_flows(D["renewals"], cat)
    units = set(zip(CAP[CAP.distance_km <= radius].governorate, CAP[CAP.distance_km <= radius].delegation)) if cat == "A" else \
        set(zip(COMM[COMM.distance_km <= radius].governorate, COMM[COMM.distance_km <= radius].commune.fillna("")))
    f = flows[[(g, u) in units for g, u in zip(flows.governorate, flows.unit)]]
    st.dataframe(f.rename(columns={"governorate": "Gouvernorat", "unit": "Unité", "maintenu": "Maintenus 2026", "non_renouvele": "Non renouvelés 2026",
                                   "sortie_autre": "Autres sorties (autorisation, retrait…)", "nouveaux_2026": "Nouveaux 2026"}),
                 hide_index=True, width="stretch")
    a = ATT[cat]
    st.subheader("Taux de sortie estimés (national)")
    st.markdown(f"- Inscrits suivis (rang ≥ 3 en 2024) : **{a['n_at_risk']}** ; sorties « autres » : {a['exits_other']} ; non renouvelés : {a['non_renewed']}\n"
                f"- Part des non-renouvelés 2024 ayant régularisé ensuite : {a['regularisation_share']:.0%}\n"
                f"- Attrition annuelle estimée : **{a['h_central']:.1%}** (IC 90 % : {a['h_low']:.1%} – {a['h_high']:.1%})\n"
                f"- Premiers de liste ayant quitté la liste en 2 ans (surtout des autorisations) : {a['rank1_two_year_exit']:.0%} (n = {a['n_rank1']})")
    bt, bc = ATT["bt_attr"], ATT["bt_cap"]
    st.subheader("Rétro-validation")
    st.markdown(f"- **Attrition** : estimée sur {bt['train_period']} ({bt['n_train']} inscrits), testée sur {bt['test_period']} : maintien sur 2 ans "
                f"prévu {bt['predicted_two_year_stay']:.1%}, observé {bt['observed_two_year_stay']:.1%} (écart {bt['abs_error_points']:.1f} points). {bt['caveat']}\n"
                f"- **Formule du décret** : officines A existantes = autorisées dans {bc['share_equal']:.0%} des {bc['n_units']} unités, à ±1 près dans "
                f"{bc['share_within_1']:.0%} ({bc['units_above']} au-dessus, {bc['units_below']} en dessous). {bc['note']}")
    st.caption(f"Rétro-validations complètes (population, attrition par engagement, convocations, file d'attente 2020→2026) : "
               f"page « {L['nav'][10]} ».")


def page_simulator():
    st.title(L["nav"][6])
    cmp = get_compare(KEY, radius, STUPLE)
    others = cmp[cmp.delegation != PROFILE["delegation"]]
    c1, c2 = st.columns(2)
    with c1:
        dflt = default_target_del(cmp)
        ids = list(others.del_code)
        tgt = st.selectbox("Délégation visée (scénarios B et E)", ids, index=ids.index(dflt) if dflt in ids else 0,
                           format_func=lambda c: f"{others.set_index('del_code').loc[c, 'delegation'].title()} – "
                                                 f"{others.set_index('del_code').loc[c, 'distance_km']:.0f} km – "
                                                 f"{int(others.set_index('del_code').loc[c, 'waiting_A'])} inscrits")
        bz = st.radio("Officine B existante à Zriba (scénario C)", [1, 0], format_func=lambda x: "1 (signalée par med.tn, non officiel)" if x else "Aucune")
    with c2:
        bc = b_candidates()
        keys = [f"{g}|{n}" for g, n in zip(bc.governorate, bc.commune_name)]
        ck = st.selectbox("Commune visée (scénario D, catégorie B)", keys,
                          format_func=lambda k: f"{k.split('|')[1]} ({k.split('|')[0].title()}) – "
                                                f"{bc.set_index(['governorate', 'commune_name']).loc[tuple(k.split('|')), 'distance_km']:.0f} km – "
                                                f"{int(bc.set_index(['governorate', 'commune_name']).loc[tuple(k.split('|')), 'waiting_B'])} inscrit(s) B")
        bt = st.radio("Officine B existante dans la commune visée", ["unk", 0, 1],
                      format_func=lambda x: {"unk": "Inconnu – afficher les deux cas", 0: "Aucune (confirmé)", 1: "Une (confirmé)"}[x])
    scen = get_scenarios(KEY, STUPLE, tgt, tuple(ck.split("|")), None if bt == "unk" else int(bt), int(bz))
    st.subheader("Résultats")
    st.dataframe(an.scenario_table(scen), hide_index=True, width="stretch")
    st.caption("Probabilités par classes, car les données ne justifient pas une précision supérieure. « Après X » = aucune autorisation "
               "simulée avant l'horizon dans la majorité des cas. Classement global = probabilité à 10 ans × risque juridique (×0,5 si perte "
               "de l'ancienneté) × confiance (×0,7 si données faibles) – indicatif.")
    fig = go.Figure()
    for s in scen:
        y = np.asarray(s["years"])
        xs = np.arange(SETTINGS.start_year, horizon + 1)
        fig.add_trace(go.Scatter(x=xs, y=[(y <= x).mean() for x in xs], name=f"{s['code']} – {s['title']}",
                                 line=dict(dash="dot" if not s["legally_feasible"] else "solid")))
    fig.update_layout(height=420, yaxis_title="Probabilité cumulée d'autorisation", yaxis_tickformat=".0%", margin=dict(t=10),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("Courbes indicatives issues de la simulation ; à lire comme des ordres de grandeur, pas des probabilités précises.")
    st.subheader("Synthèse prudente")
    st.info(an.recommendation(scen))
    for s in scen:
        with st.expander(f"{s['code']} – {s['title']}  ·  risque {s['legal_risk']}  ·  confiance {s['confidence']}"):
            st.markdown(f"**Base juridique** : {s['legal_note']}")
            st.markdown("**Points favorables** : " + " ; ".join(s["pros"]))
            st.markdown("**Points de vigilance** : " + " ; ".join(s["cons"]))
            if s.get("p_earlier_than_stay") is not None:
                st.markdown(f"Probabilité d'être autorisée avant le scénario « rester » : {fc.prob_band(s['p_earlier_than_stay'])}")
    st.error("Conséquences d'un retrait de l'inscription de Zriba : " + " ".join("• " + x for x in D["regs"]["withdrawal_consequences"]))
    st.session_state["last_scen"] = scen


def page_regulations():
    st.title(L["nav"][7])
    regs = D["regs"]
    stat = {"texte": badge("Établi par le texte", "ok"), "interpretation": badge("Interprétation – à confirmer", "warn"),
            "a_confirmer": badge("À confirmer officiellement", "bad")}
    st.subheader("Questions clés sur le changement de liste")
    for q in regs["questions"]:
        with st.expander(q["question"]):
            st.markdown(stat.get(q["status"], "") + f" <span class='small'>Réf. : {', '.join(q['refs'])}</span>", unsafe_allow_html=True)
            st.write(q["answer"])
            if q.get("confirm_with"):
                st.caption("À confirmer auprès de : " + q["confirm_with"])
    st.subheader("Règles des listes d'attente (arrêté du 19/12/2019)")
    for w in regs["waiting_list_rules"]:
        st.markdown(f"- **Art. {w['article']}** – {w['text']} " + stat.get(w["status"], ""), unsafe_allow_html=True)
    st.subheader("Textes")
    for t in regs["texts"]:
        cols = st.columns([6, 2, 2])
        cols[0].markdown(f"**{t['title']}**  \n<span class='small'>{t.get('jort', '')} · {t.get('status', '')}</span>", unsafe_allow_html=True)
        if t.get("url"):
            cols[1].markdown(f"[Source en ligne]({t['url']})")
        lp = ROOT / t["local_path"] if t.get("local_path") else None
        if lp and lp.exists():
            cols[2].download_button("Copie locale", lp.read_bytes(), lp.name, key="dl_" + t["id"])
    st.subheader("Historique du numerus clausus")
    st.dataframe(pd.DataFrame(regs["history_numerus_clausus"]), hide_index=True, width="stretch")


def page_report():
    st.title(L["nav"][8])
    st.write("Génère un rapport PDF personnel, enregistré localement dans le dossier `reports/`.")
    scen = st.session_state.get("last_scen") or get_scenarios(KEY, STUPLE, default_target_del(get_compare(KEY, radius, STUPLE)),
                                                              _default_commune(), None, 1)
    st.caption("Scénarios utilisés : ceux du dernier passage dans le simulateur (sinon les choix par défaut).")
    if st.button("Générer le rapport PDF"):
        import report
        cmp = get_compare(KEY, radius, STUPLE)
        near = cmp[cmp.distance_km <= min(radius, 40)][["delegation", "distance_km", "population_dpm", "pharmacies_A", "allowed_A",
                                                        "waiting_A", "position_used", "period", "band10"]].rename(columns={
            "delegation": "Délégation", "distance_km": "km", "population_dpm": "Pop. 2024", "pharmacies_A": "Officines A",
            "allowed_A": "Autorisées", "waiting_A": "Inscrits", "position_used": "Rang", "period": "Période", "band10": "≤ 10 ans"})
        near["km"] = near["km"].round(0)
        metrics = {"Population officielle de Zriba (RGPH 2024)": fmt(HOME.population_dpm), "Zone / seuil": "9 / 10 000 habitants",
                   "Officines A existantes / autorisées": f"{int(HOME.pharmacies_A)} / {int(HOME.allowed_A)}",
                   "Population requise pour la 4e licence": fmt(HOME.next_pop_A),
                   "Croissance retenue (vers 2035)": f"{HOME.growth_central:.2%} – {HOME.growth_basis}",
                   "Année où le seuil serait atteint (central / haut)":
                       f"{int(HOME.year_next_A_central) if pd.notna(HOME.year_next_A_central) else AFTER} / "
                       f"{int(HOME.year_next_A_high) if pd.notna(HOME.year_next_A_high) else AFTER}",
                   "Catégorie B à Zriba": f"{int(ZR_C.waiting_B)} inscrite (2014) ; 1 officine de nuit signalée (non officiel)"}
        settings = {"Mise à jour population": L[pop_mode], "Modèle de croissance": L[SETTINGS.growth_model],
                    "Scénario de croissance": L[growth],
                    "Attrition annuelle (inscrits devant elle / autres listes)": f"{h_mean:.1%} / {min(h_mean * SETTINGS.h_other_ratio, 0.5):.1%}",
                    "Créations exceptionnelles": f"{lam:.4f} / unité / an",
                    "Délai d'attribution / licences par unité et par an":
                        f"Poisson {SETTINGS.award_delay:g} an / {'illimité' if math.isinf(SETTINGS.max_per_year) else int(SETTINGS.max_per_year)}",
                    "Délai administratif (modes recensement/annuel)": f"{lag} an(s)",
                    "Horizon": f"{horizon} (projection INS officielle jusqu'en {fc.OFFICIAL_PROJECTION_END}, prolongée au-delà)",
                    "Seuil catégorie B": fmt(night), "Rayon": f"{radius} km", "Simulations": str(SETTINGS.n_sims)}
        path = report.build_pdf(D, scen, an.scenario_table(scen), an.recommendation(scen), settings, metrics, ATT["A"],
                                {"attr": ATT["bt_attr"], "cap": ATT["bt_cap"], "validation": VAL}, near)
        st.success(f"Rapport enregistré : {path.relative_to(ROOT)}")
        st.download_button("Télécharger le PDF", path.read_bytes(), path.name, "application/pdf")


def page_sources():
    st.title(L["nav"][9])
    src = pd.DataFrame(D["sources"])
    st.markdown(f"**{len(src)} documents/sources enregistrés** avec URL, date de récupération et empreinte SHA-256. "
                "Une nouvelle version d'un document est enregistrée à côté de l'ancienne, jamais par-dessus.")
    cols = [c for c in ["title", "url", "retrieved", "local_path", "publisher", "sha256"] if c in src]
    st.dataframe(src[cols], hide_index=True, width="stretch")
    st.subheader("Fichiers de données")
    rows = []
    for p in sorted(DATA.glob("*")):
        if p.is_file():
            rows.append({"Fichier": p.name, "Taille (Ko)": round(p.stat().st_size / 1024, 1),
                         "Modifié": pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Population : observé vs estimé")
    p = D["population"]
    st.dataframe(p.groupby(["level", "year", "kind", "verified"]).size().rename("lignes").reset_index(), hide_index=True, width="stretch")
    st.caption("verified = True : fichier officiel (INS/DPM). False : source secondaire, à confirmer. Les projections ne sont jamais "
               "enregistrées comme observations.")


def page_verification():
    st.title(L["nav"][10])
    if not VAL:
        st.warning("Aucun résultat de validation : exécuter `python validation.py` (hors ligne, ≈ 15 s).")
        return
    qo, qn, pb, att, now, w = VAL["queue"]["queue_old"], VAL["queue"]["queue_new"], VAL["population"], VAL["attrition"], VAL["nowcast"], VAL["waits"]
    st.caption(f"Le modèle est rejoué sur le passé avec les seules informations disponibles à l'époque, puis comparé à ce qui s'est "
               f"réellement passé. Calcul du {VAL['generated']} (`python validation.py`, données locales).")
    c = st.columns(4)
    c[0].metric("File d'attente 2020→2026 : score de Brier", f"{qn['brier']:.3f}", f"{qn['brier'] - qo['brier']:+.3f} vs ancien réglage",
                delta_color="inverse", help=f"Plus bas = meilleur. Référence « même probabilité pour tous » : {qn['brier_climatology']:.3f}.")
    c[1].metric("Licences prévues / observées (2021-2026)", f"{qn['pred_total']:.0f} / {qn['obs_total']}",
                f"ancien réglage : {qo['pred_total']:.0f}", delta_color="off",
                help=f"Observé = titulaires retrouvés dans le registre 2026 ; seuls {VAL['processing']['registry_recall']:.0%} des licences "
                     f"connues de 2016-2021 y sont retrouvées par le nom → le vrai total est probablement entre {qn['obs_total']} et "
                     f"{qn['obs_total'] / VAL['processing']['registry_recall']:.0f}.")
    c[2].metric("Unités dans l'intervalle P10–P90", pct(qn["coverage_p10_p90"]), f"ancien : {pct(qo['coverage_p10_p90'])}", delta_color="off")
    best = min(pb["table"], key=lambda r: r["RMSE (pt/an)"] if "réelle" not in r["Méthode"] else 9)
    c[3].metric("Population 2014→2024 : erreur médiane", f"{best['Erreur pop. 2024 médiane (%)']:.1f} %",
                help=f"Méthode : {best['Méthode']} ({pb['n_units']} unités).")

    st.subheader("Son attente prévue face aux attentes réellement observées")
    scen = get_scenarios(KEY, STUPLE, default_target_del(get_compare(KEY, radius, STUPLE)), _default_commune(), None, 1)
    a = next(s for s in scen if s["code"] == "A")
    reg = pd.Timestamp(PROFILE["date_inscription"]).year
    sat = w["saturated_rank4plus"]
    st.markdown(
        f"- **Prévision pour elle (rester à Zriba)** : médiane {year_txt(a)}, soit plus de "
        f"{(a['summary']['median_or_later'] - reg)} ans d'attente depuis {reg} ; {pct(a['summary']['p_horizon'])} de chances d'ici {horizon}.\n"
        f"- **Convoqués actuels** (« * », {w['convoked_wait']['n']} personnes) : inscrits en {w['convoked_reg_year']['median']:.0f} en médiane, "
        f"soit **{w['convoked_wait']['median']:.0f} ans d'attente** (P10–P90 : {w['convoked_wait']['p10']:.0f}–{w['convoked_wait']['p90']:.0f} ans).\n"
        f"- **Licences 2020-2026** retrouvées dans le registre ({w['licensed_2020_2026']['n']}) : inscrits en "
        f"{w['licensed_2020_2026']['reg_year']['median']:.0f} en médiane, rang médian {w['licensed_2020_2026']['rank_2020']['median']:.0f} en 2020.\n"
        f"- **Premiers de liste** : inscrits en {w['head_reg_year']['median']:.0f} en médiane ; à Zriba, la première attend depuis 2008.\n"
        f"- **Situation comparable** (rang ≥ 4 en 2020 dans une délégation sans place libre) : {sat['licensed']} licences sur "
        f"{sat['n']} personnes en 6 ans ({sat['licensed'] / max(sat['n'], 1):.1%}).")
    st.caption("Lecture : une attente de plus de 20 ans est la norme actuelle pour les licences de catégorie A ; l'échéance tardive de Zriba "
               "vient de la capacité saturée (3/3) et des 4 inscrits engagés qui la précèdent, pas d'une erreur de calcul.")

    with st.expander("Population : quelle méthode de projection ? (rétro-validation 2014 → 2024)", expanded=False):
        st.dataframe(pd.DataFrame(pb["table"]).round(2), hide_index=True, width="stretch")
        n = pb["national"]
        st.caption(f"Croissance nationale : {n['g_0414']:.2%}/an (2004-2014), {n['g_1424']:.2%}/an (2014-2024) ; la projection INS base 2014 "
                   f"prévoyait {n['g_ins14_proj']:.2%}/an ({fmt(n['ins14_2024'])} hab. en 2024). Retenu : projection INS 2026 + "
                   f"{pb['k_local']:.0%} de l'écart local ; incertitude {pb['growth_sd']:.2%}/an (erreur locale + erreur de la projection nationale).")
    with st.expander("Attrition des inscrits (entraînement 2020→2024, test 2024→2026)"):
        rows = [{"Groupe": k, "Inscrits suivis": v["n"], "Sorties observées": round(v["exits"]), "Sorties prévues (taux 2020-2024)":
                 round(v["predicted_exits_from_train"]), "Taux annuel mesuré": f"{v['h']:.1%} ({v['h_low']:.1%}–{v['h_high']:.1%})"}
                for k, v in att["test"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Taux 2020→2024 : {att['train']['h']:.1%}/an. Régularisation des non-renouvelés : {att['regul_2022']:.0%} (2022), "
                   f"{att['regul_2024']:.0%} (2024). Ancienne valeur unique : {att['h_old']:.1%}. Les 4 personnes devant elle sont engagées "
                   "(renouvelées en 2020, 2024 et 2026) → taux « engagés ».")
    with st.expander("Formule du décret vs convocations actuelles (« * »)"):
        st.markdown(f"- {now['stars_total']} convocations, **toutes** dans des délégations où la formule donne une place libre "
                    f"({now['stars_in_free']}) ; **aucune** dans les {now['full_units']} délégations saturées.\n"
                    f"- {now['free_units_with_star']} des {now['free_units']} délégations à place libre ont une convocation ; jamais plus "
                    "d'une par délégation → la DPM traite un candidat à la fois (1 licence / unité / an dans le modèle).\n"
                    f"- Créations hors formule : {VAL['exceptional']['lam']:.4f} / unité / an (borne haute 90 % : "
                    f"{VAL['exceptional']['lam_high90']:.4f} ; ancienne valeur {VAL['exceptional']['lam_old']}).")
        st.dataframe(pd.DataFrame(now["table"]), hide_index=True, width="stretch")
    with st.expander("File d'attente rejouée depuis 2020 : calibrage des probabilités"):
        cal = pd.DataFrame(qn["calibration"]).merge(pd.DataFrame(qo["calibration"]), on="bin", suffixes=(" (calibré)", " (ancien)"))
        st.dataframe(cal.rename(columns={"bin": "Probabilité prévue"}).round(3), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(VAL["process_grid"]).round(4), hide_index=True, width="stretch")
        pr = VAL["processing"]
        st.caption(f"Dossiers DPM archivés : révision de liste → demande {pr['list_to_request']['median']:.1f} an (médiane), → arrêté "
                   f"{pr['list_to_licence']['median']:.1f} an (P90 {pr['list_to_licence']['p90']:.1f}). Les délais plus longs obtiennent un "
                   "meilleur score car une partie des licences n'est pas retrouvée dans le registre ; délai retenu d'après les dossiers. "
                   "Limite : le nombre d'officines en 2020 n'est pas publié et a été reconstitué (2026 − licences observées).")
    with st.expander("Historique des révisions de listes et base de population (pages DPM archivées)"):
        st.dataframe(pd.read_csv(DATA / "dpm_list_updates.csv")[["list_date", "population_basis", "first_seen"]].rename(
            columns={"list_date": "Liste arrêtée au", "population_basis": "Population utilisée", "first_seen": "Vue en ligne"}),
            hide_index=True, width="stretch")
    bf = VAL["b_first"]
    st.caption(f"Catégorie B : dans {bf['n']} communes < 20 000 hab., le premier inscrit attend depuis {bf['median_wait']:.0f} ans en médiane "
               f"(P90 {bf['p90_wait']:.0f}) → la « première officine B » n'est pas immédiate. Au-delà de {fc.OFFICIAL_PROJECTION_END}, "
               "les prévisions reposent sur une prolongation (incertitude élargie de 50 % par décennie).")


PROB_COLS = ["≤ 5 ans", "≤ 10 ans", "Plus tôt que rester (prob.)"]
SORTS = {"Année médiane (la plus proche)": ("Année médiane", True), "Probabilité ≤ 10 ans": ("≤ 10 ans", False),
         "Probabilité d'ici l'horizon": ("hz", False), "Score global (classement)": ("Score global", False),
         "Gain vs rester": ("Gain vs rester (ans)", False), "Distance": ("Distance (km)", True),
         "Inscrits devant": ("Inscrits devant", True), "Places libres 2035 (proj.)": ("Places libres 2035 (proj.)", False)}


def page_alternatives():
    st.title(L["nav"][11])
    alt = get_alternatives(KEY, STUPLE).copy()
    hz = f"D'ici {horizon}"
    h1 = horizon + 1
    st.caption(f"Une ligne par option si elle changeait **aujourd'hui** : chaque délégation (catégorie A, nouveau dépôt), chaque commune "
               f"(catégorie B, sous les deux hypothèses sur les officines de nuit non publiées) et le transfert hypothétique gardant 2017. "
               f"Même modèle et mêmes définitions que la Synthèse ; la ligne « rester » (★) est identique à celle de la Synthèse. "
               f"Années : {h1} = « après {horizon} ». Lignes hors référence : 1 000 simulations chacune.")
    dmax = int(math.ceil(alt["Distance (km)"].max()))
    with st.expander("Filtres et tri", expanded=True):
        c = st.columns(4)
        dist = c[0].slider("Distance de Zriba (km)", 0, dmax, (0, min(100, dmax)), 5)
        cats = c[1].multiselect("Catégorie", sorted(alt["Catégorie"].unique()),
                                default=[x for x in sorted(alt["Catégorie"].unique()) if x != an.CAT_E])
        govs = c[2].multiselect("Gouvernorat", sorted(alt["Gouvernorat"].unique()), placeholder="Tous")
        q = c[3].text_input("Rechercher une délégation / commune", "")
        c = st.columns(4)
        ymax = c[0].slider("Année médiane au plus tard", fc.FIRST_LIST_REVISION, h1, h1)
        pmin = c[1].slider("Probabilité ≤ 10 ans au moins (%)", 0, 100, 0, 5)
        risks = c[2].multiselect("Risque juridique", ["Faible", "Élevé", "Non applicable"], default=["Faible", "Élevé", "Non applicable"])
        confs = c[3].multiselect("Confiance", ["Moyenne", "Faible"], default=["Moyenne", "Faible"])
        c = st.columns(4)
        legal = c[0].toggle("Juridiquement possible uniquement", True)
        better = c[1].toggle("Seulement mieux que rester", False, help="Médiane plus proche que celle du scénario « rester ».")
        keep_ref = c[2].toggle("Toujours afficher « rester » ★", True)
        sort = c[3].selectbox("Trier par", list(SORTS))
    stay = alt[alt["Référence"]].iloc[0]
    m = alt["Distance (km)"].between(*dist) & alt["Catégorie"].isin(cats) & (alt["Année médiane"] <= ymax)
    m &= alt["≤ 10 ans"] * 100 >= pmin
    m &= alt["Risque juridique"].isin(risks) & alt["Confiance"].isin(confs)
    if govs:
        m &= alt["Gouvernorat"].isin(govs)
    if q.strip():
        m &= alt["Unité"].map(an._norm).str.contains(an._norm(q), regex=False)
    if legal:
        m &= alt["Juridiquement possible"]
    if better:
        m &= alt["Année médiane"] < stay["Année médiane"]
    if keep_ref:
        m |= alt["Référence"]
    f = alt[m].rename(columns={hz: "hz"})
    col, asc = SORTS[sort]
    f = f.sort_values([col, "Année médiane", "≤ 10 ans"], ascending=[asc, True, False], na_position="last")
    f["Unité"] = np.where(f["Référence"], "★ " + f["Unité"], f["Unité"])
    for p in PROB_COLS + ["hz"]:
        f[p] = f[p] * 100
    f = f.rename(columns={"hz": hz})
    better_n = int((alt["Juridiquement possible"] & ~alt["Référence"] & (alt["Année médiane"] < stay["Année médiane"])
                    & alt["Distance (km)"].between(*dist)).sum())
    c = st.columns(4)
    c[0].metric("Options affichées", f"{int((~f['Référence']).sum())}", help=f"Sur {len(alt) - 1} options calculées (toute la Tunisie).")
    c[1].metric("Rester à Zriba (A) – médiane", str(stay["Année médiane"]) if stay["Année médiane"] <= horizon else f"> {horizon}",
                f"{pct(stay[hz])} d'ici {horizon}", delta_color="off")
    c[2].metric("Options légales plus rapides (médiane) dans le rayon", str(better_n))
    best = f[~f["Référence"] & f["Juridiquement possible"]].head(1)
    c[3].metric("Première option selon le tri", best["Unité"].iloc[0] if len(best) else "—",
                (f"{best['Catégorie'].iloc[0].split(',')[0]} · médiane "
                 f"{best['Année médiane'].iloc[0] if best['Année médiane'].iloc[0] <= horizon else AFTER}" if len(best) else None),
                delta_color="off")
    cols = ["Unité", "Gouvernorat", "Distance (km)", "Catégorie", "Juridiquement possible", "Perd l'ancienneté 2017",
            "Année médiane", "P10", "P90", "Période", "≤ 5 ans", "≤ 10 ans", hz, "Gain vs rester (ans)", "Plus tôt que rester (prob.)",
            "Risque juridique", "Confiance", "Score global", "Rang global", "Rang", "Inscrits devant", "Inscrits",
            "Pop. 2024", "Pop. 2035 (proj.)", "Pop. 2050 (proj.)", "Seuil N", "Officines existantes", "Autorisées (2024)",
            "Places libres", "Places libres 2035 (proj.)", "Base juridique", "Type"]
    yfmt = st.column_config.NumberColumn(format="%d", help=f"{h1} = après {horizon}")
    pfmt = {p: st.column_config.ProgressColumn(p, format="%.0f%%", min_value=0, max_value=100) for p in PROB_COLS + [hz]}
    st.dataframe(f[cols], hide_index=True, width="stretch", height=560, column_config={
        "Distance (km)": st.column_config.NumberColumn(format="%.0f"), "Année médiane": yfmt, "P10": yfmt, "P90": yfmt,
        "Gain vs rester (ans)": st.column_config.NumberColumn(format="%+d", help="Calculé sur les médianes plafonnées à l'horizon"),
        "Score global": st.column_config.NumberColumn(format="%.3f", help="Probabilité ≤ 10 ans × risque × confiance (comme le simulateur)"),
        "Rang global": st.column_config.NumberColumn(format="%d"), "Pop. 2024": st.column_config.NumberColumn(format="%d"),
        "Pop. 2035 (proj.)": st.column_config.NumberColumn(format="%d"), "Pop. 2050 (proj.)": st.column_config.NumberColumn(format="%d"),
        "Seuil N": st.column_config.NumberColumn(format="%d"), **pfmt})
    st.download_button("Télécharger la sélection (CSV)", f[cols].to_csv(index=False).encode("utf-8-sig"), "alternatives_filtrees.csv", "text/csv")
    st.caption("Catégorie B : le nombre d'officines de nuit par commune n'est pas publié – « aucune officine B » donne la borne optimiste "
               "(première licence de la commune, délai calibré sur les listes B), « au complet » la borne prudente. Cliquer sur un en-tête "
               "de colonne pour trier. Toute option implique de confirmer par écrit auprès de la DPM avant de retirer l'inscription actuelle.")


PAGES = [page_overview, page_map, page_compare, page_ab, page_population, page_history, page_simulator,
         page_regulations, page_report, page_sources, page_verification, page_alternatives]
PAGES[page_idx]()
