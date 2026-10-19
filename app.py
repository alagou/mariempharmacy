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
                "Historique des listes", "Simulateur rester / changer", "Réglementation", "Rapport", "Données & sources"],
        "settings": "Hypothèses", "radius": "Rayon autour de Zriba (km)", "pop_mode": "Mise à jour de la population officielle",
        "census": "Recensements seulement (pratique DPM actuelle)", "annual": "Estimations annuelles INS",
        "growth": "Scénario de croissance", "low": "Faible", "central": "Central", "high": "Élevé",
        "attr": "Attrition annuelle des inscrits devant elle", "exc": "Créations exceptionnelles / unité / an (art. 6)",
        "lag": "Délai administratif après mise à jour (ans)", "horizon": "Horizon des prévisions", "night": "Seuil catégorie B (annexe 2)",
        "refresh": "Actualiser les données", "rank": "Rang actuel", "registered": "Inscrite le",
    },
    "en": {
        "title": "Pharmacy licence forecast", "lang": "Language",
        "nav": ["Overview", "Map", "Compare delegations", "Category A vs B", "Population",
                "Waiting-list history", "Stay / switch simulator", "Regulations", "Report", "Data & sources"],
        "settings": "Assumptions", "radius": "Radius around Zriba (km)", "pop_mode": "Official population updates",
        "census": "Censuses only (current DPM practice)", "annual": "Annual INS estimates",
        "growth": "Growth scenario", "low": "Low", "central": "Central", "high": "High",
        "attr": "Annual attrition of people ahead", "exc": "Exceptional licences / unit / year (art. 6)",
        "lag": "Administrative lag after update (years)", "horizon": "Forecast horizon", "night": "Category B threshold (annex 2)",
        "refresh": "Refresh data", "rank": "Current rank", "registered": "Registered on",
    },
    "ar": {
        "title": "توقع الحصول على رخصة صيدلية", "lang": "اللغة",
        "nav": ["نظرة عامة", "الخريطة", "مقارنة المعتمديات", "الصنف أ مقابل ب", "السكان",
                "تاريخ قائمات الانتظار", "محاكي البقاء / التغيير", "التشريع", "التقرير", "البيانات والمصادر"],
        "settings": "الفرضيات", "radius": "الشعاع حول الزريبة (كم)", "pop_mode": "تحيين عدد السكان الرسمي",
        "census": "التعدادات فقط (الممارسة الحالية)", "annual": "تقديرات سنوية للمعهد الوطني للإحصاء",
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


KEY = data_key()
D = get_data(KEY)
ATT = get_attrition(KEY)
PROFILE = D["profile"]

# ---------------------------------------------------------------------------
# Sidebar: navigation + assumptions + manual refresh
# ---------------------------------------------------------------------------
st.sidebar.title(L["title"])
page_idx = st.sidebar.radio("Navigation", range(len(L["nav"])), format_func=lambda i: L["nav"][i], label_visibility="collapsed")

with st.sidebar.expander(L["settings"], expanded=False):
    radius = st.slider(L["radius"], 10, 150, 100, 5)
    pop_mode = st.radio(L["pop_mode"], ["census", "annual"], format_func=lambda x: L[x])
    growth = st.select_slider(L["growth"], ["low", "central", "high"], value="central", format_func=lambda x: L[x])
    h_def = round(ATT["A"]["h_central"] * 100, 1)
    h_mean = st.slider(L["attr"] + " (%)", 0.0, 10.0, h_def, 0.1,
                       help=f"Estimé sur les tableaux DPM 2024→2026 : {ATT['A']['h_central']:.1%} "
                            f"(IC 90 % {ATT['A']['h_low']:.1%}–{ATT['A']['h_high']:.1%}).") / 100
    lam = st.slider(L["exc"], 0.0, 0.05, 0.015, 0.005,
                    help="≥ 83 officines d'imada ou de commune créées depuis 2007 pour 263 unités ≈ 0,017/unité/an (borne basse) ; "
                         "les critères de 2019 sont plus restrictifs.")
    lag = st.slider(L["lag"], 1, 4, 2)
    horizon = st.slider(L["horizon"], 2035, 2060, fc.DEFAULT_HORIZON)
    night = st.radio(L["night"], [60000, 50000], format_func=lambda x: f"{x:,} hab. (zone {'2 – présumée' if x == 60000 else '1'})".replace(",", " "))

SETTINGS = an.SimSettings(pop_mode=pop_mode, admin_lag=lag, h_mean=h_mean,
                          h_sd=max((ATT["A"]["h_high"] - ATT["A"]["h_low"]) / 3.3, 0.003),
                          lam_exceptional=lam, growth_scenario=growth, horizon=horizon, night_threshold=night)
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
CAP = an.capacity_A(D["delegations"], pop_mode)
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
        f"Avec la croissance observée 2014-2024 ({HOME.growth_central:.2%}/an) et des mises à jour aux recensements, "
        f"ce seuil serait atteint au recensement de **{int(yc) if pd.notna(yc) else 'après 2054'}**, "
        f"puis il faut encore être première de la liste (4 inscrits la précèdent).\n"
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
            "- La DPM indique : « Population est arrêtée selon le recensement général de la population et de l'Habitat 2024 » – "
            "d'où l'hypothèse par défaut de mises à jour aux recensements.")


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
    for col, dash in [("Faible", "dot"), ("Central (TCAM)", "solid"), ("Élevé", "dot"), ("Linéaire", "dash")]:
        if col in proj:
            fig.add_trace(go.Scatter(x=proj.year, y=proj[col], name=f"Projection – {col}", line=dict(dash=dash)))
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
    st.caption(f"Base de croissance : {g.basis}. Lignes verticales : prochains recensements. Les projections sont des estimations, "
               "distinctes des valeurs observées.")
    rows = []
    for k in (1, 2, 3):
        target = r.next_pop_A + (k - 1) * r.zone_threshold
        rows.append({"Licence supplémentaire": f"+{k}", "Population requise": fmt(target),
                     **{f"{lab} – {mode}": str(fc.year_reaching(r.population_dpm, rate, target, mode=m) or "> 2060")
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
    st.subheader("Projection nationale INS (hypothèse moyenne, base RGPH 2014)")
    npj = D["population"][D["population"].level == "national_projection"]
    st.line_chart(npj.set_index("year")["population"], height=220)


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
                   "Croissance annuelle 2014-2024": f"{HOME.growth_central:.2%}",
                   "Recensement où le seuil serait atteint (central)": str(int(HOME.year_next_A_central)) if pd.notna(HOME.year_next_A_central) else "après 2054",
                   "Catégorie B à Zriba": f"{int(ZR_C.waiting_B)} inscrite (2014) ; 1 officine de nuit signalée (non officiel)"}
        settings = {"Mise à jour population": L[pop_mode], "Scénario de croissance": L[growth], "Attrition annuelle": f"{h_mean:.1%}",
                    "Créations exceptionnelles": f"{lam:.3f} / unité / an", "Délai administratif": f"{lag} an(s)",
                    "Horizon": str(horizon), "Seuil catégorie B": fmt(night), "Rayon": f"{radius} km", "Simulations": str(SETTINGS.n_sims)}
        path = report.build_pdf(D, scen, an.scenario_table(scen), an.recommendation(scen), settings, metrics, ATT["A"],
                                {"attr": ATT["bt_attr"], "cap": ATT["bt_cap"]}, near)
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


PAGES = [page_overview, page_map, page_compare, page_ab, page_population, page_history, page_simulator,
         page_regulations, page_report, page_sources]
PAGES[page_idx]()
