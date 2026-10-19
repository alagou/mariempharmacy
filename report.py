"""PDF report (reportlab). Written locally to reports/ – never uploaded."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def _fonts() -> tuple[str, str]:
    for reg, bold in [(r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
                      ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]:
        if Path(reg).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("Body", reg))
            pdfmetrics.registerFont(TTFont("Body-Bold", bold))
            return "Body", "Body-Bold"
    return "Helvetica", "Helvetica-Bold"


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_pdf(data: dict, scenarios: list[dict], scen_table: pd.DataFrame, recommendation: str,
              settings: dict, metrics: dict, attrition: dict, backtests: dict, nearby: pd.DataFrame | None = None) -> Path:
    REPORTS.mkdir(exist_ok=True)
    font, bold = _fonts()
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName=font, fontSize=9, leading=12, alignment=TA_LEFT)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5)
    h1 = ParagraphStyle("h1", parent=body, fontName=bold, fontSize=16, leading=20, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=body, fontName=bold, fontSize=12, leading=15, spaceBefore=10, spaceAfter=4)
    warn = ParagraphStyle("w", parent=body, textColor=colors.HexColor("#8a1c1c"))
    prof = data["profile"]
    now = datetime.now()
    path = REPORTS / f"rapport_autorisation_officine_{now:%Y%m%d_%H%M}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm, title="Prévision d'autorisation d'officine",
                            author="Outil local")

    def tbl(df: pd.DataFrame, widths=None, fs=7.5):
        data_ = [[Paragraph(f"<b>{_esc(c)}</b>", ParagraphStyle("th", parent=small, fontName=bold, fontSize=fs)) for c in df.columns]]
        for _, r in df.iterrows():
            data_.append([Paragraph(_esc("" if pd.isna(v) else v), ParagraphStyle("td", parent=small, fontSize=fs)) for v in r])
        t = Table(data_, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0b8c4")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    el = [Paragraph("Prévision d'autorisation d'ouverture d'officine – Tunisie", h1),
          Paragraph(f"Rapport personnel généré localement le {now:%d/%m/%Y à %H:%M}. Document confidentiel : contient des données personnelles.", small),
          Spacer(1, 6),
          Paragraph("1. Situation actuelle (vérifiée)", h2),
          Paragraph(f"<b>{_esc(prof['name'])}</b> – gouvernorat de Zaghouan, délégation de Zriba, catégorie A (jour). "
                    f"Inscription du <b>26/07/2017</b> ; renouvellements : 21/01/2020, 16/01/2024 (date dite de « régularisation », "
                    f"qui ne modifie pas la date d'inscription) et 2026. Rang actuel : <b>{prof['current_rank']} sur {prof['list_size']}</b> "
                    f"({_esc(prof['rank_source'])}). Prochaine échéance de renouvellement : janvier 2028.", body),
          Spacer(1, 4)]
    m = pd.DataFrame([{"Indicateur": k, "Valeur": v} for k, v in metrics.items()])
    el += [tbl(m, [9 * cm, 15 * cm]), Paragraph("2. Hypothèses du calcul", h2)]
    el.append(tbl(pd.DataFrame([{"Paramètre": k, "Valeur": v} for k, v in settings.items()]), [9 * cm, 15 * cm]))
    el += [Paragraph("3. Scénarios « rester ou changer »", h2),
           Paragraph("Probabilités présentées par classes (< 5 %, 5–20 %, 20–50 %, 50–80 %, > 80 %) car les données ne permettent pas une précision supérieure.", small),
           tbl(scen_table.drop(columns=[c for c in ["Rang (risque)"] if c in scen_table]), fs=6.5),
           Spacer(1, 6), Paragraph("<b>Synthèse prudente :</b> " + _esc(recommendation), body)]
    for s in scenarios:
        el.append(Paragraph(f"<b>{_esc(s['code'])} – {_esc(s['title'])}</b> : {_esc(s['legal_note'])}", small))
        el.append(Paragraph("Points favorables : " + _esc(" ; ".join(s["pros"])) + " — Points de vigilance : " + _esc(" ; ".join(s["cons"])), small))
    el += [Paragraph("4. Conséquences d'un retrait de l'inscription actuelle", h2)]
    for c in data["regs"]["withdrawal_consequences"]:
        el.append(Paragraph("• " + _esc(c), warn))
    el += [PageBreak(), Paragraph("5. Questions juridiques clés", h2)]
    status = {"texte": "Établi par le texte", "interpretation": "Interprétation – à confirmer", "a_confirmer": "À confirmer officiellement"}
    for q in data["regs"]["questions"]:
        el.append(Paragraph(f"<b>{_esc(q['question'])}</b> [{status.get(q['status'], q['status'])}]", body))
        el.append(Paragraph(_esc(q["answer"]) + f" <i>Réf. : {_esc(', '.join(q['refs']))}</i>", small))
        el.append(Spacer(1, 3))
    el += [Paragraph("6. Données de file d'attente et validation", h2),
           Paragraph(f"Attrition annuelle estimée des inscrits (hors 1er/2e rang, catégorie A) : {attrition['h_central']:.1%} "
                     f"(intervalle 90 % : {attrition['h_low']:.1%} – {attrition['h_high']:.1%}), à partir de {attrition['n_at_risk']} inscrits "
                     f"du tableau 2024 suivis en 2026 ; {attrition['regularisation_share']:.0%} des non-renouvelés de 2024 ont régularisé ensuite.", small),
           Paragraph(f"Rétro-validation attrition ({backtests['attr']['train_period']} → {backtests['attr']['test_period']}) : maintien sur 2 ans prévu "
                     f"{backtests['attr']['predicted_two_year_stay']:.1%}, observé {backtests['attr']['observed_two_year_stay']:.1%} "
                     f"(écart {backtests['attr']['abs_error_points']:.1f} points). {_esc(backtests['attr']['caveat'])}", small),
           Paragraph(f"Contrôle de la formule du décret : le nombre d'officines A est égal au nombre autorisé dans "
                     f"{backtests['cap']['share_equal']:.0%} des {backtests['cap']['n_units']} unités et à ±1 près dans {backtests['cap']['share_within_1']:.0%}.", small)]
    v = backtests.get("validation")
    if v:
        qo, qn, w = v["queue"]["queue_old"], v["queue"]["queue_new"], v["waits"]
        el += [Paragraph(f"Vérification historique ({_esc(v['generated'])}, python validation.py)", h2),
               Paragraph(f"File d'attente rejouée depuis 2020 ({qn['n_people']} inscrits, {qn['n_units']} délégations) : score de Brier "
                         f"{qn['brier']:.3f} (ancien réglage {qo['brier']:.3f}) ; licences prévues {qn['pred_total']:.0f} pour "
                         f"{qn['obs_total']} retrouvées dans le registre 2026 (ancien réglage {qo['pred_total']:.0f}) ; délégations dans "
                         f"l'intervalle P10–P90 : {qn['coverage_p10_p90']:.0%} (ancien {qo['coverage_p10_p90']:.0%}).", small),
               Paragraph(f"Population : projection retenue = projection nationale INS 2026 + {v['population']['k_local']:.0%} de l'écart local "
                         f"2014-2024 (choisie sur la rétro-validation 2014→2024) ; incertitude {v['population']['growth_sd']:.2%}/an, élargie "
                         "au-delà de la dernière année de projection officielle.", small),
               Paragraph(f"Attentes observées : les convoqués actuels attendent depuis {w['convoked_wait']['median']:.0f} ans en médiane ; "
                         f"rang ≥ 4 en 2020 dans une délégation saturée : {w['saturated_rank4plus']['licensed']} licences sur "
                         f"{w['saturated_rank4plus']['n']} en 6 ans.", small),
               Paragraph(f"Convocations : {v['nowcast']['stars_in_free']} sur {v['nowcast']['stars_total']} dans des délégations à place libre "
                         f"selon la formule ; créations hors formule {v['exceptional']['lam']:.4f} / unité / an.", small)]
    if nearby is not None and len(nearby):
        el += [Paragraph("7. Délégations proches (catégorie A, nouveau dépôt)", h2), tbl(nearby, fs=6.5)]
    el += [Paragraph("8. Principales sources", h2)]
    for t in data["regs"]["texts"]:
        el.append(Paragraph(f"• {_esc(t['title'])} – {_esc(t.get('status', ''))} – {_esc(t.get('url', ''))}", small))
    el += [Paragraph("• DPM – listes d'attente et liste des officines en ligne (https://dpm.tn/dpm_pharm/asppharm/), instantané du "
                     f"{_esc(data['delegations'].snapshot_date.max())}", small),
           Paragraph("• DPM – tableaux des renouvellements 2020, 2022, 2024, 2026 ; INS – RGPH 2014 et 2024 ; citypopulation.de (source secondaire, signalée)", small),
           Spacer(1, 8),
           Paragraph("Avertissement : cet outil ne soumet ni ne retire aucune demande administrative. Les prévisions reposent sur des hypothèses "
                     "explicites et sur les données publiques disponibles ; elles ne constituent pas un avis juridique. Toute décision doit être "
                     "précédée d'une confirmation écrite de la Direction de la Pharmacie et du Médicament.", warn)]
    doc.build(el)
    return path
