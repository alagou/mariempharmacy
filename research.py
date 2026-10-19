"""Data collection from official Tunisian sources (manual use only).

Everything here requires internet access and is only run on demand
(command line `python research.py ...` or the "Actualiser les données"
button in the app). The dashboard itself never calls the network.

Provenance rules:
- every downloaded file is recorded in data/sources.json (URL, date, sha256);
- an existing file is never overwritten: if the remote content changed,
  the new version is saved next to it with a date suffix;
- every waiting-list / pharmacy scrape is stored as a dated snapshot, so
  historical snapshots are never lost.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings()

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DOCS = ROOT / "documents"
SOURCES_JSON = DATA / "sources.json"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
DPM = "https://dpm.tn"
DPM_APP = "https://dpm.tn/dpm_pharm/asppharm/"

# Official documents to archive: (relative folder, url, title)
DOCUMENTS = [
    # --- Regulations / procedures ---
    ("regulations", f"{DPM}/images/pdf/reglementation_fr.pdf", "DPM – Références des textes juridiques (pharmacie privée)"),
    ("regulations", f"{DPM}/images/pdf/lpf_lst_att.pdf", "DPM – Pièces à fournir : inscription sur la liste d'attente (FR)"),
    ("regulations", f"{DPM}/images/pdf/lpf_lst_att_ar.pdf", "DPM – Pièces à fournir : inscription sur la liste d'attente (AR)"),
    ("regulations", f"{DPM}/images/pdf/declaration_honneur.pdf", "DPM – Déclaration sur l'honneur (liste d'attente)"),
    ("regulations", f"{DPM}/images/pdf/lpf_cre_off.pdf", "DPM – Pièces à fournir : création d'une officine (FR)"),
    ("regulations", f"{DPM}/images/pdf/lpf_cre_off_ar.pdf", "DPM – Pièces à fournir : création d'une officine (AR)"),
    ("regulations", f"{DPM}/images/pdf/lpf_tra_off.pdf", "DPM – Pièces à fournir : transfert d'une officine (FR)"),
    ("regulations", f"{DPM}/images/pdf/lpf_tra_off_ar.pdf", "DPM – Pièces à fournir : transfert d'une officine (AR)"),
    ("regulations", f"{DPM}/images/pdf/formulaire.pdf", "DPM – Formulaire de demande (création/transfert/cession)"),
    ("regulations", f"{DPM}/images/pdf/t-op1-03-001-ar-000-guide-officine-21092026.pdf", "ANMPS – Guide des procédures des services de la sous-direction de la pharmacie (AR, 21/09/2026)"),
    # --- Waiting-list communiqués and renewal tables ---
    ("waiting_lists", f"{DPM}/images/pdf/communique-renouvellement-2026.pdf", "Communiqué renouvellement des inscriptions 2026 (31/12/2025)"),
    ("waiting_lists", f"{DPM}/images/pdf/communique-ren-inscription-03072026.pdf", "Communiqué renouvellement des inscriptions 2026 (03/07/2026)"),
    ("waiting_lists", f"{DPM}/images/pdf/list-pharm-ren-2026.pdf", "Tableau 1 – Pharmaciens ayant régularisé leur inscription pour 2026 (03/07/2026)"),
    ("waiting_lists", f"{DPM}/images/pdf/list-pharm-sans-ren-2026.pdf", "Tableau 2 – Pharmaciens n'ayant pas régularisé leur inscription pour 2026 (03/07/2026)"),
    ("waiting_lists", f"{DPM}/images/pdf/communique-renouvellement-2024.pdf", "Communiqué renouvellement des inscriptions 2024 (02/01/2024)"),
    ("waiting_lists", f"{DPM}/images/pdf/communique-ren-12062024.pdf", "Communiqué renouvellement des inscriptions 2024 (12/06/2024)"),
    ("waiting_lists", f"{DPM}/images/pdf/liste_ph_ren_12062024.pdf", "Tableau 1 – Pharmaciens ayant régularisé leur inscription pour 2024 (12/06/2024)"),
    ("waiting_lists", f"{DPM}/images/pdf/liste_ph_ren_sans_attest_12062024.pdf", "Tableau 2 – Renouvellement sans attestation CNOPT (12/06/2024)"),
    ("waiting_lists", f"{DPM}/images/pdf/liste_phnren_12062024.pdf", "Tableau 3 – Pharmaciens n'ayant pas régularisé leur inscription pour 2024 (12/06/2024)"),
    ("waiting_lists", f"{DPM}/images/pdf/communique_la_02012023.pdf", "Communiqué mise à jour et publication des listes d'attente (02/01/2023)"),
    ("waiting_lists", f"{DPM}/images/pdf/communique-02082022.pdf", "Communiqué renouvellement 2022 (02/08/2022)"),
    ("waiting_lists", f"{DPM}/images/pdf/liste_ph_nren-02082022.pdf", "Pharmaciens n'ayant pas renouvelé leur inscription (02/08/2022)"),
    ("waiting_lists", f"{DPM}/images/pdf/communique-13042022.pdf", "Communiqué renouvellement 2022 (13/04/2022)"),
    ("waiting_lists", f"{DPM}/images/pdf/liste_ph_nren-13042022.pdf", "Pharmaciens n'ayant pas renouvelé leur inscription (13/04/2022)"),
    ("waiting_lists", f"{DPM}/images/pdf/communique_renouvellement_incription_2022.pdf", "Communiqué renouvellement 2022 (03/01/2022)"),
    ("waiting_lists", f"{DPM}/images/pdf/communique_22092020.pdf", "Communiqué régularisation des inscriptions (22/09/2020)"),
    ("waiting_lists", f"{DPM}/images/pdf/communique-la-05-10-2021.pdf", "Communiqué choix d'inscription – nouvelles délégations (05/10/2021)"),
    ("waiting_lists", "http://193.95.84.4/webdpm1/PDF/communique-09-02-2017.pdf", "Communiqué listes d'attente nouvelles communes (09/02/2017)"),
    ("waiting_lists", "http://193.95.84.4/webdpm1/PDF/communique-2868-2017.pdf", "Communiqué n°2868 du 06/09/2017 – nouvelles communes non issues d'anciennes"),
    ("waiting_lists", "http://193.95.84.4/webdpm1/PDF/communique-2866-2017.pdf", "Communiqué n°2866 du 06/09/2017 – nouvelles délégations/communes issues d'anciennes"),
    ("waiting_lists", "http://193.95.84.4/webdpm1/PDF/communes2015-2016.pdf", "Liste des communes créées en 2015-2016"),
    ("waiting_lists", "http://193.95.84.4/webdpm1/PDF/nouvelles_communes.pdf", "Nouvelles communes"),
    ("waiting_lists", "http://193.95.84.4/webdpm1/PDF/Liste_nouvelle_commune_del.pdf", "Calendrier choix d'inscription – nouvelles délégations et communes"),
    # --- INS population (census 2004/2014/2024, estimates, projections) ---
    ("statistics", "https://www.ins.tn/sites/default/files-ftp3/files/2021-03/pop_log_menages_par_gouv_deleg%20%281%29.xlsx", "INS – RGPH 2014 : population, logements et ménages par délégation (2004-2014)"),
    ("statistics", "https://www.ins.tn/sites/default/files-ftp3/files/2021-03/pop_log_menages_par_gouv_deleg%20%282%29.xls", "INS – RGPH 2014 : population par gouvernorat et milieu (2004-2014)"),
    ("statistics", "https://www.ins.tn/sites/default/files-ftp3/files/2021-03/resultat_rgph2014_pop_log_menag_com_arron_0.xls", "INS – RGPH 2014 : population par commune et arrondissement"),
    ("statistics", "https://www.ins.tn/sites/default/files-ftp3/files/2021-03/Excel%20Projection%20de%20la%20population%20par%20gouvernorat%20final%20%20%20%20%20%20%202016-2020.docx%20%281%29-4.xlsx", "INS – Projection de la population par gouvernorat 2014-2044"),
    ("statistics", "https://www.ins.tn/sites/default/files-ftp3/files/2025-05/Population_Secteur_Age_Sexe_Dynamique.xlsx", "INS – RGPH 2024 : population par secteur, âge et sexe"),
    ("statistics", "https://www.ins.tn/sites/default/files-ftp3/files/2025-05/Zaghouane.pdf", "INS – RGPH 2024 : gouvernorat de Zaghouan (population par secteur)"),
    ("statistics", "https://ins.tn/sites/default/files-ftp3/files/publication/pdf/Bilan_D%C3%A9mographique.pdf", "INS – RGPH 2024 : bilan démographique (premiers résultats)"),
    ("statistics", "https://www.ins.tn/sites/default/files-ftp3/files/publication/pdf/Estimation%20de%20la%20population%202024_0.pdf", "INS – Estimations de la population par délégation et commune au 1er janvier 2024"),
    ("statistics", "https://www.cgdr.nat.tn/upload/files/gouvchiffres/gech2016/Zaghouan%202016.pdf", "CGDR/ODNE – Gouvernorat de Zaghouan en chiffres 2016 (données INS)"),
    # --- Statistics ---
    ("statistics", f"{DPM}/images/pharmacies/hist0.html", "DPM – Évolution du nombre de pharmacies par catégorie (Jour/Nuit)"),
    ("statistics", f"{DPM}/images/pharmacies/hist3.html", "DPM – Répartition des officines de jour par gouvernorat"),
    ("statistics", f"{DPM}/images/pharmacies/hist4.html", "DPM – Répartition des officines de nuit par gouvernorat"),
    ("statistics", f"{DPM}/images/pharmacies/hist5.html", "DPM – Inscriptions sur les listes d'attente par fonction"),
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    s.verify = False
    return s


def _today() -> str:
    return dt.date.today().isoformat()


def load_sources() -> list[dict]:
    if SOURCES_JSON.exists():
        return json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    return []


def save_sources(sources: list[dict]) -> None:
    SOURCES_JSON.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")


def register_source(entry: dict) -> None:
    """Add a provenance record (deduplicated on url + sha256)."""
    sources = load_sources()
    key = (entry.get("url"), entry.get("sha256"))
    if any((s.get("url"), s.get("sha256")) == key for s in sources):
        return
    sources.append(entry)
    save_sources(sources)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download_document(folder: str, url: str, title: str, s: requests.Session | None = None) -> dict | None:
    """Download one document without ever overwriting a different earlier version."""
    s = s or _session()
    try:
        r = s.get(url, timeout=90)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {url}: {exc}")
        return None
    content = r.content
    digest = _sha256(content)
    target_dir = DOCS / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    name = unquote(url.rstrip("/").split("/")[-1].split("?")[0]) or "index.html"
    name = re.sub(r"[^\w.\-]+", "_", name)
    path = target_dir / name
    if path.exists() and _sha256(path.read_bytes()) != digest:
        path = target_dir / f"{path.stem}__{_today()}{path.suffix}"
    if not path.exists():
        path.write_bytes(content)
    entry = {
        "url": url,
        "title": title,
        "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "retrieved": _today(),
        "sha256": digest,
        "bytes": len(content),
        "publisher": "DPM / ANMPS – Ministère de la Santé" if "dpm" in url or "193.95.84.4" in url else "",
    }
    register_source(entry)
    print(f"  ok {path.name} ({len(content)} bytes)")
    return entry


def download_all_documents() -> None:
    s = _session()
    for folder, url, title in DOCUMENTS:
        download_document(folder, url, title, s)


# ----------------------------------------------------------------------------
# DPM online waiting lists and pharmacy registry (Category A / day)
# ----------------------------------------------------------------------------
def _options(html: str) -> list[tuple[str, str]]:
    return [(c, n.strip()) for c, n in re.findall(r'<option value="(\d+)">\s*([^<]+)', html)]


def _parse_header(soup: BeautifulSoup) -> dict:
    txt = " ".join(soup.get_text(" ", strip=True).split())
    out = {}
    m = re.search(r"Zone\s*:\s*(\d+)\s*\((\d+)\)", txt)
    if m:
        out["zone"], out["zone_threshold"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"Population\s*:\s*(\d+)", txt)
    if m:
        out["population_dpm"] = int(m.group(1))
    m = re.search(r"Nombre Officines\s*:\s*(\d+)", txt)
    if m:
        out["officines_dpm"] = int(m.group(1))
    return out


def _parse_waiting_table(soup: BeautifulSoup) -> list[dict]:
    rows = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) == 5 and re.match(r"^\d{4}-\d{2}-\d{2}$", cells[4]):
            rows.append({"rank": cells[0], "nom": cells[1], "prenom": cells[2],
                         "epouse": cells[3], "date_inscription": cells[4]})
    return rows


def _parse_officines_table(soup: BeautifulSoup) -> list[dict]:
    rows = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        cells = [td.get_text(" ", strip=True) for td in tds]
        if len(cells) == 3 and cells[0] and cells[0] not in ("NOM",):
            rows.append({"nom": cells[0], "adresse": cells[1], "tel": cells[2]})
    return rows


def scrape_dpm(snapshot: str | None = None, governorates: list[str] | None = None,
               pause: float = 0.4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scrape the official day (Category A) waiting lists and pharmacy registry.

    Raw HTML is archived under documents/waiting_lists/snapshots/<date>/.
    Returns (delegations, waiting_list_rows, pharmacy_rows).
    """
    snapshot = snapshot or _today()
    s = _session()
    raw_dir = DOCS / "waiting_lists" / "snapshots" / snapshot
    raw_dir.mkdir(parents=True, exist_ok=True)
    home = s.get(DPM_APP + "listgouv_att.php", timeout=60).text
    govs = _options(home)
    if governorates:
        govs = [g for g in govs if g[0] in governorates or g[1] in governorates]
    dele_rows, wl_rows, ph_rows = [], [], []
    for gcode, gname in govs:
        html = s.post(DPM_APP + "listedelega_att.php", data={"cod_gouv": gcode, "Zone4": "ON"}, timeout=60).text
        for dcode, dname in _options(html):
            # waiting list
            r = s.post(DPM_APP + "listeatt.php", data={"cod_del": dcode, "CHERCHER2": "Continuer"}, timeout=60)
            (raw_dir / f"listeatt_{dcode}.html").write_bytes(r.content)
            soup = BeautifulSoup(r.content, "html.parser")
            head = _parse_header(soup)
            rows = _parse_waiting_table(soup)
            for i, row in enumerate(rows):
                row.update({"snapshot_date": snapshot, "category": "A", "gov_code": gcode,
                            "governorate": gname, "del_code": dcode, "delegation": dname,
                            "order": i + 1})
                wl_rows.append(row)
            # pharmacy registry
            r2 = s.post(DPM_APP + "listedespharm.php", data={"cod_del": dcode, "CHERCHER2": "Continuer"}, timeout=60)
            (raw_dir / f"listedespharm_{dcode}.html").write_bytes(r2.content)
            soup2 = BeautifulSoup(r2.content, "html.parser")
            offs = _parse_officines_table(soup2)
            m = re.search(r"Nombre\s*:\s*(\d+)", soup2.get_text(" ", strip=True))
            for o in offs:
                o.update({"snapshot_date": snapshot, "category": "A", "gov_code": gcode,
                          "governorate": gname, "del_code": dcode, "delegation": dname})
                ph_rows.append(o)
            dele_rows.append({"snapshot_date": snapshot, "gov_code": gcode, "governorate": gname,
                              "del_code": dcode, "delegation": dname, **head,
                              "officines_registry": int(m.group(1)) if m else len(offs),
                              "waiting_count": len(rows),
                              "source_url": DPM_APP + "listeatt.php"})
            print(f"  {gname:12s} {dname:28s} zone={head.get('zone')} pop={head.get('population_dpm')} "
                  f"off={head.get('officines_dpm')} wl={len(rows)}")
            time.sleep(pause)
    return pd.DataFrame(dele_rows), pd.DataFrame(wl_rows), pd.DataFrame(ph_rows)


def scrape_night_codes() -> pd.DataFrame:
    """Night (Category B) selector lists communes; record them for reference."""
    s = _session()
    home = s.get(DPM_APP + "listgouv_att.php", timeout=60).text
    out = []
    for gcode, gname in _options(home):
        html = s.post(DPM_APP + "listedelega_att.php", data={"cod_gouv": gcode, "Zone4": "OFF"}, timeout=60).text
        for c, n in _options(html):
            out.append({"gov_code": gcode, "governorate": gname, "code": c, "name": n})
    return pd.DataFrame(out)


def _append_snapshot(df: pd.DataFrame, path: Path, key_cols: list[str]) -> None:
    """Append a snapshot to a CSV, keeping all older snapshots untouched."""
    if df.empty:
        return
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        snap = df["snapshot_date"].iloc[0]
        old = old[old["snapshot_date"] != snap]  # re-running the same day replaces only that day
        df = pd.concat([old, df.astype(str)], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8")


def refresh_dpm_snapshot() -> str:
    snap = _today()
    dele, wl, ph = scrape_dpm(snap)
    _append_snapshot(dele, DATA / "dpm_delegations_snapshots.csv", ["snapshot_date", "del_code"])
    _append_snapshot(wl, DATA / "waiting_lists_snapshots.csv", ["snapshot_date", "del_code", "order"])
    _append_snapshot(ph, DATA / "pharmacies_registry_snapshots.csv", ["snapshot_date", "del_code", "nom"])
    register_source({"url": DPM_APP + "listgouv_att.php", "title": f"DPM – Listes d'attente en ligne (jour), instantané {snap}",
                     "local_path": f"documents/waiting_lists/snapshots/{snap}/", "retrieved": snap,
                     "sha256": f"snapshot-{snap}", "publisher": "DPM / ANMPS – Ministère de la Santé"})
    return snap


# ----------------------------------------------------------------------------
# Renewal tables (PDF) -> structured records, Category A and B
# ----------------------------------------------------------------------------
# file name -> (publication date, list type)
RENEWAL_PDFS = {
    "list-pharm-ren-2026.pdf": ("2026-07-03", "renouvele_2026"),
    "list-pharm-sans-ren-2026.pdf": ("2026-07-03", "non_renouvele_2026"),
    "liste_ph_ren_12062024.pdf": ("2024-06-12", "renouvele_2024"),
    "liste_ph_ren_sans_attest_12062024.pdf": ("2024-06-12", "renouvele_sans_attestation_2024"),
    "liste_phnren_12062024.pdf": ("2024-06-12", "non_renouvele_2024"),
    "liste_ph_nren-02082022.pdf": ("2022-08-02", "non_renouvele_2022_aout"),
    "liste_ph_nren-13042022.pdf": ("2022-04-13", "non_renouvele_2022_avril"),
}

_COLMAP = {
    "nom": "nom", "prenom": "prenom", "prénom": "prenom", "epouse": "epouse",
    "gouvernorat": "governorate", "délégation": "delegation", "delegation": "delegation",
    "commune": "commune", "délégation & commune": "place", "catégorie": "category",
    "date inscription": "date_inscription", "date renouvellement": "date_renouvellement",
}


def _clean(v) -> str:
    return " ".join(str(v or "").split())


def parse_renewal_pdf(path: Path, pub_date: str, list_type: str) -> pd.DataFrame:
    import pymupdf
    doc = pymupdf.open(path)
    rows, header, section_cat = [], None, None
    for page in doc:
        for tab in page.find_tables().tables:
            for raw in tab.extract():
                cells = [_clean(c) for c in raw]
                first = cells[0].lower()
                if first.startswith("liste des"):
                    section_cat = "B" if "(nuit)" in first else ("A" if "(jour)" in first else section_cat)
                    continue
                if first == "nom":
                    header = [_COLMAP.get(c.lower(), c.lower()) for c in cells]
                    continue
                if header is None or not any(cells):
                    continue
                rec = dict(zip(header, cells))
                if not rec.get("nom") and rows:  # wrapped continuation line
                    for k, v in rec.items():
                        if v:
                            rows[-1][k] = (rows[-1].get(k, "") + " " + v).strip()
                    continue
                if "category" not in rec:
                    rec["category"] = section_cat or ""
                rows.append(rec)
    df = pd.DataFrame(rows)
    if "place" in df:  # 2024 table merges delegation & commune in one column
        df["delegation"] = df.apply(lambda r: r["place"] if r["category"] == "A" else "", axis=1)
        df["commune"] = df.apply(lambda r: r["place"] if r["category"] == "B" else "", axis=1)
        df = df.drop(columns="place")
    for col in ["epouse", "delegation", "commune", "date_inscription", "date_renouvellement"]:
        if col not in df:
            df[col] = ""
    df["source_file"] = path.name
    df["publication_date"] = pub_date
    df["list_type"] = list_type
    for c in ["date_inscription", "date_renouvellement"]:
        df[c] = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce").dt.date.astype(str).replace("NaT", "")
    return df


def scrape_renewals_2020() -> pd.DataFrame:
    """2020 renewal database (published 05/06/2020) – full list via empty name search."""
    raw = DOCS / "waiting_lists" / "renouvellement_2020_liste_complete.html"
    if not raw.exists():
        r = _session().post(DPM_APP + "listpharmnom.php", data={"id": ""}, timeout=180)
        r.raise_for_status()
        raw.write_bytes(r.content)
    register_source({"url": DPM_APP + "listpharmnom.php", "title": "DPM – Pharmaciens ayant renouvelé leur inscription en 2020 (publié le 05/06/2020)",
                     "local_path": str(raw.relative_to(ROOT)).replace("\\", "/"), "retrieved": _today(),
                     "sha256": _sha256(raw.read_bytes()), "publisher": "DPM / ANMPS – Ministère de la Santé"})
    soup = BeautifulSoup(raw.read_bytes(), "html.parser")
    # Rows are not always closed (</tr> missing): scan the flat sequence of cells instead.
    cells = [td.get_text(" ", strip=True) for td in soup.find_all("td")]
    is_date = re.compile(r"^\d{4}-\d{2}-\d{2}$").match
    rows, i = [], 0
    while i + 6 < len(cells):
        c = cells[i:i + 7]
        if is_date(c[4]) and is_date(c[5]) and c[6].strip().upper() in ("A", "B"):
            cat = c[6].strip().upper()
            rows.append({"list_type": "renouvele_2020", "publication_date": "2020-06-05", "category": cat,
                         "nom": c[0], "prenom": c[1], "epouse": "", "governorate": c[2],
                         "delegation": c[3] if cat == "A" else "", "commune": c[3] if cat == "B" else "",
                         "date_inscription": c[4], "date_renouvellement": c[5],
                         "source_file": raw.name})
            i += 7
        else:
            i += 1
    return pd.DataFrame(rows)


def patch_renewals_2020() -> None:
    """Re-parse only the 2020 list inside data/renewal_records.csv (fast)."""
    df = pd.read_csv(DATA / "renewal_records.csv", dtype=str).fillna("")
    new = scrape_renewals_2020()
    for c in ["nom", "prenom", "epouse", "governorate", "delegation", "commune"]:
        new[c] = new[c].fillna("").astype(str).str.upper().str.strip()
    df = pd.concat([new[df.columns], df[df.list_type != "renouvele_2020"]], ignore_index=True)
    df.to_csv(DATA / "renewal_records.csv", index=False, encoding="utf-8")
    print(new.category.value_counts())


def build_renewal_records() -> pd.DataFrame:
    frames = []
    try:
        frames.append(scrape_renewals_2020())
    except Exception as exc:  # noqa: BLE001
        print("  ! 2020 renewal list unavailable:", exc)
    for name, (pub, lt) in RENEWAL_PDFS.items():
        p = DOCS / "waiting_lists" / name
        if p.exists():
            frames.append(parse_renewal_pdf(p, pub, lt))
    df = pd.concat(frames, ignore_index=True)
    for c in ["nom", "prenom", "epouse", "governorate", "delegation", "commune"]:
        df[c] = df[c].fillna("").astype(str).str.upper().str.strip()
    cols = ["list_type", "publication_date", "category", "nom", "prenom", "epouse", "governorate",
            "delegation", "commune", "date_inscription", "date_renouvellement", "source_file"]
    df = df[cols]
    df.to_csv(DATA / "renewal_records.csv", index=False, encoding="utf-8")
    return df


# ----------------------------------------------------------------------------
# Build structured datasets from the archived documents (works offline)
# ----------------------------------------------------------------------------
GOV_HDX = {"ARIANA": "Ariana", "BEJA": "Béja", "BEN AROUS": "Ben Arous", "BIZERTE": "Bizerte", "GABES": "Gabès",
           "GAFSA": "Gafsa", "JENDOUBA": "Jendouba", "KAIROUAN": "Kairouan", "KASSERINE": "Kassérine",
           "KEBILI": "Kebili", "LE KEF": "Le Kef", "MAHDIA": "Mahdia", "MANOUBA": "Manubah", "MEDENINE": "Médenine",
           "MONASTIR": "Monastir", "NABEUL": "Nabeul", "SFAX": "Sfax", "SIDI BOUZID": "Sidi Bou Zid",
           "SILIANA": "Siliana", "SOUSSE": "Sousse", "TATAOUINE": "Tataouine", "TOZEUR": "Tozeur", "TUNIS": "Tunis",
           "ZAGHOUAN": "Zaghouan"}

# DPM unit -> INS/HDX delegation pcodes, when automatic name matching is not reliable.
# "parent": delegation created after the 2022 boundary file; located with its parent delegation.
XWALK_OVERRIDES = {
    "102": ("TN1152;TN1153;TN1154;TN1155;TN1156;TN1157;TN1158;TN1159;TN1161;TN1162;TN1163;TN1165;TN1166;TN1167;TN1168",
            "commune", "Commune de Tunis (art. 3) – plusieurs délégations INS"),
    "201": ("TN1251", "commune", "Commune de l'Ariana (art. 3) ≈ délégation Ariana Ville"),
    "210": ("TN1256", "exact", ""),
    "302": ("TN1352", "exact", ""), "309": ("TN1356", "exact", ""),
    "1401": ("TN3151;TN3152;TN3153;TN3154", "commune", "Commune de Sousse (art. 3) – délégations Sousse Médina/Riadh/Jaouhara/Sidi Abdelhamid"),
    "1402": ("TN3166", "commune", "Commune de Zaouiet Sousse (art. 3) – localisée sur la délégation Zaouiet Ksibet Thrayet"),
    "1403": ("TN3166", "commune", "Commune de Ksibet Thrayet (art. 3) – localisée sur la délégation Zaouiet Ksibet Thrayet"),
    "1415": ("TN3152", "commune", "Commune d'Ezzouhour (art. 3) – localisée sur Sousse Riadh (approx.)"),
    "1701": ("TN3451;TN3452", "commune", "Commune de Sfax (art. 3) – délégations Sfax Médina/Sfax Ouest (approx.)"),
    "2104": ("TN5157", "exact", ""), "2113": ("TN5157", "parent", "El Hamma Ouest – nouvelle délégation, issue d'El Hamma"),
    "2106": ("TN5158", "exact", ""), "2108": ("TN5159", "exact", ""),
    "1705": ("TN3455", "exact", ""),
}
# New delegations (post-2015) without their own polygon: located on the parent delegation (by name).
PARENT_BY_NAME = {
    ("GABES", "DKHILET TOUJANE"): "Mareth", ("GABES", "OUEDHREF"): "Métouia",
    ("GAFSA", "SIDI BOUBAKER"): "Gafsa Sud", ("GAFSA", "ZANOUCH"): "Gafsa Sud",
    ("KAIROUAN", "AIN JLOULA"): "Oueslatia", ("KAIROUAN", "MENZEL MHIRI"): "Kairouan Sud",
    ("KEBILI", "RJIM MAATOUG"): "Faouar", ("LE KEF", "TOUIREF"): "Nebeur",
    ("MAHDIA", "BRADAA"): "Ksour Essef", ("MAHDIA", "REJICHE"): "Mahdia",
    ("SIDI BOUZID", "ESSAIDA"): "Sidi Bouzid Ouest", ("SIDI BOUZID", "HICHRIA"): "Sidi Bouzid Est",
    ("TATAOUINE", "BENI MHIRA"): "Tataouine Nord", ("TOZEUR", "HAMET JRID"): "Tozeur",
}


def _norm_name(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"\[.*?\]|\(.*?\)", " ", s)
    s = re.sub(r"[^A-Z ]", " ", s)
    s = re.sub(r"\b(EL|EZ|ES|ET|ER|ED|EN|DE|LA|LE|DES|COMMUNE|VILLE|ANCIEN|BHIRA)\b", " ", s)
    s = s.replace("DJ", "J").replace("OU", "O").replace("EE", "E").replace("K", "Q").replace("GH", "G")
    return re.sub(r"\s+", "", s)


def build_crosswalk(dele: pd.DataFrame, hdx: pd.DataFrame) -> pd.DataFrame:
    import difflib
    rows = []
    for _, r in dele.iterrows():
        code = str(r.del_code)
        hg = hdx[hdx.adm2_name == GOV_HDX[r.governorate]]
        if code in XWALK_OVERRIDES:
            pcodes, quality, note = XWALK_OVERRIDES[code]
        elif (r.governorate, r.delegation) in PARENT_BY_NAME:
            parent = PARENT_BY_NAME[(r.governorate, r.delegation)]
            m = hg[hg.adm3_name.map(_norm_name) == _norm_name(parent)]
            pcodes = m.adm3_pcode.iloc[0] if len(m) else ""
            quality, note = "parent", f"Nouvelle délégation, localisée sur {parent} (approx.)"
        else:
            cand = {_norm_name(n): p for n, p in zip(hg.adm3_name, hg.adm3_pcode)}
            m = difflib.get_close_matches(_norm_name(r.delegation), list(cand), n=1, cutoff=0.6)
            pcodes, quality, note = (cand[m[0]], "auto", "") if m else ("", "none", "non localisée")
        rows.append({"del_code": code, "pcodes": pcodes, "location_quality": quality, "location_note": note})
    return pd.DataFrame(rows)


def _parse_citypop_md(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and cells[1] in ("Governorate", "Delegation", "Municipality"):
            def num(x):
                x = x.replace(",", "")
                return int(x) if x.isdigit() else None
            rows.append({"name": cells[0], "status": cells[1], "p2004": num(cells[3]), "p2014": num(cells[4]), "p2024": num(cells[5])})
    df = pd.DataFrame(rows)
    gov = None
    govs = []
    for _, r in df.iterrows():
        if r.status == "Governorate":
            gov = r["name"]
        govs.append(gov)
    df["governorate_cp"] = govs
    return df


def _haversine(lat1, lon1, lat2, lon2):
    import numpy as np
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_datasets(snapshot: str | None = None, log=print) -> None:
    """Create the CSV/JSON/GeoJSON files used by the app from archived sources only."""
    import numpy as np
    stat = DOCS / "statistics"
    snaps = pd.read_csv(DATA / "dpm_delegations_snapshots.csv", dtype={"del_code": str, "gov_code": str})
    snapshot = snapshot or snaps.snapshot_date.max()
    dele = snaps[snaps.snapshot_date == snapshot].copy()
    dele["del_code"] = dele.del_code.str.lstrip("0")
    hdx = pd.read_excel(stat / "tun_admin_boundaries.xlsx", sheet_name="tun_admin3")
    hdx["ins_code"] = hdx.adm3_pcode.str[2:]

    # --- crosswalk & coordinates
    xw = build_crosswalk(dele, hdx)
    dele = dele.merge(xw, on="del_code", how="left")
    cent = hdx.set_index("adm3_pcode")[["center_lat", "center_lon", "adm3_name", "adm3_name1"]]
    lat, lon, names_ar = [], [], []
    for p in dele.pcodes.fillna(""):
        ps = [x for x in p.split(";") if x in cent.index]
        lat.append(cent.loc[ps, "center_lat"].astype(float).mean() if ps else np.nan)
        lon.append(cent.loc[ps, "center_lon"].astype(float).mean() if ps else np.nan)
        names_ar.append(cent.loc[ps[0], "adm3_name1"] if len(ps) == 1 else "")
    dele["lat"], dele["lon"], dele["name_ar"] = lat, lon, names_ar
    ref = dele[dele.delegation == "ZRIBA"].iloc[0]
    dele["distance_km"] = _haversine(ref.lat, ref.lon, dele.lat, dele.lon).round(1)
    dele["ins_code"] = dele.pcodes.fillna("").map(lambda p: p[2:] if p and ";" not in p else "")
    xw.to_csv(DATA / "crosswalk_dpm_ins.csv", index=False, encoding="utf-8")

    # --- population (long format)
    pop_rows = []

    def add(level, key, name, gov, year, value, kind, source, verified, note=""):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        pop_rows.append({"level": level, "key": key, "name": name, "governorate": gov, "year": int(year),
                         "population": int(round(float(value))), "kind": kind, "source": source,
                         "verified": verified, "note": note})

    # 1) DPM delegation populations = RGPH 2024 as communicated to the ministry
    for _, r in dele.iterrows():
        add("dpm_unit", r.del_code, r.delegation, r.governorate, 2024, r.population_dpm, "recensement",
            "DPM – listes d'attente (population RGPH 2024)", True)
    # 2) INS 2014 census by delegation (official)
    x14 = pd.read_excel(stat / "pop_log_menages_par_gouv_deleg_1_.xlsx", sheet_name="Délégation", header=None, skiprows=3)
    x14 = x14[pd.to_numeric(x14[0], errors="coerce").notna()]
    ins14 = dict(zip(x14[0].astype(int).astype(str), pd.to_numeric(x14[15], errors="coerce")))
    # La Manouba was split in 2014-2016 (INS 2014 codes 14xx) and governorate 43 = Sidi Bouzid
    # in HDX vs 13 in the INS 2014 file: map by Arabic/French name when the code is unknown.
    nn = hdx.adm3_name.map(_norm_name)
    hdx_by_name = {n: c for n, c in zip(nn, hdx.ins_code) if (nn == n).sum() == 1}
    for code, name in zip(x14[0].astype(int).astype(str), x14[1]):
        if code not in set(hdx.ins_code):
            alt = hdx_by_name.get(_norm_name(name))
            if alt and alt not in ins14:
                ins14[alt] = ins14[code]
    # 3) INS 2024 census by sector, aggregated to delegations (official)
    x24 = pd.read_excel(stat / "Population_Secteur_Age_Sexe_Dynamique.xlsx", sheet_name="Population",
                        usecols=["Code_Delegation", "Délégations", "Population"])
    ins24 = x24.groupby(x24.Code_Delegation.astype(str)).Population.sum().to_dict()
    # 4) citypopulation.de (secondary compilation of INS census results) – used for 2004 and communes
    cp_del = _parse_citypop_md(stat / "citypopulation_tunisia_admin_2026-10-19.md")
    cp_mun = _parse_citypop_md(stat / "citypopulation_tunisia_mun_2026-10-19.md")

    for _, h in hdx.iterrows():
        code = h.ins_code
        gov = h.adm2_name
        add("delegation_ins", code, h.adm3_name, gov, 2014, ins14.get(code), "recensement", "INS – RGPH 2014 par délégation", True)
        add("delegation_ins", code, h.adm3_name, gov, 2024, ins24.get(code), "recensement", "INS – RGPH 2024 par secteur (agrégé)", True)
    # 2004 values: match citypopulation delegation rows to INS 2014 values (same 2014 figure => same unit)
    cp_d = cp_del[cp_del.status == "Delegation"]
    by14 = {}
    for code, v in ins14.items():
        if pd.notna(v):
            by14.setdefault(int(v), []).append(code)
    for _, r in cp_d.iterrows():
        codes = by14.get(r.p2014) or by14.get((r.p2014 or 0) - 1) or by14.get((r.p2014 or 0) + 1) or []
        if len(codes) == 1:
            h = hdx[hdx.ins_code == codes[0]].iloc[0]
            add("delegation_ins", codes[0], h.adm3_name, h.adm2_name, 2004, r.p2004, "recensement",
                "citypopulation.de (compilation des résultats INS) – source secondaire", False,
                "2004 non retrouvé dans un fichier INS téléchargeable ; valeur à confirmer")
    # Governorates 1994-2024 (official INS bilan démographique) and official projections
    gov_census = {
        "Bizerte": [483086, 524128, 568227, 607388], "Béja": [303853, 304501, 303038, 311417],
        "Jendouba": [404783, 416608, 401478, 404352], "Le Kef": [272352, 258790, 243161, 237686],
        "Tunis": [887803, 983861, 1056248, 1075306], "Ariana": [290634, 422246, 576081, 668552],
        "Ben Arous": [371745, 505773, 631840, 722828], "Manouba": [278660, 335912, 379509, 418354],
        "Nabeul": [578618, 693890, 787908, 863172], "Zaghouan": [143036, 160963, 176948, 201065],
        "Sousse": [433709, 544413, 674967, 762281], "Monastir": [363901, 455590, 548828, 599769],
        "Mahdia": [335744, 377853, 410805, 449985], "Kasserine": [386908, 412278, 439239, 492741],
        "Kairouan": [532709, 546209, 570558, 600803], "Siliana": [244910, 233985, 223094, 216242],
        "Sfax": [733687, 855256, 955423, 1047468], "Gafsa": [307513, 323709, 337333, 388776],
        "Sidi Bouzid": [377143, 395506, 429919, 489991], "Tozeur": [89055, 97526, 107906, 120036],
        "Gabès": [311713, 342630, 374307, 410847], "Médenine": [386185, 432503, 479523, 537255],
        "Tataouine": [135703, 143524, 149456, 162654], "Kébili": [131914, 143218, 156958, 183201],
    }
    for g, vals in gov_census.items():
        for y, v in zip([1994, 2004, 2014, 2024], vals):
            add("governorate", g, g, g, y, v, "recensement", "INS – RGPH 2024, bilan démographique (tableau 1994-2024)", True)
    try:
        pj = pd.read_excel(stat / "Excel_Projection_de_la_population_par_gouvernorat_final_2016-2020.docx_1_-4.xlsx", sheet_name=None, header=None)
        _add_ins_projection(pj, add)
    except Exception as exc:  # noqa: BLE001
        log(f"  ! projection INS non lue: {exc}")
    # Communes (Category B is computed per commune)
    gov = None
    for _, r in cp_mun.iterrows():
        if r.status == "Governorate":
            continue
        for y, v in [(2004, r.p2004), (2014, r.p2014), (2024, r.p2024)]:
            add("commune", f"{r.governorate_cp}|{r['name']}", r["name"], r.governorate_cp, y, v, "recensement",
                "citypopulation.de (compilation des résultats INS, communes 2016+) – source secondaire", False,
                "Population recalculée sur les limites communales actuelles ; à confirmer auprès de l'INS")
    pop = pd.DataFrame(pop_rows)
    pop.to_csv(DATA / "population.csv", index=False, encoding="utf-8")

    # --- population history per DPM unit (2004/2014/2024) for forecasting
    p14, p04, p24i = [], [], []
    for _, r in dele.iterrows():
        codes = [c[2:] for c in str(r.pcodes or "").split(";") if c]
        ok = r.location_quality in ("exact", "auto", "commune") and codes
        if ok and r.location_quality != "commune":
            v14 = sum(ins14.get(c, np.nan) for c in codes)
            v24 = sum(ins24.get(c, np.nan) for c in codes)
            sub = pop[(pop.level == "delegation_ins") & (pop.year == 2004) & (pop.key.isin(codes))]
            v04 = sub.population.sum() if len(sub) == len(codes) else np.nan
        else:
            v14 = v24 = v04 = np.nan
        p14.append(v14); p04.append(v04); p24i.append(v24)
    dele["pop_2014"], dele["pop_2004"], dele["pop_2024_ins"] = p14, p04, p24i
    # DPM 2024 vs INS 2024: same census, so the 2014 figure is only kept when the unit is comparable
    same = (dele.pop_2024_ins - dele.population_dpm).abs() <= 0.02 * dele.population_dpm
    dele.loc[~same, ["pop_2014", "pop_2004"]] = np.nan
    dele["pop_history_note"] = np.where(same, "", "Unité DPM non comparable aux délégations INS 2014 (découpage modifié) – croissance estimée au niveau du gouvernorat")

    # --- waiting lists (current snapshot, Category A) + Category B reconstructed from renewal table
    wl = pd.read_csv(DATA / "waiting_lists_snapshots.csv", dtype=str)
    wl = wl[wl.snapshot_date == snapshot].copy()
    wl["del_code"] = wl.del_code.str.lstrip("0")
    wl["source"] = "DPM – liste d'attente en ligne (jour), arrêtée au 31/12/2025, consultée le " + snapshot
    wl["verified"] = True
    rr = pd.read_csv(DATA / "renewal_records.csv", dtype=str).fillna("")
    b = rr[(rr.list_type == "renouvele_2026") & (rr.category == "B")].copy()
    b = b.sort_values(["governorate", "commune", "date_inscription", "nom"])
    b["order"] = b.groupby(["governorate", "commune"]).cumcount() + 1
    b_rows = pd.DataFrame({
        "snapshot_date": "2026-07-03", "category": "B", "governorate": b.governorate, "del_code": "",
        "delegation": "", "commune": b.commune, "rank": b.order.astype(str), "order": b.order, "nom": b.nom,
        "prenom": b.prenom, "epouse": b.epouse, "date_inscription": b.date_inscription,
        "source": "Reconstitué depuis le tableau DPM des renouvellements 2026 (nuit), publié le 03/07/2026 – rang = ordre chronologique (à confirmer)",
        "verified": False})
    wl["commune"] = ""
    wl_all = pd.concat([wl[["snapshot_date", "category", "governorate", "del_code", "delegation", "commune", "rank", "order",
                            "nom", "prenom", "epouse", "date_inscription", "source", "verified"]], b_rows], ignore_index=True)
    wl_all.to_csv(DATA / "waiting_lists.csv", index=False, encoding="utf-8")

    # --- communes table (Category B is planned per commune)
    cm = _build_communes(cp_mun, b, dele)
    cm.to_csv(DATA / "communes.csv", index=False, encoding="utf-8")

    # --- pharmacies (registry, Category A) + observations
    ph = pd.read_csv(DATA / "pharmacies_registry_snapshots.csv", dtype=str)
    ph = ph[ph.snapshot_date == snapshot].copy()
    ph["del_code"] = ph.del_code.str.lstrip("0")
    ph["source"] = "DPM – liste des officines (jour) en ligne, consultée le " + snapshot
    ph["verified"] = True
    obs = pd.DataFrame([{
        "snapshot_date": "2026-10-19", "category": "B", "governorate": "ZAGHOUAN", "del_code": "502", "delegation": "ZRIBA",
        "nom": "HEDI LAOUINI (pharmacie de nuit)", "adresse": "52, Avenue Habib Bourguiba, Hammam Zriba", "tel": "",
        "source": "med.tn (annuaire privé, non officiel) – https://www.med.tn/pharmacy/zaghouan/hedi-laouini-1825.html",
        "verified": False}])
    pd.concat([ph, obs], ignore_index=True).to_csv(DATA / "pharmacies.csv", index=False, encoding="utf-8")

    # --- master delegation table
    dele["waiting_A"] = dele.waiting_count
    dele["pharmacies_A"] = dele.officines_dpm
    keep = ["del_code", "gov_code", "governorate", "delegation", "name_ar", "zone", "zone_threshold", "population_dpm",
            "pop_2014", "pop_2004", "pop_history_note", "pharmacies_A", "waiting_A", "lat", "lon", "distance_km",
            "ins_code", "pcodes", "location_quality", "location_note", "snapshot_date"]
    dele[keep].to_csv(DATA / "delegations.csv", index=False, encoding="utf-8")

    # --- simplified boundaries
    _build_boundaries(log)
    log(f"Jeux de données reconstruits (instantané {snapshot}).")


# DPM commune spelling -> citypopulation spelling (same governorate)
COMMUNE_ALIASES = {
    ("ZAGHOUAN", "BIR MCHARGUA"): "Bir Mcherga", ("ZAGHOUAN", "JEBEL EL WAST"): "Djebel Oust",
    ("ZAGHOUAN", "FAHS"): "El Fahs", ("ZAGHOUAN", "EL AMAYEM"): "El Amaiem", ("ZAGHOUAN", "SAOUEF"): "Saouaf",
    ("MANOUBA", "JEDAIDA"): "Jedeida", ("SOUSSE", "EZ-ZOUHOUR"): "Ezzouhour", ("SOUSSE", "KSIBA THRAYET"): "Ksibet Thrayet",
    ("MEDENINE", "JERBA HOUMET SOUK"): "Houmt Souk", ("NABEUL", "SLIMENE"): "Soliman",
    ("NABEUL", "HAMMAM LAGHZEZ"): "Hammam Ghezèze", ("MAHDIA", "BOU MERDAS"): "Boumerdes",
    ("BEN AROUS", "MHAMDIA"): "Mohamedia-Fouchana",
}
# Communes whose name does not match a delegation: delegation used to locate them (approximate)
COMMUNE_LOCATION = {("ZAGHOUAN", "Djebel Oust"): "BIR MCHARGUA", ("ZAGHOUAN", "El Amaiem"): "EL FAHS"}


def _build_communes(cp_mun: pd.DataFrame, b: pd.DataFrame, dele: pd.DataFrame) -> pd.DataFrame:
    import difflib
    import numpy as np
    cm = cp_mun[cp_mun.status == "Municipality"].copy()
    cm["governorate"] = cm.governorate_cp.map(_match_gov)
    cm = cm.rename(columns={"name": "commune_name", "p2004": "pop_2004", "p2014": "pop_2014", "p2024": "pop_2024"})
    bstats = b.groupby(["governorate", "commune"]).agg(waiting_B=("nom", "size"), first_B_date=("date_inscription", "min")).reset_index()
    cm["commune"] = ""
    extra = []
    for _, r in bstats.iterrows():
        cand = cm[cm.governorate == r.governorate]
        target = COMMUNE_ALIASES.get((r.governorate, r.commune))
        if not target:
            names = {_norm_name(n): n for n in cand.commune_name}
            m = difflib.get_close_matches(_norm_name(r.commune), list(names), n=1, cutoff=0.75)
            target = names[m[0]] if m else None
        idx = cand.index[cand.commune_name == target] if target else []
        if len(idx) and not cm.loc[idx[0], "commune"]:
            cm.loc[idx[0], ["commune", "waiting_B", "first_B_date"]] = [r.commune, r.waiting_B, r.first_B_date]
        else:
            extra.append({"governorate": r.governorate, "commune_name": r.commune.title(), "commune": r.commune,
                          "waiting_B": r.waiting_B, "first_B_date": r.first_B_date})
    cm = pd.concat([cm, pd.DataFrame(extra)], ignore_index=True)
    cm["waiting_B"] = pd.to_numeric(cm.waiting_B, errors="coerce").fillna(0).astype(int)
    cm["population_note"] = np.where(cm.pop_2024.isna(), "Population communale non disponible (nom non apparié) – données manquantes",
                                     "citypopulation.de (compilation INS) – à confirmer")
    dl = dele[["governorate", "delegation", "lat", "lon", "distance_km"]].copy()
    dl["n"] = dl.delegation.map(_norm_name)
    loc = []
    for _, r in cm.iterrows():
        cand = dl[dl.governorate == r.governorate]
        forced = COMMUNE_LOCATION.get((r.governorate, r.commune_name))
        if forced:
            c = cand[cand.delegation == forced].iloc[0]
            loc.append((c.lat, c.lon, c.distance_km, c.delegation))
            continue
        m = difflib.get_close_matches(_norm_name(r.commune if r.commune else r.commune_name), list(cand.n), n=1, cutoff=0.7)
        if not m:
            m = difflib.get_close_matches(_norm_name(r.commune_name), list(cand.n), n=1, cutoff=0.7)
        if m:
            c = cand[cand.n == m[0]].iloc[0]
            loc.append((c.lat, c.lon, c.distance_km, c.delegation))
        else:
            loc.append((np.nan, np.nan, np.nan, ""))
    cm["lat"], cm["lon"], cm["distance_km"], cm["located_on_delegation"] = zip(*loc)
    cm["night_zone"] = cm.commune_name.map(_night_zone)
    return cm[["governorate", "commune_name", "commune", "pop_2004", "pop_2014", "pop_2024", "population_note", "waiting_B",
               "first_B_date", "night_zone", "lat", "lon", "distance_km", "located_on_delegation"]]


def _match_gov(name: str) -> str:
    n = _norm_name(name)
    for g in GOV_HDX:
        if _norm_name(g) == n or _norm_name(GOV_HDX[g]) == n:
            return g
    aliases = {"MANOBA": "MANOUBA", "QEF": "LE KEF", "MEDNINE": "MEDENINE"}
    return aliases.get(n, name.upper())


# Annex 2 of decree 2019-1013 lists the communes of night-pharmacy zone 1 (1 per 50 000) –
# only the large urban communes are recorded here; everything else defaults to zone 2.
NIGHT_ZONE1_HINT: set[str] = set()


def _night_zone(name: str) -> str:
    return "1" if _norm_name(name) in NIGHT_ZONE1_HINT else "2 (présumée)"


def _add_ins_projection(sheets: dict, add) -> None:
    """INS national projection 2014-2044 (medium hypothesis), sheet 'Feuil1', block 'Total'."""
    df = sheets["Feuil1"]
    first_col = df[0].astype(str).str.strip()
    total_rows = df.index[first_col == "Total"].tolist()
    if not total_rows:
        return
    tot = total_rows[-1]  # last block = both sexes
    hdr = max(i for i in df.index[:tot] if isinstance(df.iloc[i, 1], (int, float)) and df.iloc[i, 1] == 2014)
    for y, v in zip(df.iloc[hdr, 1:].tolist(), df.iloc[tot, 1:].tolist()):
        if isinstance(y, (int, float)) and not pd.isna(y) and isinstance(v, (int, float)) and not pd.isna(v):
            add("national_projection", "TN", "Tunisie", "", int(y), v, "projection",
                "INS – projections de la population 2014-2044 (hypothèse moyenne)", True,
                "Projection officielle établie sur la base du RGPH 2014 ; ne tient pas compte du RGPH 2024")


def _build_boundaries(log=print) -> None:
    import zipfile
    import geopandas as gpd
    z = zipfile.ZipFile(DOCS / "statistics" / "tun_admin_boundaries.geojson.zip")
    with z.open("tun_admin3_em.geojson") as f:
        g = gpd.read_file(f)
    g = g[["adm3_pcode", "adm3_name", "adm3_name1", "adm2_name", "geometry"]].copy()
    g["geometry"] = g.geometry.simplify(0.004, preserve_topology=True)
    g["geometry"] = g.geometry.set_precision(0.0001)
    g.to_file(DATA / "boundaries.geojson", driver="GeoJSON")
    log(f"  limites simplifiées : {(DATA / 'boundaries.geojson').stat().st_size // 1024} Ko")


def refresh_all(log=print) -> None:
    """Manual refresh entry point (used by the app button)."""
    log("Téléchargement des documents officiels…")
    download_all_documents()
    log("Instantané des listes d'attente et des officines (DPM)…")
    refresh_dpm_snapshot()
    log("Reconstruction des fichiers de données…")
    build_datasets(log=log)
    log("Terminé.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "docs":
        download_all_documents()
    elif cmd == "dpm":
        refresh_dpm_snapshot()
    elif cmd == "renewals":
        print(build_renewal_records().groupby(["list_type", "category"]).size())
    elif cmd == "night":
        scrape_night_codes().to_csv(DATA / "dpm_night_codes.csv", index=False, encoding="utf-8")
    elif cmd == "build":
        build_datasets()
    elif cmd == "patch2020":
        patch_renewals_2020()
    else:
        refresh_all()
