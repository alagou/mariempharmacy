# Prévision d'autorisation d'officine – Mariem Chkirbene (Zriba, catégorie A)

Application **locale** (Python + Streamlit) qui estime quand
**CHKIRBENE EP CHKIRBENE Mariem**, inscrite le 26/07/2017 sur la liste d'attente
de la délégation de **Zriba (gouvernorat de Zaghouan), catégorie A (jour)**, pourrait obtenir
une autorisation d'ouverture d'officine. Elle compare aussi le maintien à Zriba avec un changement de
délégation (rayon réglable, 100 km par défaut) ou de catégorie (A jour / B nuit).

> **Avertissement.** Outil d'aide à la réflexion uniquement : il ne constitue pas un avis juridique et ne remplace pas
> une réponse écrite de la Direction de la Pharmacie et du Médicament (DPM) ou du Conseil de l'Ordre (CNOPT).
> L'application ne dépose, ne modifie et ne retire **aucune** demande administrative.

---

## 1. Installation et lancement

Prérequis : Python 3.10 ou plus récent (testé avec 3.11, Windows).

```powershell
cd C:\CursorProjects\MariemPharmacy
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Ouvrir ensuite <http://localhost:8501>.

- Le serveur est lié à **127.0.0.1 uniquement** (`.streamlit/config.toml` : `server.address = "127.0.0.1"`).
  Il n'est donc pas accessible depuis le réseau local.
- La télémétrie Streamlit est désactivée (`browser.gatherUsageStats = false`).
- Le tableau de bord fonctionne **hors ligne**. La carte est dessinée à partir de `data/boundaries.geojson`,
  sans fond de carte en ligne.
- Le seul accès réseau possible est le bouton manuel **« Actualiser les données »** (voir section 4).

## 2. Contenu du tableau de bord

La langue par défaut est le français ; l'anglais et l'arabe sont disponibles pour la navigation et les libellés principaux.

| Page | Contenu |
|---|---|
| Vue d'ensemble | Situation vérifiée (rang 5/5, population, seuil), synthèse et avertissement sur le retrait |
| Carte | Délégations dans le rayon choisi, colorées selon l'indicateur ; communes (catégorie B) ; Zriba |
| Comparer les délégations | Tableau : population, officines, places libres, inscrits, prochaine place, simulation |
| Catégorie A vs B | Règles distinctes et calculs séparés (listes jamais mélangées) |
| Population | Historique 2004 / 2014 / 2024 et projections jusqu'en 2045 (linéaire, TCAM, faible / central / élevé) |
| Historique des listes | Liste de Zriba au fil des instantanés, renouvellements 2020 / 2024 / 2026, flux |
| Simulateur rester / changer | Scénarios A à E, probabilités par classes, courbes cumulées, classements |
| Réglementation | Textes (liens et copies locales), réponses aux 7 questions juridiques avec leur statut |
| Explorer toutes les options | Toutes les options A / B (tri sur plusieurs colonnes) ; PDF explicatif de l'option sélectionnée |
| Rapport | Génération d'un rapport PDF, téléchargé directement (rien n'est écrit sur le disque) |
| Données & sources | Provenance de chaque fichier (URL, date, empreinte SHA-256), données téléchargeables |

Les hypothèses (rayon, mode de mise à jour de la population, croissance, attrition, créations exceptionnelles,
délai administratif, horizon, seuil de la catégorie B) se règlent dans la barre latérale, section **Hypothèses**.

## 3. Méthode (résumé)

### Cadre juridique retenu

- **Décret n° 2019-1013** (numerus clausus) :
  - catégorie A : nombre d'officines = arrondi(population / N), avec N compris entre 3 300 et 16 000 selon la zone ;
    Zriba est en zone à 10 000 habitants ;
  - catégorie B : une officine par tranche de 50 000 (zone 1) ou 60 000 (zone 2) habitants.
    L'annexe 2 n'a pas été retrouvée : la valeur par défaut est 60 000, réglable ;
  - exceptions de l'art. 6 (première officine d'une commune, officines d'imada, première officine de nuit),
    comptées dans le numerus clausus (art. 7) ;
  - la population retenue est celle communiquée par l'INS (art. 18). La DPM indique utiliser le **RGPH 2024**.
- **Arrêté du 19/12/2019** (listes d'attente) :
  - une seule liste par pharmacien (art. 5) et classement chronologique (art. 6) ;
  - renouvellement tous les 2 ans, en janvier des années paires (art. 7) ; **prochaine échéance : janvier 2028**.
- Aucun texte trouvé ne permet de **transférer l'ancienneté** d'une liste à une autre. Changer de délégation ou de catégorie
  signifie donc, en l'état des textes consultés, une nouvelle inscription à la date du nouveau dépôt.
  Ce point est marqué « à confirmer » auprès de la DPM.

Le détail (articles, statut de chaque réponse : *texte*, *interprétation* ou *à confirmer*, et l'organisme à consulter)
se trouve dans `data/regulations.json` et sur la page **Réglementation**.

### Population

- Données officielles INS : RGPH 2004, 2014 et 2024 par délégation, communes, projection nationale.
- Projections jusqu'en 2045 :
  - linéaire ;
  - TCAM 2014-2024 (scénario central) ;
  - scénarios faible et élevé, bornés par les taux observés sur 2004-2014 et 2014-2024.
- Mode par défaut : **recensements seulement**. La population officielle n'est mise à jour qu'aux recensements
  (2034, 2044), puis un délai administratif de 2 ans est appliqué. Le mode « estimations annuelles INS » est proposé à titre de comparaison.

### Simulation de la file d'attente (Monte-Carlo, 4 000 tirages)

- **Attrition annuelle** des inscrits devant elle (abandons, décès, autre voie) : environ 4,2 % par an.
  Elle est estimée à partir des tableaux de renouvellement 2024 → 2026, avec une incertitude modélisée par une loi bêta.
- **Nouvelles places** : elles apparaissent quand la population officielle franchit le seuil suivant.
- **Créations exceptionnelles** (art. 6) : loi de Poisson, environ 0,015 par délégation et par an.
- **Délai de traitement** d'une place ouverte : 1 an plus un délai aléatoire.
- **Probabilités** : affichées **par classes** (< 5 %, 5–20 %, 20–50 %, 50–80 %, > 80 %), car les données ne justifient pas
  une précision supérieure.
- **Contrôles rétrospectifs** :
  - l'attrition prédite diffère de l'attrition observée d'environ 2,6 points ;
  - la formule du numerus clausus reproduit exactement le nombre d'officines de 59 % des délégations, et à ±1 près pour 90 % d'entre elles.

### Scénarios

| Code | Scénario |
|---|---|
| A | Rester à Zriba, catégorie A (ancienneté 2017 conservée) |
| B | Nouvelle inscription en catégorie A dans une autre délégation (choisie dans le simulateur) |
| C | Nouvelle inscription en catégorie B (nuit) à Zriba |
| D | Nouvelle inscription en catégorie B dans une commune sans officine de nuit ni inscrit (D0 : aucune officine B ; D1 : une officine B existe) |
| E | Transfert en conservant la date de 2017 : **hypothèse non prévue par les textes**, affichée pour comparaison uniquement |

Chaque scénario présente :

- la faisabilité juridique et la date d'inscription retenue ;
- le rang estimé ;
- l'année plausible au plus tôt (10 %) et la période la plus probable ;
- les classes de probabilité à 3, 5 et 10 ans ;
- le risque juridique et le niveau de confiance ;
- trois classements : par délai, par risque, et global.

Le texte de synthèse **ne recommande jamais un retrait**. Un scénario plus rapide ne suffit pas à justifier l'abandon d'une inscription.
Les conséquences d'un retrait (perte définitive de l'ancienneté du 26/07/2017, caractère irréversible, etc.) sont rappelées
partout où un changement est évoqué.

## 4. Données et provenance

| Fichier | Contenu |
|---|---|
| `data/profile.json` | Situation personnelle de Mariem (**données personnelles, à garder en local**) |
| `data/delegations.csv` | 263 unités DPM : population RGPH 2024 / 2014 / 2004, zone, seuil, officines A, inscrits A, coordonnées, distance |
| `data/communes.csv` | 363 communes : population, inscrits B, date du premier inscrit, zone de nuit, coordonnées |
| `data/population.csv` | Population en format long, avec source et indicateur `verified` |
| `data/waiting_lists.csv` | Listes d'attente : A (en ligne DPM, vérifiée) et B (reconstruite à partir du tableau 2026, `verified = False`) |
| `data/waiting_lists_snapshots.csv` | Instantanés datés des listes (jamais écrasés) |
| `data/renewal_records.csv` | Renouvellements 2020 / 2024 / 2026 extraits des documents DPM |
| `data/pharmacies.csv` | Officines (registre DPM) ; les ajouts non officiels sont marqués `verified = False` |
| `data/pharmacy_history.csv` | Nombre d'officines de jour et de nuit, 1990-2013 (graphiques DPM) |
| `data/regulations.json` | Textes, règles, historique du numerus clausus, 7 questions juridiques, conséquences d'un retrait |
| `data/crosswalk_dpm_ins.csv` | Correspondance entre codes DPM et codes INS / HDX |
| `data/boundaries.geojson` | Limites des délégations (HDX/OCHA, simplifiées) |
| `data/sources.json` | Registre de provenance : URL, date de téléchargement, SHA-256, chemin local |
| `documents/` | Copies locales des textes (`regulations/`), des listes (`waiting_lists/`) et des statistiques (`statistics/`) |

Principales sources :

- DPM, <https://dpm.tn> : listes d'attente, registre des officines, tableaux de renouvellement ;
- INS, <https://www.ins.tn> : RGPH 2004, 2014 et 2024, estimations et projections ;
- CNOPT : historique du numerus clausus ;
- HDX/OCHA : limites administratives.

Une seule information provient d'une source **non officielle** : l'existence d'une officine de nuit à Zriba, signalée par med.tn.
Elle est marquée comme telle.

**Règles d'intégrité**

- Aucune valeur manquante n'est inventée. Les communes sans population connue (13) sont signalées et exclues des calculs concernés.
- Les listes et calculs des catégories A et B restent séparés.

### Actualiser les données (manuel uniquement)

Pour actualiser depuis l'application : barre latérale, **« Actualiser les données »**, cocher la confirmation, puis lancer.
Une connexion Internet est nécessaire.

En ligne de commande :

```powershell
python research.py            # tout : documents + instantané DPM + reconstruction des fichiers data/
python research.py docs       # télécharger les documents officiels
python research.py dpm        # nouvel instantané des listes d'attente et du registre DPM
python research.py renewals   # ré-extraire les tableaux de renouvellement
python research.py night      # codes des zones de nuit DPM
python research.py build      # reconstruire data/*.csv à partir des fichiers locaux (sans réseau)
```

Garanties de l'actualisation :

- un fichier existant n'est **jamais écrasé** : une nouvelle version est enregistrée à côté, avec la date en suffixe ;
- chaque téléchargement est inscrit dans `data/sources.json` ;
- chaque instantané des listes est conservé et daté (`documents/waiting_lists/snapshots/AAAA-MM-JJ/`, `*_snapshots.csv`) ;
- aucun téléchargement automatique ni en arrière-plan.

## 5. Résultats au 19/10/2026 (hypothèses par défaut)

- **Zriba, catégorie A**
  - situation : 27 849 habitants (RGPH 2024), 3 officines pour 3 autorisées ; Mariem est **5e sur 5**.
    La liste est inchangée depuis 2020 et les cinq inscrits ont renouvelé en 2020, 2024 et 2026 ;
  - prochaine place : la 4e licence exige 35 000 habitants. Au rythme central (+1,24 %/an), ce seuil ne serait atteint
    qu'au recensement de 2044, soit une licence vers 2046 ;
  - probabilité sous 10 ans : **< 5 %** ; période la plus probable : **après 2050**.
- **Autre délégation, catégorie A** : aucune délégation proche n'offre plus de places libres que d'inscrits.
  Une nouvelle inscription n'y serait pas plus rapide.
- **Catégorie B dans une commune sans officine de nuit ni inscrit** (par ex. Saouaf, El Amaiem) : potentiellement 2027-2029 via l'art. 6.
  Ce résultat est **très incertain** :
  - l'absence d'officine de nuit doit être confirmée ;
  - certaines listes B semblables sont bloquées depuis plusieurs années (Jebel El Wast depuis 2021) ;
  - le changement serait irréversible et l'ancienneté de 2017 serait perdue.
- **Catégorie B à Zriba** : quasi impossible à l'horizon (une officine de nuit signalée, un inscrit de 2014 devant).

**Avant toute démarche**, obtenir une réponse écrite de la DPM sur :

1. le maintien éventuel de l'ancienneté en cas de changement ;
2. le rang exact sur la liste visée ;
3. l'existence d'une officine de nuit dans la commune visée.

Ne pas oublier le **renouvellement de janvier 2028**.

## 6. Limites connues

- L'annexe 2 du décret 2019-1013 (zones de la catégorie B) n'a pas été retrouvée : le seuil de 60 000 est présumé.
- Les listes B sont reconstruites à partir du tableau de renouvellement 2026 et non d'une liste officielle par commune.
- Les créations exceptionnelles, les délais administratifs et l'attrition sont estimés sur peu d'observations.
  Les probabilités restent donc indicatives et sont affichées par classes.
- Quatorze délégations récentes, sans limites propres dans le fichier HDX, sont positionnées sur leur délégation mère.

## 7. Structure du projet

```text
app.py            Tableau de bord Streamlit (10 pages, FR / EN / AR)
research.py       Collecte manuelle des sources officielles et construction des fichiers data/
analysis.py       Chargement des données, capacité par délégation / commune, scénarios A à E
forecasting.py    Règles du numerus clausus, projections de population, simulation Monte-Carlo, contrôles
report.py         Génération du rapport PDF (en mémoire)
explain.py        PDF explicatif d'une option : trajectoire centrale tracée année par année, probabilités
requirements.txt  Dépendances
.streamlit/       Configuration (127.0.0.1, télémétrie désactivée)
data/             Données structurées (CSV / JSON / GeoJSON) et registre de provenance
documents/        Copies locales : regulations/, waiting_lists/, statistics/
scratch/          Scripts d'exploration et de test (non nécessaires à l'application)
```

## 8. Confidentialité

`data/profile.json`, les rapports PDF et les copies des listes contiennent des données personnelles.
Ils restent sur cet ordinateur : ne pas les téléverser ni les partager sans l'accord de Mariem.
