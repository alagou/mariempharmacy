import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from streamlit.testing.v1 import AppTest

NAV = {11: "alternatives", 6: "simulator", 0: "overview"}


def new_page(i):
    at = AppTest.from_file("../app.py", default_timeout=240)
    at.run()
    if i:
        at.sidebar.radio[0].set_value(i).run()
    return at


def errors(at):
    return [str(e.value)[:1500] for e in at.exception]


def downloads(at):
    return [x for x in at.get("download_button")]


def check_sorted(df, levels):
    """True if df is ordered by levels [(col, ascending)] (ties broken by the next level)."""
    cols = [c for c, _ in levels]
    exp = df.sort_values(cols, ascending=[a for _, a in levels], na_position="last", kind="mergesort")
    return list(exp[cols].itertuples(index=False)) == list(df[cols].itertuples(index=False))


def explain_flow(at, key, choice=None):
    sb = at.selectbox(key=f"{key}_sel")
    if choice is not None:
        sb.set_value(choice).run()
        sb = at.selectbox(key=f"{key}_sel")
    at.button(key=f"{key}_go").click().run()
    dl = downloads(at)
    return sb.value, [d.proto.label for d in dl], errors(at)


for i in range(12):
    t = time.time()
    at = new_page(i)
    errs = errors(at)
    print(i, at.title[0].value if at.title else "", "errors:", len(errs), f"{time.time() - t:.1f}s")
    for e in errs:
        print("   ", e)
    if i == 8:
        b = [x for x in at.button if "PDF" in x.label]
        if b:
            b[0].click().run()
            print("   report:", [s.value for s in at.success], "downloads:", len(downloads(at)), errors(at))
    if i == 11:
        df = at.dataframe[0].value
        print("   rows:", len(df), "ID column present:", "ID" in df.columns, "IDs unique:", df.ID.is_unique)
        print("   default multi-sort (médiane ↑, ≤10 ans ↓, distance ↑):",
              check_sorted(df, [("Année médiane", True), ("≤ 10 ans", False), ("Distance (km)", True)]))
        at.selectbox(key="alt_sort_0").set_value("Distance (km)").run()
        at.selectbox(key="alt_sort_1").set_value("Catégorie").run()
        at.toggle(key="alt_desc_1").set_value(True).run()
        at.selectbox(key="alt_sort_2").set_value("Année médiane").run()
        df = at.dataframe[0].value
        print("   custom multi-sort (distance ↑, catégorie ↓, médiane ↑):",
              check_sorted(df, [("Distance (km)", True), ("Catégorie", False), ("Année médiane", True)]), "errors:", errors(at))
        # positional row -> stable ID: every displayed row's ID names the same unit / category as the row itself
        ok = all(str(r.ID).split("|")[2].upper() == str(r["Unité"]).replace("★ ", "").upper() for _, r in df.iterrows())
        print("   row position -> ID consistent after multi-sort:", ok, "| first rows:", list(df.ID.head(3)))
        next(t for t in at.toggle if t.label.startswith("Épingler")).set_value(True).run()
        print("   ★ pinned first:", at.dataframe[0].value[["ID", "Unité"]].iloc[0].tolist())
        for choice in ("B|ZAGHOUAN|Saouaf|0", "stay|ZAGHOUAN|ZRIBA|3"):
            sel, dls, errs = explain_flow(at, "alt", choice)
            print(f"   PDF for {sel}: downloads={dls} errors={errs}")
    if i == 6:
        sel, dls, errs = explain_flow(at, "sim")
        print(f"   simulator PDF for {sel}: downloads={dls} errors={errs}")
    if i == 0:
        sel, dls, errs = explain_flow(at, "ov")
        print(f"   overview PDF for {sel}: downloads={dls} errors={errs}")
