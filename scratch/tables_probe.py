import sys, pymupdf
sys.stdout.reconfigure(encoding="utf-8")
for f in sys.argv[1:]:
    doc = pymupdf.open(f)
    print("=====", f, doc.page_count)
    for pi in [0, doc.page_count - 1]:
        p = doc[pi]
        tabs = p.find_tables()
        print("page", pi + 1, "tables", len(tabs.tables))
        for tb in tabs.tables[:1]:
            rows = tb.extract()
            print("nrows", len(rows))
            for row in rows[:5]:
                print(row)
        if not tabs.tables:
            print(p.get_text()[:500])
