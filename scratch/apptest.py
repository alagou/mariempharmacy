import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from streamlit.testing.v1 import AppTest

for i in range(10):
    t = time.time()
    at = AppTest.from_file("../app.py", default_timeout=240)
    at.run()
    if i:
        at.sidebar.radio[0].set_value(i).run()
    errs = [e.value for e in at.exception]
    print(i, at.title[0].value if at.title else "", "errors:", len(errs), f"{time.time() - t:.1f}s")
    for e in errs:
        print("   ", str(e)[:1500])
    if i == 8:
        b = [x for x in at.button if "PDF" in x.label]
        if b:
            b[0].click().run()
            print("   report:", [s.value for s in at.success], [str(e.value)[:800] for e in at.exception])
