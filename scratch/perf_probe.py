import os, sys, time, tracemalloc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from dataclasses import asdict
import analysis as an

try:
    import psutil
    rss = lambda: f"rss={psutil.Process().memory_info().rss / 2**20:.0f} MB"
except ImportError:
    rss = lambda: ""

tracemalloc.start()


def step(name, f):
    tracemalloc.reset_peak()
    t = time.time()
    r = f()
    cur, peak = tracemalloc.get_traced_memory()
    print(f"{name}: {time.time() - t:.1f}s  traced peak={peak / 2**20:.0f} MB  {rss()}", flush=True)
    return r


D = step("load_data", an.load_data)
S = an.default_settings(D)
P = D["profile"]
step("compare_delegations", lambda: an.compare_delegations(D, 100, S))
step("build_scenarios", lambda: an.build_scenarios(D, S, None, None, None, 1))
alts = step("all_alternatives", lambda: an.all_alternatives(D, S))
print("alternatives:", alts.shape, f"{alts.memory_usage(deep=True).sum() / 2**20:.1f} MB")
