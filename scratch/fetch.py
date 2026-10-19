"""Scratch helper: fetch a URL, save raw HTML, print visible text and links."""
import sys, re, hashlib, pathlib, requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "documents" / "raw_html"


def fetch(url, method="GET", data=None):
    s = requests.Session()
    s.headers["User-Agent"] = UA
    r = s.request(method, url, data=data, timeout=60, verify=False)
    r.raise_for_status()
    return r


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()
    url = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "text"
    r = fetch(url)
    name = re.sub(r"[^A-Za-z0-9]+", "_", url.split("//", 1)[-1])[:120]
    (RAW / f"{name}.html").write_bytes(r.content)
    soup = BeautifulSoup(r.content, "html.parser")
    main = soup.select_one("#content, .item-page, main, article") or soup.body
    if mode in ("text", "both"):
        txt = re.sub(r"\n\s*\n+", "\n", main.get_text("\n", strip=True))
        sys.stdout.reconfigure(encoding="utf-8")
        print(txt[:15000])
    if mode in ("links", "both"):
        for a in main.find_all("a", href=True):
            print("LINK:", a.get_text(strip=True)[:80], "->", a["href"])
    for f in main.find_all(["form", "iframe"]):
        print("FORM/IFRAME:", f.name, f.get("action") or f.get("src"), f.get("method"))
