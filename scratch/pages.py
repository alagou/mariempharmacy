import requests, sys, re, urllib3
from bs4 import BeautifulSoup
urllib3.disable_warnings(); sys.stdout.reconfigure(encoding="utf-8")
s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0"; s.verify = False
for u in sys.argv[1:]:
    url = u if u.startswith("http") else "https://dpm.tn" + u
    r = s.get(url)
    name = re.sub(r"[^A-Za-z0-9]+", "_", url.split("//", 1)[-1])[-120:]
    open(f"documents/raw_html/{name}.html", "wb").write(r.content)
    soup = BeautifulSoup(r.content, "html.parser")
    print("=====", url, r.status_code)
    for i in soup.find_all("iframe"): print("IFRAME", i.get("src"))
    m = soup.select_one("div.item-page") or soup.select_one("[itemprop=articleBody]") or soup.select_one("div.blog") or soup.select_one("#k2Container")
    if m is None:
        print("(no article body)"); continue
    print(re.sub(r"\n\s*\n+", "\n", m.get_text("\n", strip=True))[:4000])
    for a in m.find_all("a", href=True):
        print("LINK:", a.get_text(strip=True)[:60], "->", a["href"])
