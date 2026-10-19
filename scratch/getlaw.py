"""Fetch a legal web page, save raw HTML under documents/regulations/web/, print main text."""
import sys, re, requests, urllib3, pathlib
from bs4 import BeautifulSoup
urllib3.disable_warnings(); sys.stdout.reconfigure(encoding="utf-8")
url, name = sys.argv[1], sys.argv[2]
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}, timeout=90, verify=False)
print(r.status_code, len(r.content))
d = pathlib.Path("documents/regulations/web"); d.mkdir(parents=True, exist_ok=True)
(d / f"{name}.html").write_bytes(r.content)
soup = BeautifulSoup(r.content, "html.parser")
for t in soup(["script", "style", "nav", "header", "footer"]): t.decompose()
txt = re.sub(r"\n\s*\n+", "\n", soup.get_text("\n", strip=True))
(d / f"{name}.txt").write_text(txt, encoding="utf-8")
i = txt.find(sys.argv[3]) if len(sys.argv) > 3 else 0
print(txt[max(0, i):max(0, i) + 12000])
