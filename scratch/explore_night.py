import requests, sys, re, urllib3
from bs4 import BeautifulSoup
urllib3.disable_warnings()
sys.stdout.reconfigure(encoding="utf-8")
s = requests.Session(); s.headers["User-Agent"] = "Mozilla/5.0"; s.verify = False
B = "https://dpm.tn/dpm_pharm/asppharm/"
s.get(B + "listgouv_att.php")
r = s.post(B + "listedelega_att.php", data={"cod_gouv": "05", "Zone4": "OFF"})
print(re.findall(r'<option value="(\d+)">\s*([^<]+)', r.text))
r = s.post(B + "listeatt.php", data={"cod_del": sys.argv[1] if len(sys.argv) > 1 else "0502", "CHERCHER2": "Continuer"})
open("documents/raw_html/listeatt_night_sample.html", "wb").write(r.content)
print(re.sub(r"\n\s*\n+", "\n", BeautifulSoup(r.content, "html.parser").get_text("\n", strip=True))[:5000])
