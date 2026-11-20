"""PVOutput arama sorgusu bicimlerini birkac istekle dener (yalnizca tani amacli)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pvoutput_cek as pc  # noqa: E402

DENEMELER = [
    ("Bruksel ll 25km", {"q": "25km", "ll": "50.85,4.35"}),
    ("Bruksel ll 25km seen7", {"q": "25km", "ll": "50.85,4.35", "seen": 7}),
    ("Istanbul ll 25km", {"q": "25km", "ll": "41.01,28.98"}),
    ("Istanbul ll 50km", {"q": "50km", "ll": "41.01,28.98"}),
    ("Ankara ll 50km", {"q": "50km", "ll": "39.93,32.86"}),
    ("Izmir ll 50km", {"q": "50km", "ll": "38.42,27.14"}),
    ("Antalya ll 50km", {"q": "50km", "ll": "36.89,30.71"}),
]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = pc.ayristirici().parse_args(["kimlik"] + sys.argv[1:])
    anahtar, kimlik = pc.kimlik_bilgileri(args.kimlik)
    istemci = pc.Istemci(anahtar, kimlik, asgari_aralik=0.5)
    for ad, prm in DENEMELER:
        kod, metin = istemci.istek("search.jsp", prm)
        satirlar = [s for s in metin.splitlines() if s.strip()]
        print(f"{ad:<24} kod={kod} sonuc={len(satirlar)}")
        for s in satirlar[:3]:
            alan = dict(zip(pc.ARAMA_ALANLARI, pc.csv_satiri(s)))
            print(f"    {alan.get('sistem_adi', '')[:30]:<31} {alan.get('guc_w')} W  cikti={alan.get('cikti_sayisi')}  "
                  f"son={alan.get('son_cikti')}  ({alan.get('enlem')}, {alan.get('boylam')})  mesafe={alan.get('mesafe')}")
