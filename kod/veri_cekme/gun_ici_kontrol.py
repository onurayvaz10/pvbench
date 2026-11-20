"""Bir sistemin kayit araligini ve secilen gunlerde gercek uretim olup olmadigini denetler (tani amacli).

Kullanim: gun_ici_kontrol.py SID YYYYMMDD [YYYYMMDD ...]
Her gun icin: kayit sayisi, uretim gucu > 0 olan kayit sayisi, azami guc ve azami gucun yerel saati.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pvoutput_cek as pc  # noqa: E402


def sayi(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sid, gunler = sys.argv[1], sys.argv[2:]
    args = pc.ayristirici().parse_args(["kimlik"])
    anahtar, kimlik = pc.kimlik_bilgileri(args.kimlik)
    istemci = pc.Istemci(anahtar, kimlik)
    kod, metin = istemci.istek("getstatistic.jsp", {"sid1": sid})
    print(f"{sid} istatistik: {metin.strip()[:200]}")
    for gun in gunler:
        kod, metin = istemci.istek("getstatus.jsp", {"d": gun, "h": 1, "limit": 288, "asc": 1, "sid1": sid})
        kayitlar = [k.split(",") for k in metin.split(";") if k.strip()] if kod == 200 else []
        dolu = [(k[1], sayi(k[4])) for k in kayitlar if len(k) > 4 and sayi(k[4]) > 0]
        tepe = max(dolu, key=lambda x: x[1]) if dolu else ("-", 0)
        print(f"  {gun}: kod {kod}, {len(kayitlar)} kayit, guc>0 olan {len(dolu)}, azami {tepe[1]:.0f} W @ {tepe[0]}"
              + (f", ilk {dolu[0][0]} son {dolu[-1][0]}" if dolu else ""))
