"""Bir santralin gunluk.csv takvimini ve bir ornek gunun gun ici kayitlarini ozetler (tani amacli).

Kullanim: uretim_kontrol.py SID [YYYYMMDD]
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
    sid = sys.argv[1]
    yol = Path(pc.HAM_KLASOR) / sid / "gunluk.csv"
    satirlar = [s.split(",") for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()]
    uretim = [sayi(s[1]) for s in satirlar]
    disari = [sayi(s[3]) for s in satirlar]
    tuketim = [sayi(s[4]) for s in satirlar]
    tepe = [sayi(s[5]) for s in satirlar]
    print(f"{sid}: {len(satirlar)} gun | uretim>0: {sum(u > 0 for u in uretim)} | disari>0: {sum(d > 0 for d in disari)} "
          f"| tuketim>0: {sum(t > 0 for t in tuketim)} | tepe guc>0: {sum(p > 0 for p in tepe)}")
    if len(sys.argv) > 2:
        args = pc.ayristirici().parse_args(["kimlik"])
        anahtar, kimlik = pc.kimlik_bilgileri(args.kimlik)
        istemci = pc.Istemci(anahtar, kimlik)
        kod, metin = istemci.istek("getstatus.jsp", {"d": sys.argv[2], "h": 1, "limit": 288, "asc": 1, "sid1": sid})
        kayitlar = [k.split(",") for k in metin.split(";") if k.strip()] if kod == 200 else []
        enerji = [sayi(k[2]) for k in kayitlar if len(k) > 4]
        guc = [sayi(k[4]) for k in kayitlar if len(k) > 4]
        tuk_guc = [sayi(k[8]) for k in kayitlar if len(k) > 8]
        print(f"  {sys.argv[2]}: {len(kayitlar)} kayit | uretim enerjisi azami {max(enerji, default=0):.0f} Wh | "
              f"uretim gucu azami {max(guc, default=0):.0f} W | tuketim gucu azami {max(tuk_guc, default=0):.0f} W")
        for k in kayitlar[len(kayitlar) // 2: len(kayitlar) // 2 + 2]:
            print("   ", ",".join(k[:9]))
