"""Etkin santrallerin kayitli koordinatini veriden denetler.

Boylam gunes ogle vaktinden (uretim penceresinin orta noktasi, UTC) bulunur; dogrulugu
yaklasik 0,5 derecedir, dogu/bati yonlu dizilerde pencere kaydigi icin 1-2 dereceye cikabilir.
Enlem gun uzunlugundan bulunur; uretim penceresi gercek gunden kisa oldugu icin gercek
veride 8-10 derece dusuk cikar, yalnizca kaba denetim icin raporlanir (bkz. pvbench.sitecheck).

Koordinat kaynagi onceligi: santraller.csv (enlem/boylam) > PVOutput sistem bilgisi
(veri/ham/pvoutput/sistemler.csv) > aday arama dosyalari (aday_santraller*.csv).
EPIAS santralinin verisi saatlik oldugundan boylam saatlik uretimin agirlik merkezinden bulunur.

Cikti: veri/konum_dogrulama.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "kod"))
from pvbench import epias as E  # noqa: E402
from pvbench import pvoutput as P  # noqa: E402
from pvbench.sitecheck import equation_of_time_minutes, estimate_location  # noqa: E402

HAM = KOK / "veri" / "ham"
ULKE_SAAT_DILIMI = {"Hollanda": "Europe/Amsterdam", "Belçika": "Europe/Brussels", "Türkiye": "Europe/Istanbul",
                    "Lüksemburg": "Europe/Luxembourg"}
TUTARLI_DERECE, KONTROL_DERECE = 1.0, 2.5


def oku(yol):
    with open(yol, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def sayi(x):
    try:
        v = float(str(x).replace(",", "."))
        return v if np.isfinite(v) else None
    except ValueError:
        return None


def koordinat_kaynaklari():
    kaynak = {}
    for dosya in ("aday_santraller.csv", "aday_santraller_nl.csv", "aday_santraller_tr.csv"):
        yol = HAM / "pvoutput" / dosya
        if yol.exists():
            for s in oku(yol):
                if sayi(s.get("enlem")) is not None:
                    kaynak[s["sid"]] = (sayi(s["enlem"]), sayi(s["boylam"]), s.get("saat_dilimi", ""), f"PVOutput arama ({dosya})")
    for s in oku(HAM / "pvoutput" / "sistemler.csv"):
        if sayi(s["enlem"]) is not None:
            kaynak[s["sid"]] = (sayi(s["enlem"]), sayi(s["boylam"]), s["saat_dilimi"], "PVOutput sistem bilgisi")
        elif s["sid"] not in kaynak:
            kaynak[s["sid"]] = (None, None, s["saat_dilimi"], "")
    return kaynak


def saatlik_boylam(mwh: pd.Series):
    """Saatlik enerjinin agirlik merkezinden gunes ogle vakti -> boylam (gunluk ortanca)."""
    orta = mwh.index + pd.Timedelta(minutes=30)
    df = pd.DataFrame({"e": mwh.to_numpy(), "t": orta.hour + orta.minute / 60.0, "gun": orta.normalize()})
    boylamlar = []
    for gun, g in df.groupby("gun"):
        if (g["e"] > 0).sum() >= 6:
            merkez = (g["e"] * g["t"]).sum() / g["e"].sum()
            boylamlar.append((12.0 - merkez - float(equation_of_time_minutes(gun.dayofyear)) / 60.0) * 15.0)
    return len(boylamlar), float(np.median(boylamlar)) if boylamlar else np.nan


def main():
    kaynak = koordinat_kaynaklari()
    satirlar = []
    for s in oku(KOK / "veri" / "santraller.csv"):
        if s["atla"] or s["kaynak"] not in ("pvoutput", "epias"):
            continue
        enlem, boylam = sayi(s["enlem"]), sayi(s["boylam"])
        k_enlem, k_boylam, tz, k_ad = kaynak.get(s["sid"], (None, None, "", ""))
        if enlem is not None:
            k_ad = "santraller.csv"
        else:
            enlem, boylam = k_enlem, k_boylam
        tz = tz or ULKE_SAAT_DILIMI.get(s["ulke"], "")
        sonuc = {"no": s["no"], "ad": s["ad"], "ulke": s["ulke"], "kaynak": s["kaynak"], "sid": s["sid"],
                 "saat_dilimi": tz, "enlem": enlem, "boylam": boylam, "koordinat_kaynagi": k_ad}
        if s["kaynak"] == "epias":
            n, b_tahmin = saatlik_boylam(E.read_uevm(HAM / "epias" / "uevm" / f"{s['sid']}.csv"))
            sonuc.update(n_gun=n, boylam_tahmin=round(b_tahmin, 2), enlem_tahmin_kaba="")
        else:
            raw = P.read_raw(HAM / "pvoutput" / s["sid"])
            t = estimate_location(raw, tz)
            sonuc.update(n_gun=t["n_days"], boylam_tahmin=round(t["longitude"], 2),
                         enlem_tahmin_kaba=round(t["latitude"], 1) if np.isfinite(t["latitude"]) else "")
        if boylam is None or not np.isfinite(sonuc["boylam_tahmin"]):
            sonuc.update(boylam_farki="", karar="koordinat yok" if boylam is None else "veri yetersiz")
        else:
            fark = sonuc["boylam_tahmin"] - boylam
            sonuc["boylam_farki"] = round(fark, 2)
            sonuc["karar"] = ("tutarli" if abs(fark) <= TUTARLI_DERECE else
                              "kontrol (yonelim?)" if abs(fark) <= KONTROL_DERECE else "CELISKI")
        satirlar.append(sonuc)
        print(f"{s['no']:>3} {s['ad']:<26} {s['ulke']:<17} kayitli {str(enlem):>10} {str(boylam):>11}  "
              f"tahmin boylam {sonuc['boylam_tahmin']:>7} ({sonuc['n_gun']} gun)  fark {sonuc['boylam_farki']!s:>6}  "
              f"{sonuc['karar']}")
    with open(KOK / "veri" / "konum_dogrulama.csv", "w", encoding="utf-8-sig", newline="") as f:
        alanlar = ["no", "ad", "ulke", "kaynak", "sid", "saat_dilimi", "enlem", "boylam", "koordinat_kaynagi",
                   "n_gun", "boylam_tahmin", "boylam_farki", "enlem_tahmin_kaba", "karar"]
        yazici = csv.DictWriter(f, fieldnames=alanlar, delimiter=";")
        yazici.writeheader()
        yazici.writerows(satirlar)


if __name__ == "__main__":
    for akis in (sys.stdout, sys.stderr):
        if hasattr(akis, "reconfigure"):
            akis.reconfigure(encoding="utf-8", errors="replace")
    main()
