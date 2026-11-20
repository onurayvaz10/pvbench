"""Ani kapanmadan sonra PVOutput ham dosyalarini onarir.

17.11.2026 23:14'te Windows beklenmedik sekilde kapandi (Kernel-Power 41 / olay 6008). Son birkac dakikada
yazilan dosyalarin sonu NUL baytla doldu: ilerleme.csv, site_kaynakli_gunler.csv ve bazi santrallerin ay dosyalari.

Adimlar
1. NUL bayt ya da bicimi bozuk satir iceren dosyalar once yedeklenir: veri/ham/pvoutput/_onarim_yedek/<zaman>/.
2. NUL baytlar silinir, bicimi bozuk satirlar atilir:
   ham ay dosyasi 11 alan (YYYYMMDD,HH:MM,...); ilerleme.csv 5 alan; site_kaynakli_gunler.csv 4 alan.
3. Tutarlilik: ilerleme.csv'de son durumu 'ok' olan her gun icin ham dosyadaki farkli zaman sayisi, kaydedilen
   kayit sayisindan azsa o gunun ilerleme ve site_kaynakli_gunler satirlari silinir. Gun yeniden indirilir;
   okuyucu (pvbench.pvoutput.read_raw) ayni zamanli satirlarda sonuncuyu tuttugu icin eski kismi satirlar zararsizdir.

Varsayilan yalnizca rapor verir; --uygula ile dosyalari yazar. Cekim surerken calistirilmamalidir.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

KOK = Path(__file__).resolve().parents[2]
HAM = KOK / "veri" / "ham" / "pvoutput"
AY_DOSYASI = re.compile(r"^20\d\d-\d\d\.csv$")
HAM_SATIR = re.compile(r"^\d{8},\d\d:\d\d,")


def satirlari_oku(yol: Path):
    metin = yol.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace")
    return metin.splitlines()


def ham_gecerli(satir):
    return bool(HAM_SATIR.match(satir)) and len(satir.split(",")) == 11


def ilerleme_gecerli(satir):
    alan = satir.split(";")
    return len(alan) == 5 and alan[3].isdigit() and len(alan[1]) == 8


def iz_gecerli(satir):
    alan = satir.split(";")
    return len(alan) == 4 and alan[2].isdigit() and len(alan[1]) == 8


def temizle(yol: Path, gecerli, ust_bilgi=False):
    """(yeni_satirlar, atilan_sayisi, nul_var)"""
    nul = b"\x00" in yol.read_bytes()
    satirlar = satirlari_oku(yol)
    bas = satirlar[:1] if ust_bilgi else []
    govde = satirlar[1:] if ust_bilgi else satirlar
    iyi = [s for s in govde if gecerli(s)]
    bos_olmayan = [s for s in govde if s.strip()]
    return bas + iyi, len(bos_olmayan) - len(iyi), nul


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ham", default=str(HAM))
    p.add_argument("--uygula", action="store_true", help="Dosyalari yedekleyip onar (yoksa yalnizca rapor)")
    args = p.parse_args(argv)
    ham = Path(args.ham)
    yedek = ham / "_onarim_yedek" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    yazilacak = {}

    # 1-2. Bicim temizligi
    ilerleme_satir, atilan, nul = temizle(ham / "ilerleme.csv", ilerleme_gecerli)
    print(f"ilerleme.csv: {len(ilerleme_satir)} gecerli satir, {atilan} bozuk satir, NUL {'var' if nul else 'yok'}")
    iz_satir, atilan_iz, nul_iz = temizle(ham / "site_kaynakli_gunler.csv", iz_gecerli)
    print(f"site_kaynakli_gunler.csv: {len(iz_satir)} gecerli satir, {atilan_iz} bozuk satir, NUL {'var' if nul_iz else 'yok'}")

    # Son durum (sonraki satir oncekini gecersiz kilar)
    son = {}
    for s in ilerleme_satir:
        sid, tarih, durum, kayit, _ = s.split(";")
        son[(sid, tarih)] = (durum, int(kayit))

    # 3. Ham dosyalar: temizlik + gun bazinda farkli zaman sayisi
    sayim = defaultdict(set)
    ham_degisen = {}
    for klasor in sorted(d for d in ham.iterdir() if d.is_dir() and d.name.isdigit()):
        for yol in sorted(f for f in klasor.iterdir() if AY_DOSYASI.match(f.name)):
            satirlar, atilan_h, nul_h = temizle(yol, ham_gecerli)
            if atilan_h or nul_h:
                ham_degisen[yol] = satirlar
                print(f"  {yol.relative_to(ham)}: {atilan_h} bozuk satir atilacak, NUL {'var' if nul_h else 'yok'}")
            for s in satirlar:
                tarih, saat = s.split(",", 2)[:2]
                sayim[(klasor.name, tarih)].add(saat)

    eksik = sorted(k for k, (durum, kayit) in son.items() if durum == "ok" and len(sayim.get(k, ())) < kayit)
    print(f"Ilerlemede 'ok' gorunup ham dosyada eksik kalan gun: {len(eksik)}")
    aylik = defaultdict(int)
    for sid, tarih in eksik:
        aylik[(sid, tarih[:6])] += 1
    for (sid, ay), n in sorted(aylik.items()):
        print(f"  {sid} {ay}: {n} gun")
    eksik_k = set(eksik)

    yazilacak[ham / "ilerleme.csv"] = [s for s in ilerleme_satir if tuple(s.split(";")[:2]) not in eksik_k]
    yazilacak[ham / "site_kaynakli_gunler.csv"] = [s for s in iz_satir if tuple(s.split(";")[:2]) not in eksik_k]
    yazilacak.update(ham_degisen)

    if not args.uygula:
        print("Yalnizca rapor; yazmak icin --uygula.")
        return 0
    for yol, satirlar in yazilacak.items():
        hedef = yedek / yol.relative_to(ham)
        hedef.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(yol, hedef)
        gecici = yol.with_suffix(yol.suffix + ".onarim")
        gecici.write_text("".join(s + "\n" for s in satirlar), encoding="utf-8", newline="")
        gecici.replace(yol)
    print(f"Onarildi: {len(yazilacak)} dosya; yedekler {yedek}")
    return 0


if __name__ == "__main__":
    for akis in (sys.stdout, sys.stderr):
        if hasattr(akis, "reconfigure"):
            akis.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
