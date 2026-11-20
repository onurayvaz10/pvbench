"""Hatali islemci cekirdeginde uretilen sonuclarin dogrulanmasi (20 Kasim 2026).

Olay: 20 Kasim 2026 03:38'den itibaren Windows olay gunlugu, ucuncu performans cekirdeginde (mantiksal islemci 4 ve 5,
APIC 16/17) "duzeltilen donanim hatasi" kaydetti; 14:30'da bilgisayar hata denetimiyle (0x50) yeniden basladi.
Egitim iscileri birer performans cekirdegine baglidir (`experiment.boost_process`): isci 2 bu cekirdekte calisir, ve
her iscinin sonuclari kendi dosyasindadir (`*_w2.csv`).  Bu betik, isci 2'nin KESME anindan sonra yazdigi satirlari
saglam cekirdeklerde yeniden uretmek ve eskisiyle karsilastirmak icindir.

    hazirla      <cikti>/dogrulama/<kosu> klasorlerini kurar: sonuc dosyalarinin kopyalari, isci 2'nin KESME sonrasi
                 satirlari cikarilmis olarak (kosu kodu eksik isleri kendiliginden yeniden yapar); cikarilan satirlar
                 beklenen.csv dosyasina yazilir.  Asil kosu klasorlerine dokunulmaz.
    karsilastir  yeniden uretilen satirlari beklenen.csv ile eslestirir: yontem basina satir sayisi, birebir ayni
                 olanlar, en buyuk mutlak nRMSE farki.  Ridge, LightGBM ve evrisim katmani icermeyen aglar (dense,
                 simplernn, lstm, gru, bilstm, transformer) bit duzeyinde yinelenebilir oldugundan farkin sifir olmasi
                 beklenir; evrisim katmanli aglarda (cnn, cnn_lstm, cnn_gru, cnn_lstm_gru, tcn) GPU'daki evrisim
                 yordamlari belirlenimci olmadigindan farklar olagandir (bkz. belirlenim_denemesi.py).

Cikti (karsilastir): <sonuclar>/dogrulama_isci2.csv
Kullanim:  python dogrulama_isci2.py hazirla <cikti>
           python dogrulama_isci2.py karsilastir <cikti> <sonuclar klasoru>
"""
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

CUTOFF = datetime(2026, 11, 20, 3, 30)                    # first logged hardware error: 03:38:28
BAD_WORKER = 2
RUNS = {                                                   # run folder -> (file pattern, key columns, value column)
    "ana_e3": ("metrics_w*.csv", ["stage", "plant", "horizon", "info", "model", "seed", "trial"], "nrmse"),
    "ana_e5ma": ("metrics_w*.csv", ["stage", "plant", "horizon", "info", "model", "seed", "trial"], "nrmse"),
    "e5_onem": ("importance_w*.csv", ["plant", "horizon", "info", "model", "seed", "group"], "perm_nrmse"),
    "e6": ("metrics_w*.csv", ["fold", "horizon", "info", "model", "seed", "plant"], "nrmse"),
    "e8": ("metrics_w*.csv", ["plant", "horizon", "info", "model", "seed"], "nrmse"),
}
NOT_REPRODUCIBLE = {"cnn", "cnn_lstm", "cnn_gru", "cnn_lstm_gru", "tcn"}      # networks with convolutional layers
# (belirlenim_denemesi.py: two fits with the same seed are bit-identical for the other six architectures and differ
# for these five; the GPU routines for convolutions are not deterministic)


def worker_of(path: Path) -> int:
    return int(re.search(r"_w(\d+)\.csv$", path.name).group(1))


def read(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def late(row) -> bool:
    try:
        return datetime.strptime(row["finished"], "%Y-%m-%d %H:%M:%S") >= CUTOFF
    except (ValueError, KeyError, TypeError):
        return False


def prepare(out: Path):
    for run, (pattern, keys, _) in RUNS.items():
        target = out / "dogrulama" / run
        if (target / "beklenen.csv").exists():
            print(f"{run}: zaten hazir, dokunulmadi")
            continue
        target.mkdir(parents=True, exist_ok=True)
        removed, fields_all, kept_total = [], None, 0
        for path in sorted((out / run).glob(pattern)):
            fields, rows = read(path)
            fields_all = fields
            keep = [r for r in rows if not (worker_of(path) == BAD_WORKER and late(r))]
            removed += [r for r in rows if worker_of(path) == BAD_WORKER and late(r)]
            kept_total += len(keep)
            with open(target / path.name, "w", encoding="utf-8", newline="") as f:      # strings are copied unchanged
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(keep)
        with open(target / "beklenen.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields_all)
            writer.writeheader()
            writer.writerows(removed)
        (target / "hazirlandi.txt").write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        print(f"{run}: {kept_total} satir kopyalandi, {len(removed)} satir yeniden uretilecek")


def compare(out: Path, results: Path):
    import pandas as pd

    summary = []
    for run, (pattern, keys, value) in RUNS.items():
        folder = out / "dogrulama" / run
        expected = pd.read_csv(folder / "beklenen.csv")
        started = pd.Timestamp((folder / "hazirlandi.txt").read_text(encoding="utf-8"))
        new = pd.concat([pd.read_csv(p) for p in folder.glob(pattern)], ignore_index=True)
        new = new[pd.to_datetime(new["finished"], errors="coerce") >= started]
        if "status" in new.columns:
            new = new[new["status"] == "ok"]
        both = expected.merge(new, on=keys, suffixes=("_old", "_new"))
        both["diff"] = (both[f"{value}_new"] - both[f"{value}_old"]).abs()
        for model, part in both.groupby("model"):
            n_expected = int((expected["model"] == model).sum())
            summary.append({"run": run, "model": model, "rows_expected": n_expected, "rows_recomputed": len(part),
                            "identical": int((part["diff"] == 0).sum()), "max_abs_diff": float(part["diff"].max()),
                            "mean_abs_diff": float(part["diff"].mean()), "value": value,
                            "bit_reproducible_model": model not in NOT_REPRODUCIBLE})
        missing = len(expected) - len(both)
        print(f"{run}: beklenen {len(expected)}, yeniden uretilen {len(both)}, eksik {missing}")
    table = pd.DataFrame(summary)
    table.to_csv(results / "dogrulama_isci2.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(6).to_string(index=False))
    exact = table[table["bit_reproducible_model"]]
    print("\nyinelenebilir yontemler: satir", int(exact["rows_recomputed"].sum()), "| birebir ayni",
          int(exact["identical"].sum()), "| en buyuk fark", float(exact["max_abs_diff"].max()) if len(exact) else 0.0)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    if sys.argv[1] == "hazirla":
        prepare(Path(sys.argv[2]))
    else:
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
