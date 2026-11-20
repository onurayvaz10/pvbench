"""E2, E8 ve E9 icin "en iyi 4 model" secimi (kural kosulardan once yazildi; Ek Not 2).

Kural: ana kiyasin DOGRULAMA hatasi kullanilir (test verisi secime girmez).
  1. Her (santral, ufuk, bilgi kumesi) hucresinde, 11 derin ogrenme modelinin tohum ortalamali
     dogrulama RMSE'si siralanir (1 = en iyi).  S1 hucreleri <cikti>/ana, S2 (ICON) hucreleri
     <cikti>/ana_icon klasorunden okunur.
  2. Bir hucrede 11 modelin hepsi yoksa o hucre kullanilmaz.
  3. Modeller hucreler uzerindeki ortalama siraya gore dizilir; en dusuk dort ortalama sira secilir
     (esitlikte ortalama dogrulama RMSE'si dusuk olan one gecer).

Kullanim:  python en_iyi_modeller.py <cikti klasoru> [--n 4]
Cikti:     <cikti>/e2_modeller.json  ve son satirda secilen model adlari (boslukla ayrili).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench.experiment import REFERENCES, read_metrics  # noqa: E402
from pvbench.models import MODEL_NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--expected-rows", type=int, default=0, help="stop unless ana_icon has this many rows")
    args = ap.parse_args()
    out = Path(args.out)
    parts = []
    for folder, info in (("ana", "S1"), ("ana_icon", "S2")):
        df = read_metrics(out / folder)
        if folder == "ana_icon" and args.expected_rows and len(df) < args.expected_rows:
            sys.exit(f"ana_icon tamamlanmamis: {len(df)} / {args.expected_rows}")
        parts.append(df[(df["stage"] == "main") & (df["info"] == info) & (df["status"] == "ok")
                        & ~df["model"].isin(REFERENCES)])
    df = pd.concat(parts, ignore_index=True)
    cell = df.groupby(["plant", "horizon", "info", "model"])["val_rmse"].mean().unstack("model")
    cell = cell[[m for m in MODEL_NAMES if m in cell.columns]]
    complete = cell.dropna()
    if len(cell.columns) != len(MODEL_NAMES) or complete.empty:
        sys.exit(f"eksik model ya da hucre: {list(cell.columns)}, tam hucre {len(complete)}")
    ranks = complete.rank(axis=1, method="average")
    table = pd.DataFrame({"mean_rank": ranks.mean(), "mean_val_rmse": complete.mean(),
                          "wins": (ranks == 1).sum()}).sort_values(["mean_rank", "mean_val_rmse"])
    chosen = list(table.index[: args.n])
    by_group = {f"{info} h{h}": ranks.xs((h, info), level=("horizon", "info")).mean().round(2).to_dict()
                for h, info in sorted({(h, i) for _, h, i in complete.index})}
    result = {"rule": "mean rank of seed-averaged validation RMSE over plant x horizon x {S1, S2(ICON)} cells",
              "cells_used": int(len(complete)), "cells_dropped": int(len(cell) - len(complete)),
              "chosen": chosen, "table": table.round(4).reset_index().to_dict("records"),
              "mean_rank_by_group": by_group}
    (out / "e2_modeller.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(table.round(4).to_string())
    print(f"hucre: {len(complete)} kullanildi, {len(cell) - len(complete)} disarida")
    print(" ".join(chosen))


if __name__ == "__main__":
    main()
