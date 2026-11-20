"""Makale tablolari: sonuclar/*.csv dosyalarindan Markdown tablo uretir (makale/tablolar/*.md).

Makaledeki tablolar elle yazilmaz; bu betigin ciktisindan alinir.
    tablo1_santraller.md     santral envanteri (tablo_santraller.py -> tablo_santraller.csv)
    tablo2_ana_kiyas.md      E1: 17 yontem x 4 hucre, ortalama nRMSE ve ortalama sira (sonuc_tablolari.py -> e1_ozet.csv)
    tablo_e2.md              E2: bilgi kumeleri, eslenmis ornekler ve tam donem (e2_tablolari.py -> e2_ozet.csv, e2_tam_ozet.csv)

Kullanim:  python makale_tablolari.py <sonuclar klasoru> <makale/tablolar klasoru>
"""
import sys
from pathlib import Path

import pandas as pd

LABEL = {"ridge": "Ridge", "lightgbm": "LightGBM", "gru": "GRU", "lstm": "LSTM", "tcn": "TCN", "bilstm": "BiLSTM",
         "cnn_gru": "CNN-GRU", "cnn_lstm": "CNN-LSTM", "simplernn": "Simple RNN", "transformer": "Transformer",
         "dense": "Dense (MLP)", "cnn_lstm_gru": "CNN-LSTM-GRU", "cnn": "CNN", "smart_persistence": "Clear-sky persistence",
         "persistence": "Persistence", "climatology": "Climatology", "seasonal_persistence": "24 h seasonal persistence"}
CELLS = [("S1", 1), ("S1", 24), ("S2", 1), ("S2", 24)]


def markdown(frame: pd.DataFrame) -> str:
    head = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule] + body) + "\n"


def thousands(value) -> str:
    return f"{int(value):,}"


def table_plants(results: Path) -> str:
    t = pd.read_csv(results / "tablo_santraller.csv", encoding="utf-8-sig")
    own = t["protocol"] == "own"
    out = pd.DataFrame({
        "No.": t["no"],
        "System": t["name"] + own.map({True: " †", False: ""}),
        "Country": t["country"],
        "Source ID": t["source_id"].astype(str) + t["source"].map({"PVOutput": "", "EPİAŞ": " ‡"}),
        "Capacity (kW)": [f"{v:,.2f}".rstrip("0").rstrip(".") for v in t["capacity_kw"]],
        "Climate": t["koppen"],
        "Raw resolution (min)": t["raw_resolution_min"].astype(int),
        "First valid day": t["first_valid"],
        "Valid hours": t["hours_valid"].map(thousands),
        "Missing (%)": t["missing_share_pct"].map("{:.1f}".format),
        "Test coverage (%)": t["test_coverage_pct"].map("{:.1f}".format),
        "Training / test samples": t["n_train_S1_h1"].map(thousands) + " / " + t["n_test_S1_h1"].map(thousands)})
    return markdown(out)


def table_main(results: Path) -> str:
    s = pd.read_csv(results / "e1_ozet.csv")
    s = s[s[["info", "horizon"]].apply(tuple, axis=1).isin(CELLS)]
    order = s.groupby("model")["rank_all"].mean().sort_values().index
    rows = []
    for model in order:
        part = s[s["model"] == model].set_index(["info", "horizon"])
        row = {"Method": LABEL[model], "Type": "network" if part["kind"].iloc[0] == "deep" else "reference"}
        for info, h in CELLS:
            r = part.loc[(info, h)]
            row[f"{info}, h = {h} h"] = f"{r['nrmse_mean']:.2f} ({r['rank_all']:.1f})"
        row["Mean rank"] = f"{part['rank_all'].mean():.1f}"
        rows.append(row)
    return markdown(pd.DataFrame(rows))


def table_information(results: Path) -> str:
    parts = []
    for name, title in (("e2_ozet.csv", "Matched samples"), ("e2_tam_ozet.csv", "Full test period")):
        s = pd.read_csv(results / name)
        wide = s.pivot_table(index=["horizon", "model"], columns="info", values="nrmse_mean")
        order = [c for c in ("S0m", "S1m", "S2", "S12", "S3m", "S0", "S1", "S3") if c in wide.columns]
        models = [m for m in ("ridge", "lightgbm", "gru", "tcn", "lstm", "cnn_gru") if m in wide.index.get_level_values("model")]
        rows = [{"Horizon": f"{h} h", "Method": LABEL[m], **{c: f"{wide.loc[(h, m), c]:.2f}" for c in order}}
                for h in (1, 24) for m in models]
        parts.append(f"**{title}**\n\n" + markdown(pd.DataFrame(rows)))
    return "\n".join(parts)


def main():
    results, target = Path(sys.argv[1]), Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    for name, build in (("tablo1_santraller.md", table_plants), ("tablo2_ana_kiyas.md", table_main),
                        ("tablo_e2.md", table_information)):
        text = build(results)
        (target / name).write_text(text, encoding="utf-8")
        print(f"--- {name}\n{text}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
