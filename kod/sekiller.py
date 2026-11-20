"""Makale sekilleri: sonuclar/*.csv ve kosularin kaydettigi test tahminlerinden uretilir (makale/sekiller/).

    sekil_tasarim         Sekil 1: isleme zinciri ve bir tahminin girdileri (h = 24 saat)
    sekil_siralar         Sekil 2, E4: 17 yontemin dort hucredeki ortalama sirasi ve Nemenyi kritik farki
    sekil_bilgi           Sekil 3, E2: bilgi kumelerine gore ortalama test nRMSE (eslenmis ornekler), h = 1 ve 24 saat
    sekil_ornek           Sekil 4: olculen ve tahmin edilen guc, mutlak hata: iklim grubu basina bir santral, bir hafta
    sekil_ufuk            Sekil 5, E9: ufka gore ortalama test nRMSE (S1, S2) ve S2 - S1 farki, %95 bootstrap araligi ile
    sekil_boyut           Sekil 6, E3: on bir agin ayarli boyutta ve yaklasik 50 000 parametrede ortalama test nRMSE'si
    sekil_eslifark        Ek Sekil S1: santral basina eslesmis farklar
    sekil_mimari          Ek Sekil S2: paralel CNN-LSTM-GRU melezi
    sekil_onem            Ek Sekil S3, E5: girdi gruplarinin permutasyon onemi
    sekil_gorulmemis      Ek Sekil S4, E6: yerel model ve havuz modelleri
    sekil_tum_santraller  Ek Sekil S5: 25 santralin hepsinde bir test haftasi, gun oncesi tahmin

Her sekil PDF (vektor) ve PNG (300 dpi) olarak yazilir.  Genislik 180 mm (iki sutun); yazi Arial 6-9 pt.
Dergi kurallari: son boyutta en ince cizgi 1 pt (THIN; isaret kenarlari da), etiketlerde yalniz ilk harf buyuk,
binlik ayiraci yalniz bes ve daha cok basamakli sayilarda (fmt_int).
Renkler: kimlik icin uc renk (mavi = sinir agi, turuncu = referans yontem, yesil = hibrit) + notr murekkep;
her seri ayrica isaret bicimi ya da cizgi turuyle ayrilir (yalniz renge dayanmaz).

Ornek santrallerin secim kurali (sonuclara bakilmadan): her iklim ana grubunda (A, C, D) test donemi kapsamasi en
yuksek santral; esitlikte gecerli saat sayisi fazla olan.  Hafta: 14-20 Nisan 2025 (ortak test doneminin ortasi).

Kullanim:  python sekiller.py <cikti> <sonuclar klasoru> <makale/sekiller klasoru> [sekil adlari: tasarim mimari
           siralar eslifark bilgi ufuk boyut onem gorulmemis tumsantraller ornek; verilmezse hepsi]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MM = 1 / 25.4
THIN = 1.0                    # pt: thinnest stroke at final size (no line, tick or marker edge below one point)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LABEL = {"ridge": "Ridge", "lightgbm": "LightGBM", "gru": "GRU", "lstm": "LSTM", "tcn": "TCN", "bilstm": "BiLSTM",
         "cnn_gru": "CNN-GRU", "cnn_lstm": "CNN-LSTM", "simplernn": "Simple RNN", "transformer": "Transformer",
         "dense": "Dense (MLP)", "cnn_lstm_gru": "CNN-LSTM-GRU", "cnn": "CNN", "smart_persistence": "Clear-sky persistence",
         "persistence": "Persistence", "climatology": "Climatology", "seasonal_persistence": "24 h seasonal persistence"}
REFERENCES = {"persistence", "seasonal_persistence", "smart_persistence", "climatology", "ridge", "lightgbm"}
CELLS = [("S1", 1), ("S1", 24), ("S2", 1), ("S2", 24)]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": THIN, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": THIN,
    "xtick.major.width": THIN, "ytick.major.width": THIN, "xtick.minor.width": THIN, "ytick.minor.width": THIN,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.5, "lines.markeredgewidth": THIN,
    "patch.linewidth": THIN, "hatch.linewidth": THIN,
    "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150, "savefig.dpi": 300, "legend.frameon": False})


def fmt_int(n) -> str:
    """An integer in figure text: thousands separator only from five digits on (2369, 49,249, 179,393)."""
    n = int(n)
    return f"{n:,}" if abs(n) >= 10000 else str(n)


def save(fig, target: Path, name: str):
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / f"{name}.pdf")
    fig.savefig(target / f"{name}.png")
    plt.close(fig)
    print(f"yazildi: {target / name}.pdf / .png")


def panel(ax, letter: str):
    """Bold lower-case panel label, on the baseline of the panel title and just left of it."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-5, 6), textcoords="offset points",
                ha="right", va="baseline", fontsize=9, fontweight="bold", color=INK, annotation_clip=False)


def style(model: str) -> dict:
    """Identity: colour and marker shape together."""
    if model == "cnn_lstm_gru":
        return {"color": AQUA, "marker": "D"}
    if model in REFERENCES:
        return {"color": ORANGE, "marker": "s"}
    return {"color": BLUE, "marker": "o"}


def figure_ranks(results: Path, target: Path):
    ranks = pd.read_csv(results / "e4_ortalama_siralar.csv")
    ranks = ranks[ranks["set"] == "all17"]
    cd = pd.read_csv(results / "e4_friedman.csv")
    cd = cd[cd["set"] == "all17"].set_index(["info", "horizon"])["critical_difference"]
    order = ranks.groupby("model")["mean_rank"].mean().sort_values().index.tolist()
    y = {m: i for i, m in enumerate(order)}
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 88 * MM), sharey=True)
    for ax, (info, h) in zip(axes, CELLS):
        part = ranks[(ranks["info"] == info) & (ranks["horizon"] == h)].set_index("model")["mean_rank"]
        best = part.min()
        ax.axvspan(best, best + cd.loc[(info, h)], color="#f0efec", lw=0, zorder=0)
        ax.grid(axis="x", zorder=1)
        ax.hlines([y[m] for m in order], 1, [part[m] for m in order], color=GRID, lw=THIN, zorder=1)
        for m in order:
            s = style(m)
            ax.plot(part[m], y[m], s["marker"], color=s["color"], ms=4.4, mew=0, zorder=3)
        ax.set_xlim(1, 17)
        ax.set_xticks([1, 5, 9, 13, 17])
        ax.set_title(f"{info}, h = {h} h", loc="left", color=INK)
        panel(ax, "abcd"[CELLS.index((info, h))])
        ax.set_xlabel("Mean rank (1 = best)")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LABEL[m] for m in order])
    for tick, m in zip(axes[0].get_yticklabels(), order):
        tick.set_color(INK)
        if m == "cnn_lstm_gru":
            tick.set_fontweight("bold")
    axes[0].set_ylim(len(order) - 0.4, -0.6)
    handles = [plt.Line2D([], [], ls="", marker="s", color=ORANGE, ms=4.4, mew=0, label="Reference method"),
               plt.Line2D([], [], ls="", marker="o", color=BLUE, ms=4.4, mew=0, label="Neural network"),
               plt.Line2D([], [], ls="", marker="D", color=AQUA, ms=4.4, mew=0, label="CNN-LSTM-GRU hybrid"),
               plt.Rectangle((0, 0), 1, 1, color="#f0efec", label="Within the Nemenyi critical difference of the best method")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.185, right=0.99, top=0.93, bottom=0.2, wspace=0.12)
    save(fig, target, "sekil_siralar")


def figure_pairs(results: Path, target: Path):
    """Paired differences per plant for the comparisons the conclusions rest on (one dot = one plant)."""
    cell = pd.read_csv(results / "e1_hucreler.csv")
    tests = pd.read_csv(results / "e4_wilcoxon.csv")
    source = results / "tablo_santraller.csv"
    if source.is_file():
        plants = pd.read_csv(source, encoding="utf-8-sig").set_index("no")
        own = set(plants.index[plants["protocol"] == "own"])
    else:  # public deposit: the plant inventory is not deposited; the short-record plants are those of e1_kisa_kayit
        own = set(pd.read_csv(results / "e1_kisa_kayit.csv")["plant"].unique())
    pairs = [("cnn_lstm_gru", "gru"), ("cnn_lstm_gru", "cnn_lstm"), ("cnn_lstm_gru", "lightgbm"),
             ("gru", "lightgbm"), ("gru", "ridge")]
    short = {"cnn_lstm_gru": "Hybrid", "gru": "GRU", "cnn_lstm": "CNN-LSTM", "lightgbm": "LightGBM", "ridge": "Ridge"}
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 70 * MM), sharey=True, sharex=True)
    for ax, (info, h) in zip(axes, CELLS):
        scores = cell[(cell["info"] == info) & (cell["horizon"] == h)].pivot(index="plant", columns="model", values="nrmse")
        t = tests[(tests["info"] == info) & (tests["horizon"] == h)]
        ax.axvline(0, color=AXIS, lw=THIN, zorder=1)
        ax.grid(axis="x", zorder=0)
        for i, (a, b) in enumerate(pairs):
            diff = scores[a] - scores[b]
            jitter = ((np.arange(len(diff)) * 7) % 11 - 5) / 5 * 0.24            # fixed pattern, no random numbers
            is_own = np.array([p in own for p in diff.index])
            ax.plot(diff[~is_own], i + jitter[~is_own], "o", color=BLUE, ms=3.6, mec="white", mew=THIN, alpha=0.9, zorder=2)
            ax.plot(diff[is_own], i + jitter[is_own], "o", mfc="white", mec=BLUE, ms=3.2, mew=THIN, zorder=2)
            row = t[(t["reference"] == a) & (t["model"] == b)]
            if len(row):
                lo, hi = float(row["mean_ci_lo"].iloc[0]), float(row["mean_ci_hi"].iloc[0])
            else:
                row = t[(t["reference"] == b) & (t["model"] == a)]
                lo, hi = -float(row["mean_ci_hi"].iloc[0]), -float(row["mean_ci_lo"].iloc[0])
            ax.plot([lo, hi], [i + 0.4, i + 0.4], color=INK, lw=1.2, zorder=3, solid_capstyle="butt")
            ax.plot(diff.mean(), i + 0.4, "D", color=INK, ms=3.2, zorder=4)
        ax.set_title(f"{info}, h = {h} h", loc="left", color=INK)
        ax.set_xlabel("Difference in nRMSE (points)")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(pairs)))
    axes[0].set_yticklabels([f"{short[a]} − {short[b]}" for a, b in pairs])
    for tick in axes[0].get_yticklabels():
        tick.set_color(INK)
    axes[0].set_ylim(len(pairs) - 0.35, -0.5)
    handles = [plt.Line2D([], [], ls="", marker="o", color=BLUE, ms=3.6, mec="white", mew=THIN, label="Plant (common protocol)"),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec=BLUE, mew=THIN, ms=3.2, label="Plant with a short record"),
               plt.Line2D([], [], ls="-", marker="D", color=INK, ms=3.2, lw=1.2, label="Mean and 95% bootstrap interval")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.8)
    fig.subplots_adjust(left=0.147, right=0.99, top=0.91, bottom=0.27, wspace=0.1)
    save(fig, target, "sekil_eslifark")


def figure_information(results: Path, target: Path):
    s = pd.read_csv(results / "e2_ozet.csv")
    order = ["S0m", "S1m", "S2", "S12", "S3m"]
    names = ["No\nweather", "+ Weather\nhistory", "+ NWP\nforecast", "+ History\nand forecast", "+ Perfect\nforecast"]
    models = [("ridge", "s", "-"), ("lightgbm", "D", "-"), ("gru", "o", "-"), ("tcn", "^", "-"), ("lstm", "v", "-"),
              ("cnn_gru", "P", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM))
    for ax, h in zip(axes, (1, 24)):
        wide = s[s["horizon"] == h].pivot(index="model", columns="info", values="nrmse_mean")[order]
        ax.grid(axis="y", zorder=0)
        for m, marker, ls in models:
            colour = ORANGE if m in REFERENCES else BLUE
            ax.plot(range(len(order)), wide.loc[m], ls, color=colour, lw=1.1, marker=marker, ms=5.0, mec="white",
                    mew=THIN, label=LABEL[m], zorder=4 if m in REFERENCES else 3, clip_on=False)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{c}\n{n}" for c, n in zip(order, names)])
        ax.set_xlim(-0.3, len(order) - 0.7)
        ax.set_title(f"h = {h} h", loc="left", color=INK)
        panel(ax, "ab"[(1, 24).index(h)])
        ax.set_ylabel("Mean test nRMSE (% of rated capacity)")
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylim(4.5, 7.0)
    axes[1].set_ylim(6.0, 11.5)
    axes[1].legend(loc="upper right", ncol=2, handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.2, wspace=0.2)
    save(fig, target, "sekil_bilgi")


def example_plants(results: Path) -> list:
    t = pd.read_csv(results / "tablo_santraller.csv", encoding="utf-8-sig")
    t["group"] = t["koppen"].str[0]
    t = t[t["protocol"] == "common"].sort_values(["test_coverage_pct", "hours_valid"], ascending=False)
    chosen = t.groupby("group").head(1).sort_values("group")
    return [(int(r.no), r.name, r.country, r.capacity_kw, r.koppen) for r in chosen.itertuples()]


def load_predictions(out: Path, plant: int, horizon: int, start, end) -> pd.DataFrame:
    folder = out / "ana_icon" / "pred" / f"{plant:02d}_h{horizon}_S2"
    frame = pd.read_parquet(folder / "target.parquet")[["y"]].rename(columns={"y": "measured"})
    for model in ("ridge", "gru", "smart_persistence"):
        frame[model] = np.load(folder / f"{model}_s0.npy")
    hours = pd.date_range(start, end, freq="h", tz="UTC", inclusive="right")
    return 100.0 * frame.reindex(hours)                           # per cent of rated capacity; gaps stay empty


def figure_examples(out: Path, results: Path, target: Path):
    start, end = pd.Timestamp("2025-04-14", tz="UTC"), pd.Timestamp("2025-04-21", tz="UTC")
    plants = example_plants(results)
    series = [("measured", INK, "-", 1.4, "Measured"), ("smart_persistence", MUTED, (0, (3, 1.5)), THIN, "Clear-sky persistence"),
              ("ridge", ORANGE, "-", THIN, "Ridge"), ("gru", BLUE, "-", THIN, "GRU (seed 0)")]
    fig, axes = plt.subplots(3, len(plants), figsize=(180 * MM, 135 * MM), sharex=True)
    for j, (no, name, country, capacity, koppen) in enumerate(plants):
        for i, h in enumerate((1, 24)):
            data, ax = load_predictions(out, no, h, start, end), axes[i, j]
            ax.grid(axis="y", zorder=0)
            for z, (column, colour, ls, lw, label) in enumerate(series):
                ax.plot(data.index, data[column], color=colour, ls=ls, lw=lw, label=label, zorder=2 + z)
            ax.set_ylim(0, None)
            if j == 0:
                ax.set_ylabel(f"h = {h} h\nPower (% of rated capacity)")
            if i == 0:
                size = f"{capacity / 1000:.2f} MW" if capacity >= 1000 else f"{capacity:g} kW"
                ax.set_title(f"{name}\n{country}, {size}, climate {koppen}", loc="left", color=INK, linespacing=1.25)
        data, ax = load_predictions(out, no, 24, start, end), axes[2, j]
        ax.grid(axis="y", zorder=0)
        for column, colour, label in (("ridge", ORANGE, "Ridge"), ("gru", BLUE, "GRU (seed 0)")):
            ax.plot(data.index, (data[column] - data["measured"]).abs(), color=colour, lw=THIN, label=label)
        ax.set_ylim(0, None)
        if j == 0:
            ax.set_ylabel("h = 24 h\nAbsolute error (% of rated capacity)")
        ax.set_xlim(start, end)
        ax.set_xticks(pd.date_range(start, end, freq="D"))
        ax.set_xticklabels([d.strftime("%d %b") if k % 2 == 0 else "" for k, d in enumerate(pd.date_range(start, end, freq="D"))])
        ax.set_xlabel("2025, UTC")
    for k, ax in enumerate(axes.flat):                      # rows: h = 1 h, h = 24 h, error; columns: plants
        panel(ax, "abcdefghi"[k])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), handlelength=2.4, columnspacing=1.8)
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.12, wspace=0.16, hspace=0.24)
    save(fig, target, "sekil_ornek")
    print("ornek santraller:", [(no, name) for no, name, *_ in plants])


def figure_all_plants(out: Path, results: Path, target: Path):
    """Day-ahead forecasts (S2) at all 25 plants for the week shown in Fig. 4 (one year later at the short-record plants)."""
    t = pd.read_csv(results / "tablo_santraller.csv", encoding="utf-8-sig").sort_values("no")
    fig, axes = plt.subplots(5, 5, figsize=(180 * MM, 182 * MM))
    assert len(t) == axes.size
    series = [("measured", INK, 1.1, "Measured"), ("ridge", ORANGE, THIN, "Ridge"), ("gru", BLUE, THIN, "GRU (seed 0)")]
    for k, (ax, r) in enumerate(zip(axes.ravel(), t.itertuples())):
        year = 2025 if r.protocol == "common" else 2026
        start, end = pd.Timestamp(f"{year}-04-14", tz="UTC"), pd.Timestamp(f"{year}-04-21", tz="UTC")
        data = load_predictions(out, int(r.no), 24, start, end)
        ax.grid(axis="y", zorder=0)
        for z, (column, colour, lw, label) in enumerate(series):
            ax.plot(data.index, data[column], color=colour, lw=lw, label=label, zorder=2 + z)
        ax.set_ylim(0, None)
        ax.set_xlim(start, end)
        days = pd.date_range(start, end, freq="D")
        ax.set_xticks(days)
        if k >= axes.size - axes.shape[1]:                  # bottom row: dates (the same days in 2025 and 2026)
            ax.set_xticklabels([d.strftime("%d %b") if i % 2 == 0 else "" for i, d in enumerate(days)])
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="both", length=2, labelsize=6)
        name = r.name if len(r.name) <= 22 else r.name[:21] + "…"
        ax.set_title(f"{int(r.no)}  {name}{'' if r.protocol == 'common' else ' †'}", loc="left", color=INK, fontsize=6.5, pad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Power (%)", fontsize=6.5)
    axes[-1, axes.shape[1] // 2].set_xlabel("Date (UTC; † plants: 2026)", fontsize=6.5, labelpad=3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), handlelength=2.4, columnspacing=2.0)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.975, bottom=0.085, wspace=0.26, hspace=0.3)
    save(fig, target, "sekil_tum_santraller")


def figure_design(target: Path):
    """Processing chain (a) and the time line of the inputs of one forecast for h = 24 h (b)."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    height = 130.0                                          # mm; panel a keeps its size of 180 x 53.4 mm
    fig = plt.figure(figsize=(180 * MM, height * MM))
    ax = fig.add_axes([0.0, 1 - 53.4 / height, 1.0, 53.4 / height])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, title, body, fill="#f6f5f2"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=AXIS, lw=THIN))
        ax.text(x + 1.8, y + h - 2.0, title, va="top", ha="left", fontsize=7.2, fontweight="bold", color=INK)
        ax.text(x + 1.8, y + h - 6.2, body, va="top", ha="left", fontsize=6.3, color=INK2, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": THIN,
                                                                  "shrinkA": 0, "shrinkB": 0, "mutation_scale": 7})

    ax.text(0.5, 51.5, "a", fontsize=9, fontweight="bold", va="top")
    top, h1, w1, gap = 29.5, 19.5, 40.0, 4.5
    xs = [5 + i * (w1 + gap) for i in range(4)]
    box(xs[0], top, w1, h1, "Raw records", "PVOutput: energy counter every\n5 or 15 min, local time\nEPİAŞ: metered hourly energy")
    box(xs[1], top, w1, h1, "Hourly target", "Energy integrated over UTC hours,\nlabel = end of the hour,\ndivided by rated capacity")
    box(xs[2], top, w1, h1, "Quality control", "Incomplete or implausible hours,\npower at night, stuck values\n→ missing; nothing filled or smoothed")
    box(xs[3], top, w1, h1, "Samples", "24 h input window and target p(t + h);\nwindows touching a missing hour\nare dropped; weather, clear-sky and\n"
                                       "day-of-year channels added (panel b)")
    for i in range(3):
        arrow(xs[i] + w1, top + h1 / 2, xs[i + 1], top + h1 / 2)
    low, h2, w2, gap2 = 3.0, 17.5, 31.6, 3.5
    xl = [5 + i * (w2 + gap2) for i in range(5)]
    box(xl[0], low, w2, h2, "Chronological split", "Training | validation\n(3 months) | test (12 months);\ngap of L + h − 1 hours")
    box(xl[1], low, w2, h2, "Tuning", "12 configurations per\narchitecture, 6 plants,\nvalidation RMSE only")
    box(xl[2], low, w2, h2, "Training", "Each plant, horizon and\ninformation set;\nnetworks with 5 seeds")
    box(xl[3], low, w2, h2, "Post-processing", "Limit to 0–1.15, zero when\nthe sun is down; identical\nfor all 17 methods")
    box(xl[4], low, w2, h2, "Scores and tests", "Per plant on test hours;\nFriedman, Wilcoxon (Holm),\nDiebold–Mariano")
    for i in range(4):
        arrow(xl[i] + w2, low + h2 / 2, xl[i + 1], low + h2 / 2)
    ax.plot([xs[3] + w1 / 2, xs[3] + w1 / 2, xl[0] + w2 / 2], [top, 25.0, 25.0], color=MUTED, lw=THIN)
    arrow(xl[0] + w2 / 2, 25.0, xl[0] + w2 / 2, low + h2)

    bx = fig.add_axes([0.335, 19.8 / height, 0.645, 46.0 / height])
    fig.text(0.003, (height - 55.8) / height, "b", fontsize=9, fontweight="bold", va="top")
    fig.text(0.03, (height - 56.16) / height, "Inputs of one forecast (h = 24 h)", fontsize=7.5, va="top", color=INK)
    rows = [("Target p(t + h)", [(23, 24)], "target"),
            ("Past power", [(-24, 0)], "available"),
            ("Clear-sky irradiance (computed)", [(-24, 0), (0, 24)], "computed"),
            ("Day of year, d(τ + h)", [(0, 24)], "computed"),
            ("S1: weather analysis, window hours", [(-24, 0)], "delayed"),
            ("S2: ICON forecast for the hours τ + h,\neach predicted 24 h before it is valid", [(0, 24)], "available"),
            ("S3: weather analysis for the hours τ + h\n(upper bound)", [(0, 24)], "unavailable")]
    look = {"target": {"fc": INK, "ec": INK}, "available": {"fc": BLUE, "ec": BLUE},
            "computed": {"fc": "#c3c2b7", "ec": "#c3c2b7"}, "delayed": {"fc": "#9ec5f4", "ec": "#9ec5f4"},
            "unavailable": {"fc": "white", "ec": MUTED, "hatch": "////"}}
    bx.axvspan(0, 24.6, color="#f6f5f2", lw=0, zorder=0)
    for i, (label, spans, kind) in enumerate(rows):
        for lo, hi in spans:
            bx.add_patch(Rectangle((lo, i - 0.28), hi - lo, 0.56, lw=THIN, zorder=3, **look[kind]))
        bx.text(-25.2, i, label, ha="right", va="center", fontsize=7, color=INK, linespacing=1.15)
    bx.axvline(0, color=INK, lw=THIN, zorder=4)
    bx.text(0, -0.86, "Forecast origin t", ha="center", va="bottom", fontsize=7, color=INK)   # just above the axes
    bx.text(-12, -0.86, "Input window (L = 24 h)", ha="center", va="bottom", fontsize=7, color=INK2)
    bx.text(12.3, -0.86, "After the origin", ha="center", va="bottom", fontsize=7, color=INK2)
    bx.set_xlim(-24.6, 24.6)
    bx.set_ylim(len(rows) - 0.45, -0.8)
    bx.set_yticks([])
    bx.set_xticks([-24, -18, -12, -6, 0, 6, 12, 18, 24])
    bx.set_xticklabels(["t − 24", "t − 18", "t − 12", "t − 6", "t", "t + 6", "t + 12", "t + 18", "t + 24"])
    bx.set_xlabel("Time (hours); every hourly value refers to the hour that ends at its label")
    bx.spines["left"].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, lw=THIN, **look["available"], label="Available at the origin"),
               Rectangle((0, 0), 1, 1, lw=THIN, **look["delayed"], label="Published with a delay"),
               Rectangle((0, 0), 1, 1, lw=THIN, **look["computed"], label="Computed, always known"),
               Rectangle((0, 0), 1, 1, lw=THIN, **look["unavailable"], label="Not available at the origin")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=1.6)
    save(fig, target, "sekil_tasarim")


def figure_architecture(results: Path, target: Path):
    """The parallel CNN-LSTM-GRU hybrid (layer sizes for width multiplier 1, read from the layer table)."""
    from matplotlib.patches import FancyBboxPatch

    layers_ = pd.read_csv(results / "ek_mimari_katmanlar.csv")
    hybrid = layers_[layers_["model"] == "cnn_lstm_gru"]
    size = {t: hybrid.loc[hybrid["type"] == t, "output_shape"].iloc[0] for t in ("Conv1D", "LSTM", "GRU", "Concatenate")}
    dense = hybrid.loc[hybrid["type"] == "Dense", "output_shape"].tolist()
    total = int(hybrid["parameters"].sum())
    assert (size["Conv1D"], size["LSTM"], size["GRU"], size["Concatenate"], dense) == ("24 x 32", "32", "32", "96", ["64", "32", "1"])

    fig = plt.figure(figsize=(180 * MM, 62 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, title, body="", fill="#f6f5f2", edge=AXIS):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fill, ec=edge, lw=THIN))
        ax.text(x + w / 2, y + h / 2 + (1.5 if body else 0), title, ha="center", va="center", fontsize=6.8,
                fontweight="bold", color=INK)
        if body:
            ax.text(x + w / 2, y + h / 2 - 2.0, body, ha="center", va="center", fontsize=6.0, color=INK2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": THIN,
                                                                  "shrinkA": 0, "shrinkB": 0, "mutation_scale": 7})

    h, rows = 9.0, {"cnn": 45.0, "lstm": 26.5, "gru": 8.0}
    box(3, 22.5, 22, 17, "Input window", "24 h × F channels")
    for y in rows.values():
        ax.plot([25, 29, 29], [31, 31, y + h / 2], color=MUTED, lw=THIN)
        arrow(29, y + h / 2, 33, y + h / 2)
    box(33, rows["cnn"], 25, h, "Conv1D", "32 filters, kernel 2, ReLU", fill="#e9f1fc", edge="#9ec5f4")
    box(62, rows["cnn"], 19, h, "Batch norm.", "", fill="#e9f1fc", edge="#9ec5f4")
    box(85, rows["cnn"], 25, h, "Global max pooling", "Over time → 32", fill="#e9f1fc", edge="#9ec5f4")
    arrow(58, rows["cnn"] + h / 2, 62, rows["cnn"] + h / 2)
    arrow(81, rows["cnn"] + h / 2, 85, rows["cnn"] + h / 2)
    for name, label in (("lstm", "LSTM"), ("gru", "GRU")):
        box(33, rows[name], 25, h, label, "32 units, last state", fill="#e9f1fc", edge="#9ec5f4")
        box(62, rows[name], 19, h, "Batch norm.", "→ 32", fill="#e9f1fc", edge="#9ec5f4")
        arrow(58, rows[name] + h / 2, 62, rows[name] + h / 2)
    box(116, 22.5, 15.5, 17, "Concatenate", "96")
    ax.plot([110, 113, 113], [rows["cnn"] + h / 2, rows["cnn"] + h / 2, 35], color=MUTED, lw=THIN)
    arrow(113, 35, 116, 35)
    arrow(81, rows["lstm"] + h / 2, 116, rows["lstm"] + h / 2)
    ax.plot([81, 113, 113], [rows["gru"] + h / 2, rows["gru"] + h / 2, 27], color=MUTED, lw=THIN)
    arrow(113, 27, 116, 27)
    x, previous = 135.0, 131.5
    for title, body, w in (("Dense", "64, ReLU\ndropout 0.3", 13), ("Dense", "32, ReLU\ndropout p", 13), ("Dense", "1\nforecast", 10.5)):
        ax.add_patch(FancyBboxPatch((x, 22.5), w, 17, boxstyle="round,pad=0,rounding_size=1.0", fc="#f6f5f2", ec=AXIS, lw=THIN))
        ax.text(x + w / 2, 35.0, title, ha="center", va="center", fontsize=6.8, fontweight="bold", color=INK)
        ax.text(x + w / 2, 28.5, body, ha="center", va="center", fontsize=6.0, color=INK2, linespacing=1.3)
        arrow(previous, 31, x, 31)
        previous = x + w
        x += w + 3.5
    ax.text(3, 59, f"Parallel CNN–LSTM–GRU hybrid: {fmt_int(total)} parameters with 12 input channels and width multiplier 1",
            fontsize=7.2, color=INK, va="center")
    save(fig, target, "sekil_mimari")


def figure_horizons(results: Path, target: Path):
    """E9: error against forecast horizon for both information sets (a, b), and the change brought by forecast inputs
    with the 95% bootstrap interval of each mean paired difference (c; full width so that the intervals stay apart)."""
    s = pd.read_csv(results / "e9_ozet.csv")
    t = pd.read_csv(results / "e9_wilcoxon.csv")
    t = t[t["comparison"] == "S2 - S1"]
    hours = [1, 2, 3, 6, 12, 18, 24]
    models = [("ridge", "s"), ("lightgbm", "D"), ("gru", "o"), ("tcn", "^"), ("lstm", "v"), ("cnn_gru", "P")]
    fig = plt.figure(figsize=(180 * MM, 126 * MM))
    grid = fig.add_gridspec(2, 2, left=0.065, right=0.99, top=0.95, bottom=0.085, wspace=0.17, hspace=0.42)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    for ax, info, title, letter in zip(axes, ("S1", "S2"), ("S1: retrospective weather", "S2: weather forecast"), "ab"):
        panel(ax, letter)
        wide = s[s["info"] == info].pivot(index="model", columns="horizon", values="nrmse_mean")[hours]
        ax.grid(axis="y", zorder=0)
        for m, ls, text, at in (("smart_persistence", (0, (4, 2)), "Clear-sky\npersistence", (3.7, 9.25)),
                                ("climatology", (0, (1, 1.5)), "Climatology", (13.5, 9.2))):
            ax.plot(hours, wide.loc[m], color=MUTED, lw=THIN, ls=ls, zorder=2)
            ax.text(*at, text, color=INK2, fontsize=6.5, va="bottom", ha="left", linespacing=0.95)
        for m, marker in models:
            ax.plot(hours, wide.loc[m], "-", color=ORANGE if m in REFERENCES else BLUE, lw=1.1, marker=marker, ms=4.8,
                    mec="white", mew=THIN, label=LABEL[m], zorder=4 if m in REFERENCES else 3)
        ax.set_ylim(4.5, 10.0)
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylabel("Mean test nRMSE (% of rated capacity)")
    axes[0].legend(loc="lower right", ncol=2, handlelength=2.0, columnspacing=1.2, borderaxespad=0.2)
    ax = axes[2]
    ax.grid(axis="y", zorder=0)
    ax.axhline(0, color=INK2, lw=THIN, zorder=2)
    mean, lo, hi = (t.pivot(index="a_model", columns="horizon", values=v)[hours] for v in ("mean_diff", "mean_ci_lo", "mean_ci_hi"))
    assert ((lo <= mean) & (mean <= hi)).all().all()
    shift = np.linspace(-0.38, 0.38, len(models))          # hours; methods side by side so that the intervals stay apart
    for (m, marker), dx in zip(models, shift):
        colour = ORANGE if m in REFERENCES else BLUE
        x = np.array(hours) + dx
        ax.errorbar(x, mean.loc[m], yerr=[mean.loc[m] - lo.loc[m], hi.loc[m] - mean.loc[m]], fmt="none", ecolor=colour,
                    elinewidth=THIN, capsize=1.2, capthick=THIN, zorder=3)
        ax.plot(x, mean.loc[m], "-", color=colour, lw=1.1, marker=marker, ms=3.8, mew=0, zorder=5 if m in REFERENCES else 4)
    key = ax.errorbar([np.nan], [np.nan], yerr=[[np.nan], [np.nan]], fmt="none", ecolor=INK2, elinewidth=THIN, capsize=1.6,
                      capthick=THIN)
    ax.legend([key], ["95% bootstrap interval"], loc="upper right", bbox_to_anchor=(1.0, 0.86), handlelength=1.0,
              handletextpad=0.5)
    ax.set_ylim(-2.3, 1.3)
    ax.set_yticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    ax.text(24.6, 1.22, "Forecast inputs raise the error", color=INK2, fontsize=6.5, ha="right", va="top")
    ax.text(0.4, -2.22, "Forecast inputs lower the error", color=INK2, fontsize=6.5, ha="left", va="bottom")
    ax.set_title("S2 minus S1", loc="left", color=INK)
    panel(ax, "c")
    ax.set_ylabel("Difference in nRMSE (points)")
    for ax in axes:
        ax.set_xticks(hours)
        ax.set_xlim(0, 25)
        ax.set_xlabel("Forecast horizon (h)")
    fig.align_ylabels([axes[0], axes[2]])
    save(fig, target, "sekil_ufuk")


def figure_matched(results: Path, target: Path):
    """E3: every network at its tuned size (open marker) and at about 50,000 parameters (filled marker)."""
    s = pd.read_csv(results / "e3_ozet.csv")
    e1 = pd.read_csv(results / "e1_ozet.csv")
    order = s.groupby("model")["rank_matched"].mean().sort_values().index.tolist()
    y = {m: i for i, m in enumerate(order)}
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 72 * MM), sharey=True)
    for ax, (info, h) in zip(axes, CELLS):
        part = s[(s["info"] == info) & (s["horizon"] == h)].set_index("model")
        ridge = e1[(e1["info"] == info) & (e1["horizon"] == h) & (e1["model"] == "ridge")]["nrmse_mean"].iloc[0]
        ax.grid(axis="x", zorder=0)
        ax.axvline(ridge, color=ORANGE, lw=THIN, ls=(0, (4, 2)), zorder=2)
        ax.text(ridge, len(order) - 0.45, " Ridge", color=INK2, fontsize=6.5, ha="left", va="center")
        for m in order:
            hybrid = m == "cnn_lstm_gru"
            colour = AQUA if hybrid else BLUE
            a, b = part.loc[m, "nrmse_tuned"], part.loc[m, "nrmse_matched"]
            ax.plot([a, b], [y[m], y[m]], color=colour, lw=THIN, zorder=3, solid_capstyle="round")
            ax.plot(a, y[m], "D" if hybrid else "o", ms=4.2 if hybrid else 4.6, mfc="white", mec=colour, mew=THIN, zorder=4)
            ax.plot(b, y[m], "D" if hybrid else "o", ms=4.8 if hybrid else 5.2, mfc=colour, mec="white", mew=THIN, zorder=5)
        ax.set_title(f"{info}, h = {h} h", loc="left", color=INK)
        panel(ax, "abcd"[CELLS.index((info, h))])
        ax.set_xlabel("Mean test nRMSE (%)")
        ax.set_ylim(len(order) - 0.1, -0.6)
        ax.tick_params(axis="y", length=0)
        lo, hi = min(ridge, part[["nrmse_tuned", "nrmse_matched"]].min().min()), part[["nrmse_tuned", "nrmse_matched"]].max().max()
        ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.06 * (hi - lo))
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LABEL[m] for m in order])
    low, high = int(s["params_tuned"].min()), int(s["params_tuned"].max())
    equal_low, equal_high = int(s["params_min_matched"].min()), int(s["params_max_matched"].max())
    handles = [plt.Line2D([], [], ls="", marker="o", ms=4.6, mfc="white", mec=BLUE, mew=THIN,
                          label=f"Tuned size ({fmt_int(low)}–{fmt_int(high)} parameters)"),
               plt.Line2D([], [], ls="", marker="D", ms=4.8, mfc=AQUA, mec="white", mew=THIN, label="CNN-LSTM-GRU hybrid"),
               plt.Line2D([], [], ls="", marker="o", ms=5.2, mfc=BLUE, mec="white", mew=THIN,
                          label=f"Equal size ({fmt_int(equal_low)}–{fmt_int(equal_high)} parameters)"),
               plt.Line2D([], [], color=ORANGE, lw=THIN, ls=(0, (4, 2)), label="Ridge regression")]
    fig.legend(handles=handles, loc="lower center", ncol=2, handlelength=2.2, columnspacing=2.4, bbox_to_anchor=(0.55, 0.0))
    fig.subplots_adjust(left=0.125, right=0.99, top=0.925, bottom=0.3, wspace=0.1)
    save(fig, target, "sekil_boyut")


def figure_importance(results: Path, target: Path):
    """E5: permutation importance of the input groups (mean increase of test nRMSE over the 25 plants)."""
    s = pd.read_csv(results / "e5_onem_ozet.csv")
    groups = ["pv", "geometry", "irradiance", "cloud", "temperature", "humidity", "wind"]
    names = ["Past power", "Solar geometry", "Irradiance", "Cloud cover", "Temperature", "Humidity", "Wind speed"]
    models = [("ridge", "s"), ("lightgbm", "D"), ("gru", "o"), ("cnn_lstm_gru", "D")]
    offsets = [0.27, 0.09, -0.09, -0.27]
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 62 * MM), sharey=True, sharex=True)
    for ax, (info, h) in zip(axes, CELLS):
        part = s[(s["info"] == info) & (s["horizon"] == h)].pivot(index="group", columns="model", values="delta_mean")
        ax.grid(axis="x", zorder=0)
        for (m, marker), dy in zip(models, offsets):
            colour = AQUA if m == "cnn_lstm_gru" else ORANGE if m in REFERENCES else BLUE
            rows = [i - dy for i in range(len(groups))]
            ax.hlines(rows, 0, part.loc[groups, m], color=colour, lw=THIN, zorder=3)
            ax.plot(part.loc[groups, m], rows, ls="", marker=marker, ms=4.3, mfc=colour, mec="white", mew=THIN,
                    label=LABEL[m], zorder=4)
        ax.set_title(f"{info}, h = {h} h", loc="left", color=INK)
        ax.set_xlabel("Increase in nRMSE (points)")
        ax.set_ylim(len(groups) - 0.45, -0.55)
        ax.set_xlim(-0.15, 7.6)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels(names)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, columnspacing=2.4, bbox_to_anchor=(0.55, 0.0))
    fig.subplots_adjust(left=0.115, right=0.99, top=0.91, bottom=0.27, wspace=0.1)
    save(fig, target, "sekil_onem")


def figure_unseen(results: Path, target: Path):
    """E6: local model, pooled model that has not seen the plant, pooled model that has (22 common-protocol plants)."""
    cell = pd.read_csv(results / "e6_hucreler.csv")
    cell = cell[cell["seen"].notna()]
    assert cell["plant"].nunique() == 22
    s = cell.groupby(["info", "horizon", "model"])[["local", "unseen", "seen"]].mean()
    order = ["ridge", "lightgbm", "gru", "cnn_lstm_gru"]
    kinds = [("local", "o", "Own model of the plant"), ("unseen", "s", "Pooled model, plant not in the training set"),
             ("seen", "^", "Pooled model, plant in the training set")]
    size = {"local": 4.6, "unseen": 5.0, "seen": 5.4}       # filled markers carry a white edge of THIN
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 50 * MM), sharey=True)
    for ax, (info, h) in zip(axes, CELLS):
        ax.grid(axis="x", zorder=0)
        for i, m in enumerate(order):
            colour = AQUA if m == "cnn_lstm_gru" else ORANGE if m in REFERENCES else BLUE
            row = s.loc[(info, h, m)]
            ax.plot([row.min(), row.max()], [i, i], color=colour, lw=THIN, zorder=3, solid_capstyle="round")
            for kind, marker, _ in kinds:
                face = "white" if kind == "local" else colour
                ax.plot(row[kind], i, marker, ms=size[kind], mfc=face, mec=colour if kind == "local" else "white", mew=THIN,
                        zorder=5 - (kind == "local"))
        ax.set_title(f"{info}, h = {h} h", loc="left", color=INK)
        ax.set_xlabel("Mean test nRMSE (%)")
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        part = s.loc[(info, h)]
        lo, hi = part.min().min(), part.max().max()
        ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LABEL[m] for m in order])
    handles = [plt.Line2D([], [], ls="", marker=marker, ms=size[kind], mfc="white" if kind == "local" else INK2,
                          mec=INK2 if kind == "local" else "white", mew=THIN, label=label)
               for kind, marker, label in kinds]
    fig.legend(handles=handles, loc="lower center", ncol=3, columnspacing=2.2, bbox_to_anchor=(0.55, 0.0))
    fig.subplots_adjust(left=0.125, right=0.99, top=0.89, bottom=0.33, wspace=0.1)
    save(fig, target, "sekil_gorulmemis")


def main():
    out, results, target = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    figures = {"tasarim": lambda: figure_design(target), "mimari": lambda: figure_architecture(results, target),
               "siralar": lambda: figure_ranks(results, target), "eslifark": lambda: figure_pairs(results, target),
               "bilgi": lambda: figure_information(results, target), "ufuk": lambda: figure_horizons(results, target),
               "boyut": lambda: figure_matched(results, target), "onem": lambda: figure_importance(results, target),
               "gorulmemis": lambda: figure_unseen(results, target),
               "tumsantraller": lambda: figure_all_plants(out, results, target),
               "ornek": lambda: figure_examples(out, results, target)}
    for name in sys.argv[4:] or figures:      # optional: names of the figures to draw (default: all)
        figures[name]()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
