"""Mimari tablolari (Ek): her sinir aginin katman katman tanimi ve secilen ayarlarla parametre sayilari.

    ek_mimari_katmanlar.csv   11 mimari, genislik carpani 1, 24 x 12 girdi: katman sirasi, turu, ayarlari, girdisi,
                              cikti boyutu, parametre sayisi (Keras modelinden okunur; elle yazilmaz)
    ek_mimari_parametre.csv   mimari x girdi kanal sayisi (5, 12, 19) x genislik (0.5, 1, 2): parametre sayisi
    ek_secilen_ayarlar.csv    ayar asamasinda secilen yapilandirma (bilgi kumesi x ufuk x mimari): genislik, dropout,
                              ogrenme orani, yigin boyutu, ortalama dogrulama RMSE'si, parametre sayisi

Modeller yalniz islemcide kurulur (GPU kullanilmaz).
Kullanim:  python mimari_tablolari.py <cikti> <sonuclar klasoru>
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""                      # build on CPU only; never touch the GPU of running jobs

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import models  # noqa: E402

LOOKBACK = 24
SHOW = ["units", "filters", "kernel_size", "dilation_rate", "padding", "activation", "rate", "num_heads", "key_dim"]
TUNING = {"S1": "ayar", "S2": "ayar_icon", "S0": "ayar_e2", "S3": "ayar_e2"}
CHANNELS = {"S0": 5, "S1": 12, "S2": 12, "S3": 12}


def describe(layer) -> str:
    cfg = layer.get_config()
    if layer.__class__.__name__ == "Bidirectional":
        inner = cfg["layer"]["config"]
        return f"LSTM units={inner['units']} in each direction, return_sequences={inner['return_sequences']}"
    parts = []
    for key in SHOW:
        if key in cfg and cfg[key] not in (None, "linear"):
            value = cfg[key]
            value = value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value
            if key == "dilation_rate" and value == 1:
                continue
            if key == "padding" and layer.__class__.__name__ != "Conv1D":
                continue
            parts.append(f"{key}={value}")
    if "return_sequences" in cfg:
        parts.append(f"return_sequences={cfg['return_sequences']}")
    return ", ".join(parts)


def index_text(key) -> str:
    """Text of an indexing operation such as x[:, -1, :]."""
    def one(k):
        if isinstance(k, slice):
            return ":" if (k.start, k.stop, k.step) == (None, None, None) else f"{k.start or ''}:{k.stop or ''}"
        return str(k)
    return "index [" + ", ".join(one(k) for k in (key if isinstance(key, tuple) else (key,))) + "]"


def layer_rows(name: str, n_features: int = 12) -> list:
    """All operations of the model graph in topological order: layers and the parameter-free operations between
    them (residual additions, selection of the last time step), so that every `input_from` name is a row."""
    model = models.build_model(name, LOOKBACK, n_features, 1.0, 0.2, compile_model=False)
    rows = []
    for i, op in enumerate(getattr(model, "operations", None) or model.layers):
        nodes = getattr(op, "_inbound_nodes", None) or []
        try:
            sources = [node.operation.name for node in nodes[0].parent_nodes]
        except (AttributeError, IndexError):
            sources = []
        try:
            shape = tuple(nodes[0].output_tensors[0].shape[1:])
        except (AttributeError, IndexError):
            shape = tuple(op.output.shape[1:]) if hasattr(op, "output") else ""
        is_layer = hasattr(op, "count_params")
        settings = describe(op) if is_layer else (index_text(op.key) if hasattr(op, "key") else "")
        rows.append({"model": name, "order": i, "layer": op.name, "type": op.__class__.__name__,
                     "settings": settings, "input_from": " + ".join(dict.fromkeys(sources)),
                     "output_shape": " x ".join(str(d) for d in shape) if shape else "",
                     "parameters": op.count_params() if is_layer else 0})
    assert sum(r["parameters"] for r in rows) == model.count_params()
    names = {r["layer"] for r in rows}
    missing = {s for r in rows for s in r["input_from"].split(" + ") if s and s not in names}
    assert not missing, f"{name}: girdisi listede olmayan islemler {missing}"
    return rows


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    layers = pd.DataFrame([r for name in models.MODEL_NAMES for r in layer_rows(name)])
    layers.to_csv(target / "ek_mimari_katmanlar.csv", index=False)

    counts = [{"model": name, "channels": c, "width": w, "parameters": models.count_parameters(name, LOOKBACK, c, w)}
              for name in models.MODEL_NAMES for c in (5, 12, 19) for w in (0.5, 1.0, 2.0)]
    counts = pd.DataFrame(counts)
    counts.to_csv(target / "ek_mimari_parametre.csv", index=False)

    chosen = []
    for info, folder in TUNING.items():
        path = out / folder / "selected.csv"
        if not path.exists():
            continue
        s = pd.read_csv(path)
        s = s[s["info"] == info]
        for r in s.itertuples():
            chosen.append({"info": info, "horizon": int(r.horizon), "model": r.model, "width": float(r.width),
                           "dropout": float(r.dropout), "learning_rate": float(r.learning_rate),
                           "batch_size": int(r.batch_size), "val_rmse": float(r.val_rmse), "tuning_plants": int(r.n_plants),
                           "parameters": models.count_parameters(r.model, LOOKBACK, CHANNELS[info], float(r.width))})
    chosen = pd.DataFrame(chosen).sort_values(["info", "horizon", "model"])
    chosen.to_csv(target / "ek_secilen_ayarlar.csv", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_colwidth", 70)
    for name, part in layers.groupby("model", sort=False):
        print(f"\n=== {name}: {part['parameters'].sum():,} parametre (genislik 1, 12 kanal) ===")
        print(part[["order", "type", "settings", "input_from", "output_shape", "parameters"]].to_string(index=False))
    print("\n=== parametre sayisi (genislik 1) ===")
    print(counts[counts["width"] == 1.0].pivot(index="model", columns="channels", values="parameters").loc[models.MODEL_NAMES].to_string())
    print("\n=== secilen ayarlar ===")
    print(chosen.to_string(index=False))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
