"""Environment check: Keras 3 on the PyTorch backend, training on the GPU.

Builds the three-branch CNN-LSTM-GRU on synthetic data, trains briefly and
reports device, parameter count, epoch time and seed reproducibility.
"""

import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import keras  # noqa: E402
from keras import layers  # noqa: E402


def build(lookback=24, n_features=10):
    inp = keras.Input((lookback, n_features))
    c = layers.Conv1D(32, 2, padding="same", activation="relu")(inp)
    c = layers.GlobalMaxPooling1D()(layers.BatchNormalization()(c))
    l = layers.BatchNormalization()(layers.LSTM(32)(inp))
    g = layers.BatchNormalization()(layers.GRU(32)(inp))
    x = layers.Concatenate()([c, l, g])
    x = layers.Dropout(0.3)(layers.Dense(64, activation="relu")(x))
    x = layers.Dropout(0.2)(layers.Dense(32, activation="relu")(x))
    model = keras.Model(inp, layers.Dense(1)(x))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def run(seed, x, y, epochs=3):
    keras.utils.set_random_seed(seed)
    model = build(x.shape[1], x.shape[2])
    start = time.perf_counter()
    hist = model.fit(x, y, batch_size=256, epochs=epochs, verbose=0, shuffle=True)
    return model, hist.history["loss"], (time.perf_counter() - start) / epochs


def main():
    print("python", sys.version.split()[0], "| keras", keras.__version__, "| backend", keras.backend.backend())
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20000, 24, 10)).astype("float32")
    y = (x[:, -1, 0] * 0.5 + x[:, :, 1].mean(axis=1) + rng.normal(scale=0.1, size=20000)).astype("float32")
    model, loss_a, sec = run(42, x, y)
    device = next(iter(model.trainable_variables)).value.device
    print(f"weights on: {device} | parameters: {model.count_params():,} | {sec:.1f} s/epoch (20,000 samples)")
    print("loss per epoch:", [round(v, 4) for v in loss_a])
    _, loss_b, _ = run(42, x, y)
    _, loss_c, _ = run(7, x, y)
    print("same seed, max |loss diff|:", f"{max(abs(a - b) for a, b in zip(loss_a, loss_b)):.2e}")
    print("different seed, max |loss diff|:", f"{max(abs(a - c) for a, c in zip(loss_a, loss_c)):.2e}")
    ok = loss_a[-1] < loss_a[0] and (not torch.cuda.is_available() or "cuda" in str(device))
    print("RESULT:", "OK" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
