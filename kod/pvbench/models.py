"""Neural forecasting models (Keras 3; backend chosen via KERAS_BACKEND).

Every builder maps an input window of shape (lookback, n_features) to one
scalar forecast. `width` scales all hidden sizes, which is how the
parameter-matched comparison is produced (see `match_width`).
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
from keras import layers  # noqa: E402

MODEL_NAMES = ["dense", "simplernn", "lstm", "gru", "bilstm", "cnn", "cnn_lstm", "cnn_gru",
               "cnn_lstm_gru", "transformer", "tcn"]


def _u(units, width, multiple=1):
    """Scaled layer size, at least 4 and a multiple of `multiple`."""
    return max(4, multiple, int(round(units * width / multiple)) * multiple)


def _head(x, sizes, dropouts, width):
    for size, drop in zip(sizes, dropouts):
        x = layers.Dense(_u(size, width), activation="relu")(x)
        if drop:
            x = layers.Dropout(drop)(x)
    return layers.Dense(1, name="forecast")(x)


def _recurrent(cell, inp, width, dropout):
    x = cell(_u(64, width), return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = cell(_u(32, width))(x)
    x = layers.Dropout(dropout)(x)
    return _head(x, [16], [0.0], width)


def _dense(inp, width, dropout):
    return _head(layers.Flatten()(inp), [128, 64, 32], [0.3, dropout, 0.0], width)


def _bilstm(inp, width, dropout):
    x = layers.Bidirectional(layers.LSTM(_u(48, width), return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(_u(24, width)))(x)
    return _head(layers.Dropout(dropout)(x), [16], [0.0], width)


def _cnn(inp, width, dropout):
    x = layers.Conv1D(_u(64, width), 2, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(_u(32, width), 2, padding="same", activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    return _head(x, [32, 16], [0.3, 0.0], width)


def _cnn_recurrent(cell, inp, width, dropout):
    x = layers.Conv1D(_u(32, width), 2, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = cell(_u(32, width), return_sequences=True)(x)
    x = cell(_u(16, width))(x)
    return _head(layers.Dropout(dropout)(x), [16], [0.0], width)


def _cnn_lstm_gru(inp, width, dropout):
    """Three parallel branches on the same input, fused by concatenation."""
    c = layers.Conv1D(_u(32, width), 2, padding="same", activation="relu")(inp)
    c = layers.GlobalMaxPooling1D()(layers.BatchNormalization()(c))
    l = layers.BatchNormalization()(layers.LSTM(_u(32, width))(inp))
    g = layers.BatchNormalization()(layers.GRU(_u(32, width))(inp))
    x = layers.Concatenate()([c, l, g])
    return _head(x, [64, 32], [0.3, dropout], width)


@keras.saving.register_keras_serializable(package="pvbench")
class AddPosition(layers.Layer):
    """Adds a learned positional embedding to a (batch, steps, channels) tensor."""

    def build(self, input_shape):
        self.position = self.add_weight(shape=(input_shape[1], input_shape[2]),
                                        initializer="random_normal", name="position")

    def call(self, x):
        return x + self.position


def _transformer(inp, width, dropout, heads=4, blocks=2):
    d = _u(32, width, multiple=heads)
    x = AddPosition()(layers.Dense(d)(inp))
    for _ in range(blocks):
        a = layers.MultiHeadAttention(heads, d // heads, dropout=dropout)(x, x)
        x = layers.LayerNormalization()(x + a)
        f = layers.Dense(d)(layers.Dense(2 * d, activation="relu")(x))
        x = layers.LayerNormalization()(x + layers.Dropout(dropout)(f))
    return _head(layers.GlobalAveragePooling1D()(x), [32], [dropout], width)


def _tcn(inp, width, dropout, kernel=3, dilations=(1, 2, 4, 8)):
    filters = _u(32, width)
    x = layers.Conv1D(filters, 1)(inp)
    for rate in dilations:
        r = layers.Conv1D(filters, kernel, padding="causal", dilation_rate=rate, activation="relu")(x)
        r = layers.Dropout(dropout)(r)
        r = layers.Conv1D(filters, kernel, padding="causal", dilation_rate=rate, activation="relu")(r)
        x = layers.Add()([x, r])
    return _head(x[:, -1, :], [32], [dropout], width)


_BUILDERS = {
    "dense": _dense,
    "simplernn": lambda i, w, d: _recurrent(layers.SimpleRNN, i, w, d),
    "lstm": lambda i, w, d: _recurrent(layers.LSTM, i, w, d),
    "gru": lambda i, w, d: _recurrent(layers.GRU, i, w, d),
    "bilstm": _bilstm,
    "cnn": _cnn,
    "cnn_lstm": lambda i, w, d: _cnn_recurrent(layers.LSTM, i, w, d),
    "cnn_gru": lambda i, w, d: _cnn_recurrent(layers.GRU, i, w, d),
    "cnn_lstm_gru": _cnn_lstm_gru,
    "transformer": _transformer,
    "tcn": _tcn,
}


def build_model(name, lookback, n_features, width=1.0, dropout=0.2, learning_rate=1e-3, compile_model=True):
    """Build (and by default compile with Adam + MSE) the model called `name`."""
    if name not in _BUILDERS:
        raise ValueError(f"unknown model '{name}'; choose from {MODEL_NAMES}")
    inp = keras.Input((lookback, n_features), name="window")
    model = keras.Model(inp, _BUILDERS[name](inp, width, dropout), name=name)
    if compile_model:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0), loss="mse")
    return model


def count_parameters(name, lookback, n_features, width=1.0):
    return build_model(name, lookback, n_features, width, compile_model=False).count_params()


def match_width(name, target, lookback, n_features, low=0.05, high=16.0, steps=24):
    """Width multiplier whose parameter count is closest to `target` (bisection).

    Returns (width, parameter_count). Layer sizes are integers, so the match is
    approximate; the achieved count is reported alongside the results.
    """
    best = None
    for _ in range(steps):
        mid = (low * high) ** 0.5
        n = count_parameters(name, lookback, n_features, mid)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (mid, n)
        if n < target:
            low = mid
        else:
            high = mid
    return best
