"""pvbench: multi-site PV power forecasting benchmark (data, models, evaluation)."""

import os

# Must be set before numpy is imported by any module of the package: worker processes are bound to one
# core (two logical CPUs), and an unlimited BLAS thread pool on two CPUs makes linear algebra many times slower.
os.environ.setdefault("KERAS_BACKEND", "torch")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
