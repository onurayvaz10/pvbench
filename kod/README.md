# pvbench

Code for the paper *A Multi Site Benchmark of Deep Learning Models for Photovoltaic Power Forecasting Using Real World
Data from 25 Plants*. Every number, table and figure of the paper is produced by this code from the raw records.

File and script names are Turkish; the table below gives their meaning. Docstrings of the library modules
(`pvbench/`) are in English, those of the table scripts in Turkish.

## Requirements

Python 3.13 on Windows 11 with an NVIDIA GPU (the paper used an RTX 4080 Laptop GPU with 12 GB). All package versions
are pinned in `requirements.txt`:

```
python -m venv %USERPROFILE%\.venvs\pvbench
%USERPROFILE%\.venvs\pvbench\Scripts\python -m pip install -r requirements.txt
```

The library runs on any platform; the launch scripts (`egitim_*.ps1`, `hepsini_uret.ps1`) are PowerShell scripts and
expect the environment at the path above. Results of the training runs are written to `%USERPROFILE%\pvbench_cikti`
("cikti" = output), outside the repository.

## Layout

| Path | Content |
|---|---|
| `pvbench/pvoutput.py`, `pvbench/epias.py` | raw records → hourly interval-averaged power |
| `pvbench/dataset.py` | hourly table per plant: power, quality control, weather, solar geometry |
| `pvbench/features.py` | information sets (S0, S1, S2, S3, S12, …), splits, samples, post-processing |
| `pvbench/models.py` | the eleven network architectures (Keras 3) |
| `pvbench/experiment.py` | tuning and main runs, reference methods, training loop |
| `pvbench/metrics.py`, `pvbench/stats.py` | error metrics; Friedman/Nemenyi, Wilcoxon, Holm, Diebold–Mariano, bootstrap |
| `pvbench/importance.py`, `unseen.py`, `resolution.py`, `later.py` | experiments E5, E6, E8 and E10 |
| `pvbench/sitecheck.py` | checks of location and time alignment |
| `veri_cekme/` | retrieval of PVOutput, EPİAŞ and Open-Meteo data; candidate searches for the plant selection |
| `egitim_*.ps1` | launch scripts of the experiments ("egitim" = training) |
| `*_tablolari.py` and the other scripts in this folder | result tables (`../sonuclar`), figures and the tables of the paper |
| `testler/` | tests of the parsers, features, models, metrics and statistics |
| `../veri/` | plant list (`santraller.csv`), location file (`konum_dogrulama.csv`), raw data (`ham/`), processed tables (`islenmis/`); not distributed |
| `../sonuclar/` | result tables (CSV) from which the paper is written; not distributed |

## Data

The generation records come from two public platforms but are not redistributed: they were obtained for
non-commercial academic use only. PVOutput systems are retrieved through the PVOutput API, which requires the user's
own API key (`veri_cekme/pvoutput_cek.py`), or from the PVOutput website with the user's own login
(`veri_cekme/pvoutput_site_cek.py`, same file format); the EPİAŞ Transparency Platform requires a registered account;
Open-Meteo needs no key. The scripts read credentials from environment variables or from a key file (for PVOutput
by default `~/pvoutput_kimlik.txt`, otherwise `--kimlik <file>`); the website script asks at run time if the
variables are not set. No credential is stored in the repository. Each script documents its arguments in its
docstring. `python -m pvbench.dataset` builds the hourly tables from the raw files, and
`python -m pvbench.resolution --build` the 5 min tables of experiment E8 (10 and 20 min are aggregated from them).

## Reproducing tables and figures from the stored runs

```
powershell -File hepsini_uret.ps1
```

regenerates all result tables and the tables and figures of the paper from the run outputs in
`%USERPROFILE%\pvbench_cikti` (no training, no GPU). With `-Sonuclar <folder> -YalnizTablolar` the result tables are
written to another folder, for comparison with the stored ones.

## Re-running the experiments

Runs are resumable: a finished fit is skipped when a command is repeated. Workers share one GPU.

| Experiment | Launch script | Output folders | Table script |
|---|---|---|---|
| E1 main benchmark (S1; S2B with the provider's `best_match` forecast, sensitivity only) | `egitim_baslat.ps1 -Bilgi "S1 S2B"` | `ayar`, `ana` | `sonuc_tablolari.py` |
| E1 with the ICON forecast (S2) | `egitim_s2_icon.ps1` | `ayar_icon`, `ana_icon` | `sonuc_tablolari.py` |
| E2 information sets (full period and matched samples); forecast age | `egitim_e2.ps1`, `egitim_s2d2.ps1` | `ayar_e2`, `ana_e2`, `ayar_e2m`, `ana_e2m`, `ana_s2d2` | `e2_tablolari.py`, `s2d2_tablosu.py` |
| E3 equal size | `egitim_e3.ps1` | `ana_e3` | `e3_tablolari.py` |
| E4 statistics | – | – | `istatistik.py` |
| E5 input importance, moving-average input | `egitim_e5.ps1` | `e5_onem`, `ana_e5ma` | `e5_tablolari.py` |
| E6 unseen plants | `egitim_e6.ps1` | `e6_ref`, `e6` | `e6_tablolari.py` |
| E7 stratified analysis, level shifts | – | – | `katmanli_analiz.py`, `duzey_kaymasi.py` |
| E8 sub-hourly resolution | `egitim_e8.ps1` | `e8_ref`, `e8` | `e8_tablolari.py` |
| E9 horizons | `egitim_e9.ps1` | `ana_e9` | `e9_tablolari.py` |
| E10 second test year | `egitim_e10.ps1` | `e10_ref`, `e10` | `ikinci_yil_tablosu.py` |
| Training and prediction times | `python sure_olcumu.py <cikti> ../sonuclar` | – | `../sonuclar/sure_olcumu.csv` |

## Tests

Every file in `testler/` is run on its own, for example:

```
python testler/test_stats.py
python testler/test_metrics_stats.py
python testler/test_features.py
```

## Notes

- Ridge regression, LightGBM and the networks without convolutional layers (dense, simple RNN, LSTM, GRU, BiLSTM,
  Transformer) are bit-reproducible on the hardware used. The five networks with convolutional layers are not,
  because the GPU routines for convolutions are not deterministic (`belirlenim_denemesi.py`).
- Training times recorded by the benchmark runs are not comparable between methods, because several processes shared
  the GPU; `sure_olcumu.py` measures them with a single process.
- Scripts that bind the process to a core (`experiment.boost_process`) must import `pvbench.experiment` before NumPy,
  because that module sets the number of BLAS threads.

## Licence and citation

MIT licence (`../LICENSE`). Cite as: Ayvaz, O. & Tomak, Ö. PVBench: code and verification notebook for a multi-site
benchmark of deep learning models for photovoltaic power forecasting (25 plants), version 1.0.0. Zenodo
https://doi.org/10.5281/zenodo.24400401 (2026).
