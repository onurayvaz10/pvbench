# PVBench

Code for *A Multi Site Benchmark of Deep Learning Models for Photovoltaic Power Forecasting Using Real World Data from
25 Plants* (O. Ayvaz and Ö. Tomak). The benchmark compares eleven neural network architectures and six reference
methods for hour-ahead and day-ahead forecasts of photovoltaic (PV) power at 25 plants in 12 countries, under
information sets that separate what is known when a forecast is issued from what is not.

This repository contains code only. It holds no data: no generation record, no table derived from the records and no
result table. The PVOutput records and the records supplied by companies were obtained for non-commercial academic use
only and are not redistributed. The results of every experiment are reported in the Supplementary Information of the
paper; the plant identifiers are given in its Table 1.

## Contents

| Path | Contents |
|---|---|
| `kod/pvbench/` | Library: raw records to hourly power (`pvoutput`, `epias`), hourly tables (`dataset`), information sets, splits and reference forecasts (`features`), the eleven networks (`models`), tuning and main runs (`experiment`), metrics and statistical tests (`metrics`, `stats`), experiments E5, E6, E8 and E10 (`importance`, `unseen`, `resolution`, `later`) |
| `kod/veri_cekme/` | Retrieval of PVOutput, EPİAŞ and Open-Meteo data, candidate searches of the plant selection, location check |
| `kod/*.py` | Result tables of every experiment, statistics, figures and the tables of the paper |
| `kod/egitim_*.ps1` | Original launch scripts (Windows PowerShell) with the exact arguments of every experiment |
| `kod/hepsini_uret.ps1` | Regenerates every result table, the tables and the figures of the paper from stored runs |
| `kod/testler/` | Tests of the parsers, features, models, metrics and statistics (synthetic data only) |
| `kod/requirements.txt` | Pinned package versions of the original environment |
| `calistir_hepsi.py`, `karsilastir.py` | Cross-platform runner of the result-table scripts and CSV comparator |
| `pvbench_colab.ipynb` | Google Colab notebook: tests, result tables, figures and model refits from a local archive of the data |
| `paket_olustur.py` | Builds a ZIP archive with a SHA-256 manifest (`--tur acik`: this code archive; `--tur tam --ham`: a private archive of the code with a local copy of the data, which the notebook needs and which must not be redistributed) |

File and script names are Turkish (`veri` = data, `ham` = raw, `islenmis` = processed, `sonuclar` = results,
`cikti` = run outputs, `egitim` = training, `ayar` = tuning).

## Input data

The code reads its inputs from a folder `veri/` beside `kod/`. The plant list is written by hand from Table 1 of the
paper (`pvoutput_cek.py kimlik` only fills in missing PVOutput system IDs); the other files are produced by the
retrieval scripts, `konum_dogrula.py` and the table builders below.

**Plant list** `veri/santraller.csv` (semicolon-separated, UTF-8). Columns: `no` (plant number), `ad` (name),
`pvoutput_adi` (system name on PVOutput), `kaynak` (`pvoutput` or `epias`), `sid` (PVOutput system ID or EPİAŞ
plant ID), `kapasite_kw` (rated capacity, kW), `ulke` (country, in Turkish: `Almanya`, `Belçika`, `Birleşik
Krallık`, `Hindistan`, `Hollanda`, `İtalya`, `Japonya`, `Lüksemburg`, `Macaristan`, `Tayland`, `Türkiye`,
`Yunanistan`), `tez_veri_baslangic`, `tez_veri_bitis` (first and last month of the record, `YYYY-MM`), `enlem`,
`boylam` (latitude and longitude), `atla` (non-empty: plant not used), `aciklama` (free-text note),
`cekim_baslangic`, `cekim_bitis` (optional retrieval period, `YYYY-MM-DD`).

**Location file** `veri/konum_dogrulama.csv`, written by `kod/veri_cekme/konum_dogrula.py`: plant number, time zone
(`saat_dilimi`, IANA name; it must be filled for every plant), verified latitude and longitude, and the location
checks. The time zone comes from the PVOutput system information (`pvoutput/sistemler.csv`) or, where that has none,
from the country (the Netherlands, Belgium, Türkiye and Luxembourg).

**Excluded systems** `veri/dislama_denetimi.json`: the systems, power band and dates checked by
`kod/dislama_denetimi.py` (Supplementary Note 1). It is kept with the data; without it that step is skipped.

**Raw downloads** under `veri/ham/` (`<no>` is the plant number with two digits, e.g. `01`):

| Source | Files | Content |
|---|---|---|
| PVOutput status service (`kod/veri_cekme/pvoutput_cek.py cek`) | `pvoutput/<sid>/YYYY-MM.csv` | Intraday status records in the system's local time: date, time, energy generated since local midnight (Wh), instantaneous power (W) and the further fields of the status service |
| PVOutput website (`kod/veri_cekme/pvoutput_site_cek.py`, the user's own login) | `pvoutput/<sid>/YYYY-MM.csv` | The same records in the same format, appended to the same month files; the script first checks a sample of days against the API records |
| PVOutput daily output service (`pvoutput_cek.py gunluk`) | `pvoutput/<sid>/gunluk.csv` | Daily totals; the interval from the last record of a day to midnight is valid only if the energy counter reached the daily total |
| PVOutput system service (`pvoutput_cek.py sistem`) | `pvoutput/sistemler.csv` | Location, time zone and reporting interval of each system (read by `konum_dogrula.py`) |
| EPİAŞ Transparency Platform (`kod/veri_cekme/epias_cek.py`, registered account) | `epias/uevm/<id>.csv`, `epias/gercek_zamanli/<id>.csv` | Hourly settlement injection of the plant (`zaman_tr`, start of the hour in Türkiye time; `gunes`, MWh) and the real-time generation used as a cross-check |
| Open-Meteo (`kod/veri_cekme/openmeteo_cek.py`, no key) | `openmeteo/<no>/arsiv_*.json`, `icon_*.json`, `tahmin_*.json` | Hourly reanalysis, and archived forecasts issued 24 h and 48 h before the valid hour (ICON Global; the provider's `best_match` series), for shortwave radiation, direct normal irradiance, diffuse radiation, 2 m temperature, 2 m relative humidity, 10 m wind speed and total cloud cover |
| Open-Meteo climate normals (`kod/veri_cekme/iklim_cek.py`) | `openmeteo/iklim_<n>.json` (plant number without leading zero) | Daily mean temperature and precipitation 1991–2020 for the climate classes of E7 |

The PVOutput API scripts read the API key and system ID from the environment variables `PVOUTPUT_API_KEY` and
`PVOUTPUT_SYSTEM_ID` or from a key file (default `~/pvoutput_kimlik.txt`, or `--kimlik <file>`). The website script
reads `PVOUTPUT_KULLANICI` and `PVOUTPUT_SIFRE` or asks at run time; the EPİAŞ script reads `EPIAS_KULLANICI` and
`EPIAS_SIFRE` or a key file given with `--kimlik`. No credential is stored in the repository.

**Processed tables**, built from `kod/` with `python -m pvbench.dataset` (hourly; also writes
`veri/islenmis/envanter.csv`) and `python -m pvbench.resolution --build` (experiment E8: 5 min tables in
`veri/islenmis/altsaatlik/<no>.parquet`; 10 and 20 min are aggregated from them when loaded):
`veri/islenmis/saatlik/<no>.parquet`, one table per plant indexed by the end of the hour in UTC, with `power_kw`
(interval-averaged AC power), `p` (power divided by rated capacity, the forecast target), `cs_ghi` (clear-sky
irradiance), `elev`, `elev_max` (solar elevation), `era_<variable>` (reanalysis), `nwp_<variable>`, `nwp2_<variable>`
(ICON forecast issued 24 h and 48 h earlier) and `nwpbm_<variable>` (`best_match` forecast) for the seven weather
variables above. Quality control sets the target to missing for hours not fully covered by valid records, night-time
generation and stuck values.

**Model inputs** (information sets, `kod/pvbench/features.py`), for an input window of 24 hours ending at the forecast
origin:

| Set | Channels |
|---|---|
| S0 | Past power, clear-sky irradiance at each hour and at the target hour, day of year of the target hour |
| S1 | S0 + reanalysis weather up to the origin |
| S2 | S0 + archived ICON forecast valid at the target hours (issued 24 h before its valid time) |
| S2B | as S2 with the provider's `best_match` forecast archive (sensitivity analysis only) |
| S3 | S0 + reanalysis weather at the target hours (not available in operation; upper bound) |
| S12 | S1 + S2 |
| S0m, S1m, S3m | S0, S1, S3 restricted to the samples S2 has |
| S1a | S1 + trailing 3 h mean of past power (ablation) |
| S2d2 | as S2 with the forecast issued 48 h earlier |

## Running the benchmark

Python 3.13 with the packages in `kod/requirements.txt` (Keras 3 with the PyTorch backend; `KERAS_BACKEND=torch`). Run
the modules from `kod/`. The order is: retrieval (`veri_cekme/` except `iklim_cek.py`), location check
(`veri_cekme/konum_dogrula.py`), hourly and sub-hourly tables (`python -m pvbench.dataset`,
`python -m pvbench.resolution --build`), climate normals (`python veri_cekme/iklim_cek.py`, which reads
`veri/islenmis/envanter.csv`), tuning and main runs (`python -m pvbench.experiment --stage tune ...` and
`--stage main ...`), result tables and figures. The arguments of every experiment are those in `kod/egitim_*.ps1`,
for example for E1 with S1 and S2B inputs (S2 with the ICON forecast is run the same way with `--info S2` into
`ayar_icon` and `ana_icon`):

```sh
cd kod
python -m pvbench.experiment --stage tune --out ../cikti/ayar --info S1 S2B --trials 12 --worker 0
python -m pvbench.experiment --stage main --out ../cikti/ana --tuning ../cikti/ayar --info S1 S2B --trials 12 --seeds 5 --worker 0
```

| Experiment | Launch script | Run outputs | Table script |
|---|---|---|---|
| E1 main benchmark (S1 and S2B in `ana`; S2 with the ICON forecast in `ana_icon`) | `egitim_baslat.ps1`, `egitim_s2_icon.ps1` | `ayar`, `ana`, `ayar_icon`, `ana_icon` | `sonuc_tablolari.py` |
| E2 information sets; forecast age | `egitim_e2.ps1`, `egitim_s2d2.ps1` | `ayar_e2`, `ana_e2`, `ayar_e2m`, `ana_e2m`, `ana_s2d2` | `e2_tablolari.py`, `s2d2_tablosu.py` |
| E3 equal size | `egitim_e3.ps1` | `ana_e3` | `e3_tablolari.py` |
| E4 statistics | – | – | `istatistik.py` |
| E5 input importance; moving-average input | `egitim_e5.ps1` | `e5_onem`, `ana_e5ma` | `e5_tablolari.py` |
| E6 unseen plants | `egitim_e6.ps1` | `e6_ref`, `e6` | `e6_tablolari.py` |
| E7 stratified analysis, level shifts | – | – | `katmanli_analiz.py`, `duzey_kaymasi.py` |
| E8 sub-hourly resolution | `egitim_e8.ps1` | `e8_ref`, `e8` | `e8_tablolari.py` |
| E9 horizons | `egitim_e9.ps1` | `ana_e9` | `e9_tablolari.py` |
| E10 second test year | `egitim_e10.ps1` | `e10_ref`, `e10` | `ikinci_yil_tablosu.py` |

`python calistir_hepsi.py --kod kod --cikti cikti --sonuclar <folder> --yalniz-tablolar` runs the result-table
scripts in their original order; `python karsilastir.py <stored> <regenerated>` compares two folders of result tables
(relative paths, shape, column order, values with an absolute tolerance). `python kod/sekiller.py <cikti> <sonuclar>
<figures>` draws the figures.

Ridge regression, LightGBM and the networks without convolutional layers are bit-reproducible on the same hardware with
a fixed seed; the five networks with convolutional layers are not, because GPU convolution routines are not
deterministic. Small differences also arise between devices and library versions.

## Google Colab notebook

`pvbench_colab.ipynb` runs in Google Colab on an archive built with `python paket_olustur.py --tur tam --ham` from a
local copy of the data (such an archive contains data and must not be redistributed; without `--ham` the three audits
of the raw records cannot run). It verifies the archive against its SHA-256 manifest, runs the tests, regenerates the
result tables and compares them with the stored ones (except three separately measured files: the timing measurement,
the determinism test and the check of one worker process), draws the figures, refits selected plants, information
sets, horizons and models with the stored settings, and writes a report.
The archive can be read from Google Drive (a path in My Drive or the ID of a file shared with the user), uploaded, or
dropped into the Files pane. With the code archive alone, the notebook runs the tests.

## Licence

The code is released under the MIT licence (`LICENSE`), © 2026 Onur Ayvaz and Özgür Tomak. The libraries it uses keep
their own licences, among them PyTorch (BSD-3-Clause, with bundled components under their own licences), Keras (Apache
2.0), LightGBM (MIT), scikit-learn, pvlib, NumPy, pandas and SciPy (BSD-3-Clause), Matplotlib (Matplotlib licence,
PSF-based), PyArrow and requests (Apache 2.0) and python-docx (MIT). Data retrieved with the scripts remain under the
terms of their providers (PVOutput, EPİAŞ; Open-Meteo data under CC BY 4.0).

## Citation

Ayvaz, O. & Tomak, Ö. PVBench: code and verification notebook for a multi-site benchmark of deep learning models for
photovoltaic power forecasting (25 plants), version 1.0.0. Zenodo https://doi.org/10.5281/zenodo.24400401 (2026).
Source repository: https://github.com/onurayvaz10/pvbench. The reference of the paper will be added when it is
published.
