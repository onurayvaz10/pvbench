"""Regenerate the benchmark tables and figures with the current Python interpreter."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def table_steps(cikti: Path, sonuclar: Path):
    """Return each script and its positional arguments in hepsini_uret.ps1 order."""
    return [
        ("sonuc_tablolari.py", cikti, sonuclar),
        ("istatistik.py", cikti, sonuclar),
        ("duyarlilik_ortak_protokol.py", cikti, sonuclar),
        ("tablo_santraller.py", cikti, sonuclar),
        ("hizalama_denetimi.py", sonuclar),
        ("duzey_kaymasi.py", sonuclar),
        ("katmanli_analiz.py", sonuclar),
        ("yanlilik_konum.py", cikti, sonuclar),
        ("e2_tablolari.py", cikti, sonuclar),
        ("s2d2_tablosu.py", cikti, sonuclar),
        ("ek_tablolar.py", cikti, sonuclar),
        ("mimari_tablolari.py", cikti, sonuclar),
        ("e3_tablolari.py", cikti, sonuclar),
        ("e5_tablolari.py", cikti, sonuclar),
        ("e6_tablolari.py", cikti, sonuclar),
        ("e8_tablolari.py", cikti, sonuclar),
        ("e9_tablolari.py", cikti, sonuclar),
        ("ikinci_yil_tablosu.py", cikti, sonuclar),
        ("ham_sureklilik.py", sonuclar),
        ("saat_siniri_denetimi.py", sonuclar),
        ("epias_denetimi.py", sonuclar),
        ("dislama_denetimi.py", sonuclar),
    ]


# Inputs kept with the data rather than with the code: without them the step is skipped (its CSV is then reported
# as missing by karsilastir.py).
OPTIONAL_INPUTS = {"dislama_denetimi.py": "veri/dislama_denetimi.json"}


def run_step(kod, env, script, *paths):
    """Keep child errors visible and report elapsed wall time even on failure."""
    label = script
    if not (kod / script).is_file():
        print(f"SKIP  {label}  (script not in this archive)", flush=True)
        return 0
    optional = OPTIONAL_INPUTS.get(script)
    if optional and not (kod.parent / optional).is_file():
        print(f"SKIP  {label}  ({optional} not present)", flush=True)
        return 0
    print(f"START {label}", flush=True)
    started = time.perf_counter()
    try:
        # The PVOutput reader accepts an empty glob as a dataset. A missing
        # download directory must not yield a successful raw-record audit.
        raw_subdirs = {"saat_siniri_denetimi.py": "pvoutput", "epias_denetimi.py": "epias/uevm",
                       "dislama_denetimi.py": "pvoutput"}
        if script in raw_subdirs:
            raw = kod.parent / "veri" / "ham" / raw_subdirs[script]
            if not raw.is_dir():
                raise FileNotFoundError(f"raw downloads required for {script}: {raw}")
        command = [sys.executable, "-B", "-W", "ignore", str(kod / script), *map(str, paths)]
        subprocess.run(command, cwd=kod, env=env, stdout=subprocess.DEVNULL, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"FAIL  {label}  {time.perf_counter() - started:.2f} s  {exc}", file=sys.stderr, flush=True)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) and exc.returncode > 0 else 1
    print(f"OK    {label}  {time.perf_counter() - started:.2f} s", flush=True)
    return 0


def main():
    here = Path(__file__).resolve().parent
    default_code = here / "kod"
    if not default_code.is_dir() and (here.parent / "kod").is_dir():
        default_code = here.parent / "kod"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kod", type=Path, default=default_code, help="benchmark code directory")
    parser.add_argument("--cikti", type=Path, help="stored runs (default: cikti beside kod, or ~/pvbench_cikti)")
    parser.add_argument("--sonuclar", type=Path, help="output CSV directory (default: sonuclar beside kod)")
    parser.add_argument("--yalniz-tablolar", action="store_true", help="stop after the result CSV scripts")
    parser.add_argument("--sekiller", type=Path, help="figure directory (default: makale/sekiller beside kod)")
    args = parser.parse_args()

    kod = args.kod.expanduser().resolve()
    root = kod.parent
    paper = root / "makale"
    default_runs = root / "cikti"
    if not default_runs.is_dir():
        default_runs = Path.home() / "pvbench_cikti"
    runs = (args.cikti or default_runs).expanduser().resolve()
    results = (args.sonuclar or root / "sonuclar").expanduser().resolve()
    figures = (args.sekiller or paper / "sekiller").expanduser().resolve()
    for directory in (kod, runs):
        if not directory.is_dir():
            parser.error(f"missing input directory: {directory}")
    results.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", KERAS_BACKEND="torch", PYTHONIOENCODING="utf-8")
    steps = table_steps(runs, results)
    if not args.yalniz_tablolar:
        steps.extend([("makale_tablolari.py", results, paper / "tablolar"),
                      ("sekiller.py", runs, results, figures)])

    for script, *paths in steps:
        status = run_step(kod, env, script, *paths)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
