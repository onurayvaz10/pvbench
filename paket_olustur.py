"""Build a PVBench archive (ZIP) and hash its payload as it is written.

Variants
    acik   public code archive: benchmark code, verification tools and notebook. It holds no data: the PVOutput and
           company records were obtained for academic use only, so no generation record, no table derived from the
           records and no result table is redistributed.
    tam    private archive for the authors: the code plus the plant list, the processed hourly and sub-hourly tables,
           the stored runs with their test predictions and all result tables. `--ham` adds the raw downloads
           (PVOutput, EPİAŞ, Open-Meteo), which the raw-record audits need.
Neither variant contains the tools that build the manuscript documents or one-off retrieval scripts.
"""

import argparse
import csv
import hashlib
import io
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEPOSIT_FILES = (
    "LICENSE",
    "calistir_hepsi.py",
    "karsilastir.py",
    "paket_olustur.py",
    "pvbench_colab.ipynb",
    "README_depo.md",
    ".zenodo.json",
)
VARIANT_MARKER = "<!-- pvbench-package-variant -->"
FULL_NOTE = """Private archive for the authors. It includes the plant list, the processed hourly and
sub-hourly tables, the stored runs with their test predictions and all result tables, and, if
built with `--ham`, the raw PVOutput, EPİAŞ and Open-Meteo downloads. The PVOutput and company
records were obtained for non-commercial academic use only: do not redistribute this archive.
"""
OPEN_NOTE = """Public code archive. It holds no data: no generation record, no table derived
from the records and no result table. The section "Input data" describes the inputs the code
expects; the retrieval scripts in `kod/veri_cekme/` download them.
"""


# Tools that build the manuscript documents, and one-off retrieval scripts with private plant lists.
PRIVATE_CODE = {"kod/kaynakca.py", "kod/ek_dosya.py", "kod/word_donustur.py", "kod/belgeler_uret.ps1",
                "kod/veri_cekme/ek_cekim_it.cmd"}


def files_in(directory, run_outputs=False):
    """Walk a source tree in stable order, pruning excluded folders before entry."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing input directory: {directory}")

    def raise_walk_error(error):
        raise error

    for current, folders, filenames in os.walk(directory, onerror=raise_walk_error):
        parent = Path(current)
        keep = []
        for folder in sorted(folders):
            name = folder.casefold()
            if name in {"__pycache__", "claims", "dogrulama"} or name.startswith("yedek_"):
                continue
            # This also skips Windows directory junctions on older Python versions.
            if (parent / folder).resolve() == parent / folder:
                keep.append(folder)
        folders[:] = keep
        in_pred = any(p.casefold() == "pred" for p in parent.relative_to(directory).parts)
        for filename in sorted(filenames):
            name = filename.casefold()
            if "kimlik" in name or name.endswith((".docx", ".log")):
                continue
            if name == "log.txt" or (name.startswith("log_") and name.endswith(".txt")):
                continue
            if run_outputs and not in_pred and not name.endswith((".csv", ".json")):
                continue
            path = parent / filename
            if not path.is_symlink() and path.is_file():
                yield path


# Candidate lists of the PVOutput searches keep identifiers, checks and decisions.
# Names, postcodes, coordinates and equipment of private systems are left out even in the private archive.
CANDIDATE_DROP = {"sistem_adi", "posta_kodu", "panel", "evirici", "enlem", "boylam", "sistem_enlem", "sistem_boylam"}


def reduced_candidates(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, delimiter=";"))
    if not rows:
        raise ValueError(f"Empty candidate list: {path}")
    keep = [i for i, name in enumerate(rows[0]) if name not in CANDIDATE_DROP]
    if len(keep) == len(rows[0]):
        raise ValueError(f"No identifying columns found in {path}")
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter=";", lineterminator="\n")
    for row in rows:
        writer.writerow([row[i] if i < len(row) else "" for i in keep])
    return output.getvalue().encode("utf-8")


RAW_KEEP = re.compile(r"pvoutput/\d+/(\d{4}-\d{2}\.csv|gunluk\.csv)"
                      r"|epias/(uevm|gercek_zamanli)/\d+\.csv"
                      r"|openmeteo/(\d+/(arsiv|icon|tahmin)_[\w.-]+\.json|iklim_\d+\.json)")


def add_tree(members, prefix, directory, run_outputs=False):
    if directory.resolve() != directory:
        raise ValueError(f"Source directory is a symbolic link or junction: {directory}")
    for source in files_in(directory, run_outputs=run_outputs):
        members[f"{prefix}/{source.relative_to(directory).as_posix()}"] = source


def inventory(root, runs, variant, raw=False):
    """Return archive-relative POSIX names with their local source paths (or generated bytes)."""
    members = {name: HERE / name for name in DEPOSIT_FILES}
    if variant == "tam":
        for name in ("santraller.csv", "konum_dogrulama.csv", "dislama_denetimi.json"):
            members[f"veri/{name}"] = root / "veri" / name
        epias_candidates = root / "veri" / "ham" / "epias" / "aday_santraller_epias.csv"
        if epias_candidates.is_file():                      # written by the EPİAŞ search (epias_cek.py tara)
            members["veri/adaylar/aday_santraller_epias.csv"] = epias_candidates
    for path in members.values():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing regular input file: {path}")
    add_tree(members, "kod", root / "kod")
    for name in PRIVATE_CODE:
        members.pop(name, None)
    if variant == "tam":
        for path in sorted((root / "veri" / "ham" / "pvoutput").glob("aday_santraller*.csv")):
            members[f"veri/adaylar/{path.name}"] = reduced_candidates(path)
        add_tree(members, "sonuclar", root / "sonuclar")
        add_tree(members, "cikti", runs, run_outputs=True)
        add_tree(members, "veri/islenmis", root / "veri" / "islenmis")
        if raw:
            # Only the raw files the code reads: PVOutput month files and daily totals, EPİAŞ hourly series,
            # Open-Meteo responses and climate normals. Retrieval logs, system and owner metadata, candidate lists,
            # progress files and backups stay out.
            with (root / "veri" / "santraller.csv").open(encoding="utf-8-sig", newline="") as stream:
                epias_ids = {r["sid"] for r in csv.DictReader(stream, delimiter=";") if r["kaynak"] == "epias"}
            for source in ("pvoutput", "epias", "openmeteo"):
                for path in files_in(root / "veri" / "ham" / source):
                    relative = path.relative_to(root / "veri" / "ham").as_posix()
                    if source == "epias" and path.stem not in epias_ids:   # screened candidates stay out
                        continue
                    if RAW_KEEP.fullmatch(relative):
                        members[f"veri/ham/{relative}"] = path
    elif raw:
        raise ValueError("--ham applies to the private archive (--tur tam) only")
    return members


def package_readme(path, variant):
    # Replace an existing variant note when repackaging an extracted archive.
    base = path.read_text(encoding="utf-8-sig").split(VARIANT_MARKER, 1)[0].rstrip()
    note = FULL_NOTE if variant == "tam" else OPEN_NOTE
    return f"{base}\n\n{VARIANT_MARKER}\n## Contents of this ZIP (`{variant}`)\n\n{note}".encode("utf-8")


def build_archive(root, runs, destination, variant, raw=False):
    members = inventory(root, runs, variant, raw)
    source_dirs = [root / "kod", root / "sonuclar", root / "veri", runs]
    if any(destination.is_relative_to(directory) for directory in source_dirs):
        raise ValueError(f"Choose an archive destination outside the source trees: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    byte_count = 0
    with tempfile.TemporaryDirectory(prefix=".pvbench-", dir=destination.parent) as staging:
        staged = Path(staging) / "deposit.zip"
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            manifest = io.StringIO(newline="")
            rows = csv.writer(manifest, lineterminator="\n")
            rows.writerow(("path", "bytes", "sha256"))
            for name, source in sorted(members.items()):
                digest = hashlib.sha256()
                size = 0
                if name == "README_depo.md":
                    reader = io.BytesIO(package_readme(source, variant))
                elif isinstance(source, bytes):
                    reader = io.BytesIO(source)
                else:
                    reader = source.open("rb")
                with reader, archive.open(name, "w", force_zip64=True) as output:
                    while chunk := reader.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                rows.writerow((name, size, digest.hexdigest()))
                byte_count += size
            # A digest of the manifest itself cannot be stored inside that manifest.
            with archive.open("MANIFEST.csv", "w") as output:
                output.write(manifest.getvalue().encode("utf-8"))
        os.replace(staged, destination)
    print(f"{destination} ({len(members)} payload files, {byte_count:,} uncompressed bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kok", type=Path, default=HERE if (HERE / "kod").is_dir() else HERE.parent,
                        help="Project root containing kod, sonuclar and veri")
    parser.add_argument("--cikti", type=Path, help="Run output directory (default: <kok>/cikti or ~/pvbench_cikti)")
    parser.add_argument("--hedef", type=Path, default=Path.cwd(),
                        help="Destination ZIP path or directory (default: ./pvbench_<tur>.zip)")
    parser.add_argument("--tur", choices=("acik", "tam"), required=True,
                        help="acik = public code archive; tam = private archive with data and results")
    parser.add_argument("--ham", action="store_true",
                        help="tam only: also include the raw PVOutput, EPİAŞ and Open-Meteo downloads")
    args = parser.parse_args()
    root = args.kok.expanduser().resolve()
    runs = args.cikti or ((root / "cikti") if (root / "cikti").is_dir() else Path.home() / "pvbench_cikti")
    destination = args.hedef.expanduser().absolute()
    if destination.is_dir() or destination.suffix.casefold() != ".zip":
        destination /= f"pvbench_{args.tur}.zip"
    if destination.is_symlink():
        parser.error(f"Archive target is a symbolic link: {destination}")
    try:
        build_archive(root, runs.expanduser().resolve(), destination.resolve(), args.tur, args.ham)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Packaging failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
