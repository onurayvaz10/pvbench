"""Compare result CSVs by relative path, header, row order and column values."""

import argparse
import csv
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path


def read_table(path):
    """Keep original text until both copies of a column can be inspected."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        rows = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError(
                    f"line {reader.line_num} has {len(row)} fields, expected {len(header)}"
                )
            rows.append(row)
    return header, rows


def csv_paths(folder, skipped):
    if not folder.is_dir():
        raise ValueError(f"result folder does not exist or is not a directory: {folder}")
    return {
        path.relative_to(folder).as_posix(): path
        for path in folder.rglob("*")
        if path.suffix.lower() == ".csv" and path.name not in skipped and path.is_file()
    }


def numbers(cells):
    """Return numeric values with None for missing entries, or None for a text column."""
    values = []
    for cell in cells:
        if cell.strip() == "":
            values.append(None)
            continue
        try:
            value = Decimal(cell)
        except InvalidOperation:
            return None
        values.append(None if value.is_nan() else value)
    return values


def column_differences(left, right, tolerance):
    a, b = numbers(left), numbers(right)
    if a is None or b is None:
        count = sum(x != y for x, y in zip(left, right))
        return [f"{count} differing string value(s)"] if count else []

    missing = 0
    over = 0
    largest = Decimal(0)
    for x, y in zip(a, b):
        if x is None or y is None:
            missing += (x is None) != (y is None)
        elif x != y:
            if x.is_finite() and y.is_finite():
                # Keep the last place of both inputs, including around the tolerance boundary.
                with localcontext() as ctx:
                    high = max(x.adjusted(), y.adjusted())
                    low = min(x.as_tuple().exponent, y.as_tuple().exponent)
                    ctx.prec = max(ctx.prec, high - low + 2)
                    diff = (x - y).copy_abs()
            else:
                diff = Decimal("Infinity")
            largest = max(largest, diff)
            over += diff > tolerance
    issues = []
    if missing:
        issues.append(f"{missing} NaN position mismatch(es)")
    if over:
        issues.append(f"max abs diff {largest} over paired numbers, {over} above tolerance")
    return issues


def column_keys(header):
    """Identify repeated names by occurrence so no physical column is lost."""
    counts = Counter()
    keys = []
    for name in header:
        counts[name] += 1
        keys.append((name, counts[name]))
    return keys


def differences(left_path, right_path, tolerance):
    left_header, left = read_table(left_path)
    right_header, right = read_table(right_path)
    issues = []
    if (len(left), len(left_header)) != (len(right), len(right_header)):
        issues.append(
            f"shape (rows, columns): ({len(left)}, {len(left_header)}) vs "
            f"({len(right)}, {len(right_header)})"
        )
    if left_header != right_header:
        issues.append(f"column names/order: {left_header!r} vs {right_header!r}")
    right_index = {key: i for i, key in enumerate(column_keys(right_header))}
    for i, key in enumerate(column_keys(left_header)):
        name, occurrence = key
        if name == "finished" or key not in right_index:
            continue
        j = right_index[key]
        problems = column_differences([r[i] for r in left], [r[j] for r in right], tolerance)
        if problems:
            label = repr(name) if occurrence == 1 else f"{name!r} (occurrence {occurrence})"
            issues.append(f"column {label}: {', '.join(problems)}")
    return issues


def parse_tolerance(text):
    try:
        value = Decimal(text)
    except InvalidOperation:
        value = None
    if value is None or not value.is_finite() or value < 0:
        raise argparse.ArgumentTypeError("tolerance must be finite and nonnegative")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stored", type=Path)
    parser.add_argument("regenerated", type=Path)
    parser.add_argument("--tolerans", type=parse_tolerance, default=Decimal("1e-9"),
                        help="maximum absolute numeric difference (default: 1e-9)")
    parser.add_argument("--yoksay", nargs="*", metavar="FILENAME", action="append",
                        help="replace the default skip list (sure_olcumu.csv); pass no names to skip none")
    args = parser.parse_args()
    skipped = ({"sure_olcumu.csv"} if args.yoksay is None
               else {name for group in args.yoksay for name in group})
    try:
        stored = csv_paths(args.stored, skipped)
        regenerated = csv_paths(args.regenerated, skipped)
    except (OSError, ValueError) as error:
        print(f"Summary: comparison failed ({error})", file=sys.stderr)
        return 1
    names = sorted(stored.keys() | regenerated.keys())
    different = 0
    for name in names:
        if name not in regenerated:
            issues = ["missing from regenerated folder"]
        elif name not in stored:
            issues = ["only in regenerated folder"]
        else:
            try:
                issues = differences(stored[name], regenerated[name], args.tolerans)
            except (OSError, ValueError, csv.Error, ArithmeticError) as error:
                issues = [f"could not compare: {error}"]
        if issues:
            different += 1
            print(name)
            for issue in issues:
                print(f"  {issue}")
    print(f"Summary: {len(names)} CSV path(s), {len(names) - different} matched, "
          f"{different} differed; atol={args.tolerans}; skip={', '.join(sorted(skipped)) or '(none)'}")
    return int(different != 0)


if __name__ == "__main__":
    sys.exit(main())
