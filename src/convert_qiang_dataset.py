from __future__ import annotations

# Convert Qiang coordinate files into RP folders.

import argparse
import re
import shutil
from pathlib import Path


# Parse Qiang coordinate filenames into x and y values.
def parse_coordinate_stem(stem: str) -> tuple[int, int]:
    m = re.search(r"coordinate\s*([0-9]+)$", stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse coordinate from filename stem: {stem}")
    digits = m.group(1)
    if len(digits) < 3:
        raise ValueError(f"Coordinate code too short in filename stem: {stem}")
    x = int(digits[:-2])
    y = int(digits[-2:])
    return x, y


# Program entry point.
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert qiang CSI dataset layout")
    parser.add_argument("--src", type=Path, required=True, help="Path to the downloaded repository or one scenario folder")
    parser.add_argument("--dst", type=Path, required=True, help="Output directory for DeepFi pipeline")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    parser.add_argument("--include-imaginary", action="store_true", help="Also copy files under imaginary_part folders")
    args = parser.parse_args()

    src = args.src
    dst = args.dst
    dst.mkdir(parents=True, exist_ok=True)

    files = []
    for p in src.rglob("*.mat"):
        if not args.include_imaginary and "imaginary_part" in {part.lower() for part in p.parts}:
            continue
        if re.search(r"coordinate\s*[0-9]+$", p.stem, re.IGNORECASE):
            files.append(p)

    if not files:
        raise SystemExit("No coordinate*.mat files found under source path")

    op = shutil.copy2 if args.copy else shutil.move

    count = 0
    for f in files:
        x, y = parse_coordinate_stem(f.stem)
        rp_dir = dst / f"x_{x}_y_{y}"
        rp_dir.mkdir(parents=True, exist_ok=True)
        target = rp_dir / f.name
        if target.exists():

            base = target.stem
            ext = target.suffix
            idx = 2
            while (rp_dir / f"{base}_{idx}{ext}").exists():
                idx += 1
            target = rp_dir / f"{base}_{idx}{ext}"
        op(f, target)
        count += 1

    print(f"Done. Converted {count} files into folder-per-reference-point format at: {dst}")


if __name__ == "__main__":
    main()
