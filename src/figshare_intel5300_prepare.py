from __future__ import annotations

# Prepare Figshare Intel 5300 CSI data.

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import scipy.io as sio
except Exception as e:
    sio = None

try:
    import h5py
except Exception:
    h5py = None


FILENAME_RE = re.compile(r"loc_(30deg|minus60deg)_(\d+)m\.mat$", re.IGNORECASE)


# One Figshare angle-distance condition.
@dataclass
class Condition:
    file: str
    angle_label: str
    angle_deg: float
    distance_m: float
    X: np.ndarray


# Create output folders when needed.
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Replace NaN and Inf values before modelling.
def sanitise_numeric_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# Fit feature-wise min-max scaling on training data.
def normalise_per_feature_minmax(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    X = sanitise_numeric_array(X)
    min_v = X.min(axis=0)
    max_v = X.max(axis=0)
    scale = np.maximum(max_v - min_v, eps)
    Xn = ((X - min_v) / scale).astype(np.float32)
    return sanitise_numeric_array(Xn), {"min": min_v.astype(np.float32), "scale": scale.astype(np.float32)}


# Apply saved training normalisation to new data.
def apply_minmax(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    X = sanitise_numeric_array(X)
    return sanitise_numeric_array(((X - stats["min"]) / stats["scale"]).astype(np.float32))


# Read angle and distance from Figshare filenames.
def parse_file_metadata(path: Path) -> Tuple[str, float, float]:
    m = FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse angle/distance from filename: {path.name}")
    angle_label = m.group(1).lower()
    distance = float(m.group(2))
    angle = 30.0 if angle_label == "30deg" else -60.0
    return angle_label, angle, distance


# Load raw Figshare CSI from MATLAB files.
def load_mat_array(path: Path) -> np.ndarray:
    if sio is None:
        raise RuntimeError("scipy is required to read MATLAB .mat files. Install with: pip install scipy")
    try:
        mat = sio.loadmat(path)
        if "csi_complex_data" in mat:
            return np.asarray(mat["csi_complex_data"])
        keys = [k for k in mat.keys() if not k.startswith("__")]
        three_d_keys = [k for k in keys if np.asarray(mat[k]).ndim == 3]
        if len(three_d_keys) == 1:
            print(f"[WARN] {path.name}: using key {three_d_keys[0]!r}; expected 'csi_complex_data'.")
            return np.asarray(mat[three_d_keys[0]])
        raise ValueError(f"Could not find 'csi_complex_data' in {path}; keys={keys}")
    except NotImplementedError:


        if h5py is None:
            raise RuntimeError("This MAT file appears to be v7.3/HDF5. Install h5py: pip install h5py")
        with h5py.File(path, "r") as f:
            if "csi_complex_data" not in f:
                raise ValueError(f"Could not find 'csi_complex_data' in HDF5 MAT file {path}; keys={list(f.keys())}")
            arr = np.array(f["csi_complex_data"])

            return arr


# Convert Figshare CSI to N x 90 amplitude features.
def amplitude_90_from_csi(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    amp = np.abs(arr).astype(np.float32)

    if amp.ndim != 3:
        raise ValueError(f"Expected 3D CSI matrix, got shape {amp.shape}")

    if amp.shape[0] == 3 and amp.shape[1] == 30:
        X = np.transpose(amp, (2, 0, 1)).reshape(amp.shape[2], 90)
    elif amp.shape[0] == 30 and amp.shape[1] == 3:
        X = np.transpose(amp, (2, 1, 0)).reshape(amp.shape[2], 90)
    elif amp.shape[1:] == (3, 30):
        X = amp.reshape(amp.shape[0], 90)
    elif amp.shape[1:] == (30, 3):
        X = np.transpose(amp, (0, 2, 1)).reshape(amp.shape[0], 90)
    else:
        raise ValueError(f"Unsupported CSI shape {amp.shape}; expected 3x30xN or equivalent.")

    return sanitise_numeric_array(X)


# Load all Figshare angle-distance conditions.
def load_conditions(data_dir: Path, max_packets_per_condition: Optional[int] = None) -> List[Condition]:
    mats = sorted(data_dir.rglob("*.mat"))
    matched = []
    for p in mats:
        if FILENAME_RE.search(p.name):
            matched.append(p)
    if not matched:
        raise FileNotFoundError(
            f"No Figshare files like loc_30deg_1m.mat were found under {data_dir}. "
            "Point --data-dir at the folder containing the 10 .mat files."
        )

    conditions: List[Condition] = []
    for p in matched:
        angle_label, angle_deg, distance_m = parse_file_metadata(p)
        raw = load_mat_array(p)
        X = amplitude_90_from_csi(raw)
        if max_packets_per_condition is not None and len(X) > max_packets_per_condition:
            X = X[:max_packets_per_condition]
        conditions.append(Condition(str(p), angle_label, angle_deg, distance_m, X))
        print(f"Loaded {p.name}: angle={angle_deg:g}, distance={distance_m:g}m, packets={len(X)}, shape={X.shape}")


    conditions.sort(key=lambda c: (0 if c.angle_deg == 30.0 else 1, c.distance_m))
    return conditions


# Convert polar distance and angle to 2D coordinates.
def polar_xy(distance: float, angle_deg: float) -> Tuple[float, float]:
    rad = math.radians(angle_deg)
    return distance * math.cos(rad), distance * math.sin(rad)


# Build train/test arrays for a selected Figshare mode.
def build_processed(conditions: List[Condition], mode: str, train_ratio: float) -> Dict[str, np.ndarray]:
    X_train, y_train, rp_train = [], [], []
    X_test, y_test, rp_test = [], [], []
    rp_coords_rows = []
    counts = {}
    records = []

    if mode == "mixed-coordinate":

        for rp_id, c in enumerate(conditions):
            x, y = polar_xy(c.distance_m, c.angle_deg)
            n = len(c.X)
            n_train = max(1, int(round(n * train_ratio)))
            n_train = min(n_train, n - 1) if n > 1 else n
            tr, te = c.X[:n_train], c.X[n_train:]
            X_train.append(tr); y_train.append(np.tile([[x, y]], (len(tr), 1))); rp_train.append(np.full(len(tr), rp_id))
            if len(te):
                X_test.append(te); y_test.append(np.tile([[x, y]], (len(te), 1))); rp_test.append(np.full(len(te), rp_id))
            rp_coords_rows.append([rp_id, x, y])
            counts[rp_id] = {"train": int(len(tr)), "test": int(len(te)), "angle_deg": c.angle_deg, "distance_m": c.distance_m}
            records.append({"rp_id": rp_id, "x": x, "y": y, "angle_deg": c.angle_deg, "distance_m": c.distance_m, "file": c.file})

    elif mode == "mixed-distance":

        distances = sorted({c.distance_m for c in conditions})
        dist_to_rp = {d: i for i, d in enumerate(distances)}
        for d, rp_id in dist_to_rp.items():
            rp_coords_rows.append([rp_id, d, 0.0])
            counts[rp_id] = {"train": 0, "test": 0, "distance_m": d}
            records.append({"rp_id": rp_id, "x": d, "y": 0.0, "distance_m": d, "file": "multiple angles"})
        for c in conditions:
            rp_id = dist_to_rp[c.distance_m]
            x, y = c.distance_m, 0.0
            n = len(c.X)
            n_train = max(1, int(round(n * train_ratio)))
            n_train = min(n_train, n - 1) if n > 1 else n
            tr, te = c.X[:n_train], c.X[n_train:]
            X_train.append(tr); y_train.append(np.tile([[x, y]], (len(tr), 1))); rp_train.append(np.full(len(tr), rp_id))
            if len(te):
                X_test.append(te); y_test.append(np.tile([[x, y]], (len(te), 1))); rp_test.append(np.full(len(te), rp_id))
            counts[rp_id]["train"] += int(len(tr)); counts[rp_id]["test"] += int(len(te))

    elif mode in {"train30-testminus60", "trainminus60-test30"}:
        train_angle = 30.0 if mode == "train30-testminus60" else -60.0
        test_angle = -60.0 if mode == "train30-testminus60" else 30.0
        distances = sorted({c.distance_m for c in conditions})
        dist_to_rp = {d: i for i, d in enumerate(distances)}
        for d, rp_id in dist_to_rp.items():
            rp_coords_rows.append([rp_id, d, 0.0])
            counts[rp_id] = {"train": 0, "test": 0, "distance_m": d, "train_angle": train_angle, "test_angle": test_angle}
            records.append({"rp_id": rp_id, "x": d, "y": 0.0, "distance_m": d, "train_angle": train_angle, "test_angle": test_angle})
        for c in conditions:
            rp_id = dist_to_rp[c.distance_m]
            x, y = c.distance_m, 0.0
            if c.angle_deg == train_angle:
                X_train.append(c.X); y_train.append(np.tile([[x, y]], (len(c.X), 1))); rp_train.append(np.full(len(c.X), rp_id))
                counts[rp_id]["train"] += int(len(c.X))
            elif c.angle_deg == test_angle:
                X_test.append(c.X); y_test.append(np.tile([[x, y]], (len(c.X), 1))); rp_test.append(np.full(len(c.X), rp_id))
                counts[rp_id]["test"] += int(len(c.X))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not X_train or not X_test:
        raise ValueError(f"Mode {mode} produced empty train or test data. Check files and mode.")

    X_train_raw = sanitise_numeric_array(np.concatenate(X_train, axis=0))
    X_test_raw = sanitise_numeric_array(np.concatenate(X_test, axis=0))
    X_train_norm, norm_stats = normalise_per_feature_minmax(X_train_raw)
    X_test_norm = apply_minmax(X_test_raw, norm_stats)

    return {
        "X_train": X_train_norm.astype(np.float32),
        "y_train": np.concatenate(y_train, axis=0).astype(np.float32),
        "rp_train": np.concatenate(rp_train, axis=0).astype(np.int64),
        "X_test": X_test_norm.astype(np.float32),
        "y_test": np.concatenate(y_test, axis=0).astype(np.float32),
        "rp_test": np.concatenate(rp_test, axis=0).astype(np.int64),
        "rp_coords": np.asarray(rp_coords_rows, dtype=np.float32),
        "counts": counts,
        "records": records,
        "norm_stats": norm_stats,
    }


# Save processed Figshare data and metadata.
def save_processed(bundle: Dict[str, np.ndarray], out_dir: Path, args: argparse.Namespace, conditions: List[Condition]) -> None:
    ensure_dir(out_dir)
    np.savez_compressed(
        out_dir / "processed_dataset.npz",
        X_train=bundle["X_train"],
        y_train=bundle["y_train"],
        rp_train=bundle["rp_train"],
        X_test=bundle["X_test"],
        y_test=bundle["y_test"],
        rp_test=bundle["rp_test"],
        rp_coords=bundle["rp_coords"],
    )

    meta = {
        "source": "Figshare Intel 5300 device-to-device localisation dataset",
        "mode": args.mode,
        "data_dir": str(Path(args.data_dir).resolve()),
        "train_ratio": args.train_ratio,
        "max_packets_per_condition": args.max_packets_per_condition,
        "counts": bundle["counts"],
        "records": bundle["records"],
        "conditions": [
            {"file": c.file, "angle_deg": c.angle_deg, "distance_m": c.distance_m, "packets": int(len(c.X))}
            for c in conditions
        ],
        "feature_shape": "N x 90 amplitude features from abs(3 x 30 x packets)",
        "normalisation": "feature-wise min-max fitted on training split only",
        "norm_min": bundle["norm_stats"]["min"].tolist(),
        "norm_scale": bundle["norm_stats"]["scale"].tolist(),
    }
    with open(out_dir / "prepare_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(out_dir / "rp_metadata.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in bundle["records"] for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bundle["records"])

    print(f"Saved processed dataset to {out_dir / 'processed_dataset.npz'}")
    print(f"Saved metadata to {out_dir / 'prepare_meta.json'}")
    print(f"Train packets: {len(bundle['X_train'])}")
    print(f"Test packets: {len(bundle['X_test'])}")
    print(f"RPs/classes: {len(bundle['rp_coords'])}")


# Optionally export Figshare data as RP folders.
def export_raw_layout(conditions: List[Condition], raw_root: Path, max_packets_per_condition: Optional[int] = None) -> None:
    if raw_root.exists():
        shutil.rmtree(raw_root)
    ensure_dir(raw_root)
    for c in conditions:
        x, y = polar_xy(c.distance_m, c.angle_deg)
        folder = raw_root / f"x_{x:.3f}_y_{y:.3f}_ang_{int(c.angle_deg)}_d_{int(c.distance_m)}m"
        ensure_dir(folder)
        X = c.X[:max_packets_per_condition] if max_packets_per_condition else c.X
        np.save(folder / "amplitude.npy", X.astype(np.float32))
    print(f"Exported raw-folder layout to {raw_root}")


# Program entry point.
def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Figshare Intel 5300 CSI dataset for existing localisation scripts")
    p.add_argument("--data-dir", required=True, help="Folder containing the 10 loc_*.mat files")
    p.add_argument("--out-dir", required=True, help="Output folder for processed_dataset.npz")
    p.add_argument(
        "--mode",
        default="mixed-coordinate",
        choices=["mixed-coordinate", "mixed-distance", "train30-testminus60", "trainminus60-test30"],
        help="Experiment split/label mode",
    )
    p.add_argument("--train-ratio", type=float, default=0.8, help="Within-condition train ratio for mixed modes")
    p.add_argument("--max-packets-per-condition", type=int, default=None, help="Optional cap for quick experiments")
    p.add_argument("--export-raw-root", default=None, help="Optional folder to export x_*_y_* .npy layout")
    args = p.parse_args()

    conditions = load_conditions(Path(args.data_dir), max_packets_per_condition=args.max_packets_per_condition)
    if args.export_raw_root:
        export_raw_layout(conditions, Path(args.export_raw_root), max_packets_per_condition=args.max_packets_per_condition)
    bundle = build_processed(conditions, args.mode, args.train_ratio)
    save_processed(bundle, Path(args.out_dir), args, conditions)


if __name__ == "__main__":
    main()
