from __future__ import annotations

# DeepFi-style CSI localisation pipeline.

import argparse
import csv
import json
import pickle
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.io as sio
except Exception:
    sio = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    print("PyTorch is required. Install with: pip install torch", file=sys.stderr)
    raise

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Create output folders when needed.
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Store Python metadata for later reuse.
def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# Load saved Python metadata.
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# Extract reference-point coordinates from folder names.
def parse_coords_from_name(name: str) -> Tuple[float, float]:
    patterns = [
        r"x[_\-]?(-?\d+(?:\.\d+)?)_?y[_\-]?(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, name)
        if m:
            return float(m.group(1)), float(m.group(2))
    raise ValueError(
        f"Could not parse coordinates from folder/file name: {name}. Use a name like x_0.0_y_1.2"
    )


# Convert complex CSI to amplitude features.
def complex_to_amplitude(arr: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return arr.astype(np.float32)


# Fit feature-wise min-max scaling on training data.
def normalise_per_feature_minmax(X: np.ndarray, eps: float = 1e-8):
    min_v = X.min(axis=0)
    max_v = X.max(axis=0)
    scale = max_v - min_v
    scale[scale < eps] = 1.0
    Xn = (X - min_v) / scale
    stats = {"min": min_v, "scale": scale}
    return Xn.astype(np.float32), stats


# Apply saved training normalisation to new data.
def apply_minmax(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return ((X - stats["min"]) / stats["scale"]).astype(np.float32)


def _find_first_matching_key(d: Dict, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    lower_map = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    return None


# Reshape supported CSI arrays to N x 90 features.
def _amplitude_from_any_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = complex_to_amplitude(arr)

    if arr.ndim == 2:
        if arr.shape[1] == 90:
            return arr.astype(np.float32)
        if arr.shape[0] == 90:
            return arr.T.astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[1:] == (3, 30):
            return arr.reshape(arr.shape[0], 90).astype(np.float32)
        if arr.shape[1:] == (30, 3):
            return np.transpose(arr, (0, 2, 1)).reshape(arr.shape[0], 90).astype(np.float32)
        if arr.shape[:2] == (3, 30):
            return np.transpose(arr, (2, 0, 1)).reshape(arr.shape[2], 90).astype(np.float32)
        if arr.shape[:2] == (30, 3):
            return np.transpose(arr, (2, 1, 0)).reshape(arr.shape[2], 90).astype(np.float32)

    raise ValueError(f"Unsupported CSI/amplitude shape: {arr.shape}")


# Read CSI arrays from MATLAB files.
def load_from_mat(path: Path) -> np.ndarray:
    if sio is None:
        raise RuntimeError("scipy is required to read .mat files")
    mat = sio.loadmat(path)
    key = _find_first_matching_key(mat, ["csi", "amp", "amplitude", "data", "X"])
    if key is None:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if len(keys) == 1:
            key = keys[0]
        else:
            raise ValueError(f"Could not find CSI array in MAT file {path}; keys={keys}")
    return _amplitude_from_any_array(mat[key])


# Read CSI arrays from NumPy files.
def load_from_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    return _amplitude_from_any_array(arr)


# Read CSI arrays from compressed NumPy files.
def load_from_npz(path: Path) -> np.ndarray:
    npz = np.load(path, allow_pickle=True)
    key = None
    for candidate in ["csi", "amp", "amplitude", "data", "X"]:
        if candidate in npz:
            key = candidate
            break
    if key is None:
        if len(npz.files) == 1:
            key = npz.files[0]
        else:
            raise ValueError(f"Could not find CSI array in NPZ file {path}; keys={npz.files}")
    return _amplitude_from_any_array(npz[key])


# Read CSI amplitude features from CSV files.
def load_from_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is not None:
        numeric_cols = []
        for name in data.dtype.names:
            lname = name.lower()
            if lname in {"x", "y", "label", "rp", "rp_id", "sample", "sample_id"}:
                continue
            col = np.asarray(data[name], dtype=np.float32)
            numeric_cols.append(col)
        X = np.stack(numeric_cols, axis=1)
    else:
        raw = np.genfromtxt(path, delimiter=",", dtype=np.float32)
        X = raw
    return _amplitude_from_any_array(X)


# Dispatch CSI loading based on file extension.
def load_csi_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return load_from_mat(path)
    if suffix == ".npy":
        return load_from_npy(path)
    if suffix == ".npz":
        return load_from_npz(path)
    if suffix == ".csv":
        return load_from_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


# Container for one reference point.
@dataclass
class RPData:
    rp_id: int
    x: float
    y: float
    packets: np.ndarray


# Find reference-point folders in a raw dataset.
def discover_rp_dirs(raw_root: Path) -> List[Path]:
    dirs = [p for p in raw_root.iterdir() if p.is_dir()]
    if dirs:
        return sorted(dirs)
    raise ValueError(f"No reference-point directories found in {raw_root}. Expected subfolders like x_0.0_y_0.0")


# Load all CSI packets grouped by reference point.
def load_raw_dataset(raw_root: Path, max_packets_per_rp: Optional[int] = None) -> List[RPData]:
    rp_dirs = discover_rp_dirs(raw_root)
    rps: List[RPData] = []

    for rp_id, rp_dir in enumerate(rp_dirs):
        x, y = parse_coords_from_name(rp_dir.name)
        chunks = []
        for f in sorted(rp_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".mat", ".npy", ".npz", ".csv"}:
                continue
            try:
                arr = load_csi_file(f)
            except Exception as e:
                print(f"[WARN] Skipping {f}: {e}")
                continue
            if arr.size == 0:
                continue
            chunks.append(arr)
        if not chunks:
            raise ValueError(f"No readable CSI files found in {rp_dir}")
        packets = np.concatenate(chunks, axis=0)
        if max_packets_per_rp is not None and packets.shape[0] > max_packets_per_rp:
            packets = packets[:max_packets_per_rp]
        rps.append(RPData(rp_id=rp_id, x=x, y=y, packets=packets))
    return rps


# Create packet-level train/test splits per reference point.
def split_train_test_by_packet(rps: List[RPData], train_ratio: float = 0.8) -> Dict[str, np.ndarray]:
    X_train, y_train, rp_train = [], [], []
    X_test, y_test, rp_test = [], [], []
    counts = {}

    for rp in rps:
        packets = rp.packets
        n = packets.shape[0]
        n_train = max(1, int(round(n * train_ratio)))
        n_train = min(n_train, n - 1) if n > 1 else n

        train = packets[:n_train]
        test = packets[n_train:] if n > 1 else packets[:0]

        X_train.append(train)
        y_train.append(np.tile(np.array([[rp.x, rp.y]], dtype=np.float32), (train.shape[0], 1)))
        rp_train.append(np.full((train.shape[0],), rp.rp_id, dtype=np.int64))

        if len(test) > 0:
            X_test.append(test)
            y_test.append(np.tile(np.array([[rp.x, rp.y]], dtype=np.float32), (test.shape[0], 1)))
            rp_test.append(np.full((test.shape[0],), rp.rp_id, dtype=np.int64))

        counts[rp.rp_id] = {"train": int(train.shape[0]), "test": int(test.shape[0])}

    result = {
        "X_train": np.concatenate(X_train, axis=0).astype(np.float32),
        "y_train": np.concatenate(y_train, axis=0).astype(np.float32),
        "rp_train": np.concatenate(rp_train, axis=0).astype(np.int64),
        "rp_coords": np.array([[rp.rp_id, rp.x, rp.y] for rp in rps], dtype=np.float32),
        "counts": counts,
    }
    if X_test:
        result["X_test"] = np.concatenate(X_test, axis=0).astype(np.float32)
        result["y_test"] = np.concatenate(y_test, axis=0).astype(np.float32)
        result["rp_test"] = np.concatenate(rp_test, axis=0).astype(np.int64)
    else:
        result["X_test"] = np.zeros((0, 90), dtype=np.float32)
        result["y_test"] = np.zeros((0, 2), dtype=np.float32)
        result["rp_test"] = np.zeros((0,), dtype=np.int64)
    return result


# Autoencoder used for DeepFi-style matching.
class DeepFiAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 90, hidden_dims: Sequence[int] = (500, 300, 150, 50)):
        super().__init__()
        h1, h2, h3, h4 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.Sigmoid(),
            nn.Linear(h1, h2),
            nn.Sigmoid(),
            nn.Linear(h2, h3),
            nn.Sigmoid(),
            nn.Linear(h3, h4),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h4, h3),
            nn.Sigmoid(),
            nn.Linear(h3, h2),
            nn.Sigmoid(),
            nn.Linear(h2, h1),
            nn.Sigmoid(),
            nn.Linear(h1, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruct(self, x):
        return self.forward(x)


# Train one DeepFi-style autoencoder for a reference point.
def train_one_rp_model(X: np.ndarray, hidden_dims: Sequence[int], epochs: int, batch_size: int, lr: float, noise_std: float, device: torch.device):
    model = DeepFiAutoencoder(input_dim=X.shape[1], hidden_dims=hidden_dims).to(device)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {"loss": []}

    for _ in range(epochs):
        model.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            noisy = xb
            if noise_std > 0:
                noisy = torch.clamp(xb + noise_std * torch.randn_like(xb), 0.0, 1.0)
            recon = model(noisy)
            loss = criterion(recon, xb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        history["loss"].append(float(np.mean(losses)))
    return model, history


# Split packet streams into fixed-size windows.
def chunk_array(X: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    if chunk_size <= 1:
        return [X[i:i+1] for i in range(len(X))]
    return [X[i:i+chunk_size] for i in range(0, len(X), chunk_size) if len(X[i:i+chunk_size]) > 0]


# Convert reconstruction error to matching likelihood.
def likelihood_from_reconstruction_errors(errors: np.ndarray, lambda_sigma: float) -> float:
    sigma = float(np.std(errors))
    sigma = max(sigma, 1e-6)
    vals = np.exp(-(errors / (lambda_sigma * sigma)))
    return float(np.mean(vals))


# Estimate location from one packet window.
def localise_packet_window(packet_window: np.ndarray, rp_models: Dict[int, DeepFiAutoencoder], rp_coords: Dict[int, np.ndarray], device: torch.device, lambda_sigma: float = 1.0, top_k: Optional[int] = None) -> Dict:
    x_t = torch.tensor(packet_window, dtype=torch.float32, device=device)
    scores = []
    with torch.no_grad():
        for rp_id, model in rp_models.items():
            model.eval()
            recon = model.reconstruct(x_t).cpu().numpy()
            errors = np.linalg.norm(packet_window - recon, axis=1)
            p = likelihood_from_reconstruction_errors(errors, lambda_sigma=lambda_sigma)
            scores.append((rp_id, p))

    probs = np.array([p for _, p in scores], dtype=np.float64)
    if probs.sum() <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / probs.sum()

    rp_ids = [rp_id for rp_id, _ in scores]
    if top_k is not None and top_k < len(rp_ids):
        order = np.argsort(probs)[::-1][:top_k]
        rp_ids = [rp_ids[i] for i in order]
        probs = probs[order]
        probs = probs / probs.sum()

    coords = np.stack([rp_coords[rp_id] for rp_id in rp_ids], axis=0)
    pred_xy = (coords * probs[:, None]).sum(axis=0)
    best_idx = int(np.argmax(probs))
    return {
        "pred_xy": pred_xy.astype(np.float32),
        "top_rp_id": int(rp_ids[best_idx]),
        "top_prob": float(probs[best_idx]),
        "posterior": {int(rp_ids[i]): float(probs[i]) for i in range(len(rp_ids))},
    }


# CLI command: prepare a processed dataset.
def cmd_prepare(args):
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    rps = load_raw_dataset(raw_root, max_packets_per_rp=args.max_packets_per_rp)
    split = split_train_test_by_packet(rps, train_ratio=args.train_ratio)

    X_train_raw = split["X_train"]
    X_train, norm_stats = normalise_per_feature_minmax(X_train_raw)
    X_test_raw = split["X_test"]
    X_test = apply_minmax(X_test_raw, norm_stats) if len(X_test_raw) else X_test_raw.astype(np.float32)

    meta = {
        "rp_coords": split["rp_coords"],
        "counts": split["counts"],
        "norm_stats": norm_stats,
        "train_ratio": args.train_ratio,
        "raw_root": str(raw_root.resolve()),
    }

    np.savez_compressed(
        out_dir / "processed_dataset.npz",
        X_train=X_train,
        y_train=split["y_train"],
        rp_train=split["rp_train"],
        X_test=X_test,
        y_test=split["y_test"],
        rp_test=split["rp_test"],
        rp_coords=split["rp_coords"],
    )
    save_pickle(meta, out_dir / "prepare_meta.pkl")
    print(f"Saved processed dataset to {out_dir / 'processed_dataset.npz'}")
    print(f"Reference points: {len(rps)}")
    print(f"Train packets: {len(X_train)}")
    print(f"Test packets: {len(X_test)}")


# CLI command: train a model.
def cmd_train(args):
    proc_path = Path(args.processed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "models")

    data = np.load(proc_path, allow_pickle=True)
    X_train = data["X_train"].astype(np.float32)
    rp_train = data["rp_train"].astype(np.int64)
    rp_coords_arr = data["rp_coords"].astype(np.float32)
    rp_coords = {int(row[0]): np.array([row[1], row[2]], dtype=np.float32) for row in rp_coords_arr}
    hidden_dims = tuple(args.hidden_dims)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model_index = []
    for rp_id in sorted(set(rp_train.tolist())):
        X_rp = X_train[rp_train == rp_id]
        if len(X_rp) == 0:
            continue
        print(f"Training RP {rp_id} with {len(X_rp)} packets...")
        model, hist = train_one_rp_model(X_rp, hidden_dims, args.epochs, args.batch_size, args.lr, args.noise_std, device)
        save_path = out_dir / "models" / f"rp_{rp_id}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "hidden_dims": hidden_dims,
            "rp_id": int(rp_id),
            "coord": rp_coords[rp_id],
        }, save_path)
        model_index.append({
            "rp_id": int(rp_id),
            "path": str(save_path.resolve()),
            "coord": rp_coords[rp_id].tolist(),
            "final_loss": hist["loss"][-1],
        })

    train_meta = {
        "processed_path": str(proc_path.resolve()),
        "hidden_dims": hidden_dims,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "noise_std": args.noise_std,
        "model_index": model_index,
    }
    save_pickle(train_meta, out_dir / "train_meta.pkl")
    print(f"Saved {len(model_index)} RP models into {out_dir / 'models'}")


# Load all saved reference-point models.
def load_trained_models(model_dir: Path, device: torch.device):
    rp_models = {}
    rp_coords = {}
    for model_file in sorted((model_dir / "models").glob("rp_*.pt")):
        obj = torch.load(model_file, map_location=device, weights_only=False)
        model = DeepFiAutoencoder(input_dim=90, hidden_dims=obj["hidden_dims"]).to(device)
        model.load_state_dict(obj["state_dict"])
        model.eval()
        rp_id = int(obj["rp_id"])
        rp_models[rp_id] = model
        rp_coords[rp_id] = np.array(obj["coord"], dtype=np.float32)
    if not rp_models:
        raise ValueError(f"No models found in {model_dir / 'models'}")
    return rp_models, rp_coords


# CLI command: evaluate a trained model.
def cmd_evaluate(args):
    proc = np.load(Path(args.processed), allow_pickle=True)
    X_test = proc["X_test"].astype(np.float32)
    y_test = proc["y_test"].astype(np.float32)
    rp_test = proc["rp_test"].astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rp_models, rp_coords = load_trained_models(Path(args.model_dir), device)

    pred_rows = []
    for rp_id in sorted(set(rp_test.tolist())):
        X_rp = X_test[rp_test == rp_id]
        y_rp = y_test[rp_test == rp_id]
        windows = chunk_array(X_rp, args.window_size)
        y_windows = chunk_array(y_rp, args.window_size)

        for window_idx, (wX, wy) in enumerate(zip(windows, y_windows)):
            result = localise_packet_window(wX, rp_models, rp_coords, device, lambda_sigma=args.lambda_sigma, top_k=args.top_k)
            true_xy = wy[0]
            pred_xy = result["pred_xy"]
            err = float(np.linalg.norm(pred_xy - true_xy))
            pred_rows.append({
                "true_x": float(true_xy[0]),
                "true_y": float(true_xy[1]),
                "pred_x": float(pred_xy[0]),
                "pred_y": float(pred_xy[1]),
                "error_m": err,
                "true_rp": int(rp_id),
                "top_rp": int(result["top_rp_id"]),
                "top_prob": float(result["top_prob"]),
                "window_idx": int(window_idx),
                "num_packets": int(len(wX)),
            })

    if not pred_rows:
        raise ValueError("No test windows were produced. Check your processed dataset split.")

    errors = np.array([r["error_m"] for r in pred_rows], dtype=np.float32)
    metrics = {
        "num_estimates": int(len(pred_rows)),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "p90_error_m": float(np.percentile(errors, 90)),
        "within_1m": float(np.mean(errors <= 1.0)),
        "within_2m": float(np.mean(errors <= 2.0)),
    }

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {out_dir / 'predictions.csv'}")
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


# CLI command: predict locations for new CSI files.
def cmd_predict(args):
    input_path = Path(args.input)
    out_csv = Path(args.out)

    if input_path.suffix.lower() == ".npz":
        arr = np.load(input_path, allow_pickle=True)
        key = "X" if "X" in arr else arr.files[0]
        X = _amplitude_from_any_array(arr[key])
    else:
        X = load_csi_file(input_path)

    prepare_meta = load_pickle(Path(args.prepare_meta))
    X = apply_minmax(X, prepare_meta["norm_stats"])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rp_models, rp_coords = load_trained_models(Path(args.model_dir), device)

    windows = chunk_array(X, args.window_size)
    rows = []
    for i, w in enumerate(windows):
        result = localise_packet_window(w, rp_models, rp_coords, device, lambda_sigma=args.lambda_sigma, top_k=args.top_k)
        rows.append({
            "window_idx": i,
            "num_packets": len(w),
            "pred_x": float(result["pred_xy"][0]),
            "pred_y": float(result["pred_xy"][1]),
            "top_rp": int(result["top_rp_id"]),
            "top_prob": float(result["top_prob"]),
        })

    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved predictions to {out_csv}")


# Define command-line interface.
def build_argparser():
    p = argparse.ArgumentParser(description="DeepFi-style CSI localisation pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="Load raw Intel 5300 CSI dataset and build a processed dataset")
    p_prepare.add_argument("--raw-root", required=True, help="Folder containing RP subfolders")
    p_prepare.add_argument("--out-dir", required=True, help="Where to save processed_dataset.npz")
    p_prepare.add_argument("--train-ratio", type=float, default=0.8, help="Packet-level train split ratio per RP")
    p_prepare.add_argument("--max-packets-per-rp", type=int, default=None, help="Optional cap for faster experiments")
    p_prepare.set_defaults(func=cmd_prepare)

    p_train = sub.add_parser("train", help="Train one DeepFi-style autoencoder per RP")
    p_train.add_argument("--processed", required=True, help="Path to processed_dataset.npz")
    p_train.add_argument("--out-dir", required=True, help="Output folder for trained models")
    p_train.add_argument("--hidden-dims", nargs=4, type=int, default=[500, 300, 150, 50])
    p_train.add_argument("--epochs", type=int, default=40)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--noise-std", type=float, default=0.01)
    p_train.add_argument("--cpu", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate trained models on processed test data")
    p_eval.add_argument("--processed", required=True, help="Path to processed_dataset.npz")
    p_eval.add_argument("--model-dir", required=True, help="Folder produced by train")
    p_eval.add_argument("--out-dir", required=True, help="Folder for metrics and predictions")
    p_eval.add_argument("--window-size", type=int, default=10, help="Packets per location estimate")
    p_eval.add_argument("--lambda-sigma", type=float, default=1.0)
    p_eval.add_argument("--top-k", type=int, default=None, help="Restrict weighted average to top-k RPs")
    p_eval.add_argument("--cpu", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Predict locations for new CSI packet windows")
    p_pred.add_argument("--input", required=True, help="New CSI file (.csv/.npy/.npz/.mat)")
    p_pred.add_argument("--prepare-meta", required=True, help="prepare_meta.pkl from prepare stage")
    p_pred.add_argument("--model-dir", required=True, help="Folder produced by train")
    p_pred.add_argument("--out", required=True, help="Output CSV path")
    p_pred.add_argument("--window-size", type=int, default=10)
    p_pred.add_argument("--lambda-sigma", type=float, default=1.0)
    p_pred.add_argument("--top-k", type=int, default=None)
    p_pred.add_argument("--cpu", action="store_true")
    p_pred.set_defaults(func=cmd_predict)
    return p


# Program entry point.
def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
