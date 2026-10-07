from __future__ import annotations

# Windowed TCN model for CSI localisation.

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import f1_score
except Exception:
    f1_score = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Create output folders when needed.
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Replace NaN and Inf values before modelling.
def sanitise_numeric_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# Split packet streams into fixed-size windows.
def chunk_array(X: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    if chunk_size <= 1:
        return [X[i:i+1] for i in range(len(X))]
    return [X[i:i+chunk_size] for i in range(0, len(X), chunk_size) if len(X[i:i+chunk_size]) > 0]


# Small terminal progress bar.
class ProgressBar:
    def __init__(self, total: int, desc: str = "Progress", width: int = 30, enabled: bool = True):
        self.total = max(int(total), 1)
        self.desc = desc
        self.width = width
        self.enabled = enabled
        self.start_time = None
        self.last_len = 0

    def _format_time(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def start(self):
        if not self.enabled:
            return
        self.start_time = time.time()
        self.update(0)

    def update(self, n: int, extra: str = ""):
        if not self.enabled:
            return
        if self.start_time is None:
            self.start()
            return
        n = min(max(int(n), 0), self.total)
        frac = n / self.total
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start_time
        rate = n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - n) / rate if rate > 0 else 0.0
        msg = f"\r{self.desc} [{bar}] {n}/{self.total} ({frac*100:5.1f}%) elapsed {self._format_time(elapsed)} eta {self._format_time(eta)}"
        if extra:
            msg += f" | {extra}"
        padding = max(0, self.last_len - len(msg))
        print(msg + " " * padding, end="", flush=True)
        self.last_len = len(msg)

    def close(self, extra: str = ""):
        if not self.enabled:
            return
        self.update(self.total, extra=extra)
        print(flush=True)


# Convert packet-level CSI into window-level samples.
def build_windows(X: np.ndarray, rp: np.ndarray, y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = sanitise_numeric_array(X)
    windows, labels, coords = [], [], []
    for rp_id in sorted(set(rp.tolist())):
        X_rp = X[rp == rp_id]
        y_rp = y[rp == rp_id]
        for wX, wy in zip(chunk_array(X_rp, window_size), chunk_array(y_rp, window_size)):
            windows.append(wX.astype(np.float32))
            labels.append(int(rp_id))
            coords.append(wy[0].astype(np.float32))
    return np.stack(windows, axis=0), np.asarray(labels, dtype=np.int64), np.stack(coords, axis=0)


# Create a class-balanced validation split.
def stratified_window_split(Xw: np.ndarray, yw: np.ndarray, coords: np.ndarray, val_ratio: float, seed: int = 42):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for cls in sorted(set(yw.tolist())):
        idx = np.where(yw == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio))) if len(idx) > 1 else 0
        n_val = min(n_val, len(idx) - 1) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    return (Xw[train_idx], yw[train_idx], coords[train_idx]), (Xw[val_idx], yw[val_idx], coords[val_idx])


# PyTorch dataset for windowed CSI samples.
class WindowDataset(Dataset):
    def __init__(self, Xw: np.ndarray, yw: np.ndarray):
        self.Xw = torch.tensor(Xw, dtype=torch.float32)
        self.yw = torch.tensor(yw, dtype=torch.long)

    def __len__(self):
        return len(self.yw)

    def __getitem__(self, idx):
        return self.Xw[idx], self.yw[idx]


# Trim padded timesteps in causal convolutions.
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size]


# Residual TCN block.
class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)


# Temporal convolutional classifier for CSI windows.
class CSITCNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, window_size: int, proj_dim: int = 64, channels: int = 64, levels: int = 4, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.window_size = window_size
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_ch = proj_dim
        for i in range(levels):
            dilation = 2 ** i
            blocks.append(TemporalBlock(in_ch, channels, kernel_size, dilation, dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):

        b, t, f = x.shape
        x = x[:, : self.window_size, :]
        if t < self.window_size:
            pad = torch.zeros((b, self.window_size - t, f), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        return self.head(x)


# Run one training or validation epoch.
def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds, gts = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(xb)
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        gts.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.int64)
    gts = np.concatenate(gts) if gts else np.zeros((0,), dtype=np.int64)
    acc = float((preds == gts).mean()) if len(gts) else 0.0
    return total_loss / max(len(loader.dataset), 1), acc


# Load a saved neural-network checkpoint.
def load_model(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    meta = ckpt["meta"]
    model = CSITCNClassifier(
        input_dim=meta["input_dim"],
        num_classes=meta["num_classes"],
        window_size=meta["window_size"],
        proj_dim=meta["proj_dim"],
        channels=meta["channels"],
        levels=meta["levels"],
        kernel_size=meta["kernel_size"],
        dropout=meta["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta


# Convert neural-network probabilities to a coordinate estimate.
def probs_to_prediction(probs: np.ndarray, class_to_rp: Dict[int, int], rp_coords: Dict[int, np.ndarray], aggregation: str, top_k: Optional[int]):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    if probs.sum() <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / probs.sum()
    if aggregation == "argmax":
        cls_idx = int(np.argmax(probs))
        rp_id = int(class_to_rp[cls_idx])
        return rp_coords[rp_id].astype(np.float32), rp_id, float(probs[cls_idx])
    order = np.argsort(probs)[::-1]
    if top_k is not None and top_k < len(order):
        order = order[:top_k]
        probs = probs[order]
        probs = probs / probs.sum()
    else:
        probs = probs[order]
    rp_ids = [int(class_to_rp[int(i)]) for i in order]
    coords = np.stack([rp_coords[rp] for rp in rp_ids], axis=0)
    pred_xy = (coords * probs[:, None]).sum(axis=0)
    best_rp = rp_ids[int(np.argmax(probs))]
    return pred_xy.astype(np.float32), best_rp, float(np.max(probs))


# CLI command: train a model.
def cmd_train(args):
    proc = np.load(Path(args.processed), allow_pickle=True)
    X_train = proc["X_train"].astype(np.float32)
    y_train_xy = proc["y_train"].astype(np.float32)
    rp_train = proc["rp_train"].astype(np.int64)
    rp_coords_arr = proc["rp_coords"].astype(np.float32)

    rp_ids = sorted(set(rp_train.tolist()))
    rp_to_class = {rp_id: i for i, rp_id in enumerate(rp_ids)}
    class_to_rp = {i: rp_id for rp_id, i in rp_to_class.items()}
    rp_coords = {int(row[0]): np.array([row[1], row[2]], dtype=np.float32) for row in rp_coords_arr}

    Xw_train, yw_rp, coords_train = build_windows(X_train, rp_train, y_train_xy, args.window_size)
    yw_cls = np.asarray([rp_to_class[int(r)] for r in yw_rp], dtype=np.int64)
    (Xtr, ytr, _), (Xval, yval, _) = stratified_window_split(Xw_train, yw_cls, coords_train, args.val_ratio, seed=args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "models")

    train_loader = DataLoader(WindowDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(WindowDataset(Xval, yval), batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = CSITCNClassifier(
        input_dim=Xtr.shape[2],
        num_classes=len(rp_ids),
        window_size=args.window_size,
        proj_dim=args.proj_dim,
        channels=args.channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val_acc = -1.0
    best_path = out_dir / "models" / "best_tcn.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {row['seconds']:.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            meta = {
                "input_dim": int(Xtr.shape[2]),
                "num_classes": int(len(rp_ids)),
                "window_size": int(args.window_size),
                "proj_dim": int(args.proj_dim),
                "channels": int(args.channels),
                "levels": int(args.levels),
                "kernel_size": int(args.kernel_size),
                "dropout": float(args.dropout),
                "rp_to_class": rp_to_class,
                "class_to_rp": class_to_rp,
                "rp_coords": {int(k): v.tolist() for k, v in rp_coords.items()},
                "seed": int(args.seed),
                "val_ratio": float(args.val_ratio),
            }
            torch.save({"state_dict": model.state_dict(), "meta": meta}, best_path)

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved best model to {best_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


# CLI command: evaluate a trained model.
def cmd_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, meta = load_model(Path(args.model), device)

    proc = np.load(Path(args.processed), allow_pickle=True)
    X_test = proc["X_test"].astype(np.float32)
    y_test_xy = proc["y_test"].astype(np.float32)
    rp_test = proc["rp_test"].astype(np.int64)

    Xw_test, yw_rp, coords_test = build_windows(X_test, rp_test, y_test_xy, meta["window_size"])
    y_true_class = np.asarray([meta["rp_to_class"][int(r)] for r in yw_rp], dtype=np.int64)

    loader = DataLoader(WindowDataset(Xw_test, y_true_class), batch_size=args.batch_size, shuffle=False, drop_last=False)

    rows, probs_all, cls_preds = [], [], []
    progress = ProgressBar(len(loader), desc="Evaluating TCN", enabled=not args.no_progress)
    progress.start()
    seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
            cls_preds.append(np.argmax(probs, axis=1))
            seen += 1
            progress.update(seen)
    progress.close(extra="done")

    probs_all = np.concatenate(probs_all, axis=0)
    cls_preds = np.concatenate(cls_preds, axis=0)
    rp_coords = {int(k): np.array(v, dtype=np.float32) for k, v in meta["rp_coords"].items()}
    class_to_rp = {int(k): int(v) for k, v in meta["class_to_rp"].items()}

    for i, probs in enumerate(probs_all):
        pred_xy, top_rp, top_prob = probs_to_prediction(probs, class_to_rp, rp_coords, args.aggregation, args.top_k)
        true_xy = coords_test[i]
        err = float(np.linalg.norm(pred_xy - true_xy))
        rows.append({
            "true_x": float(true_xy[0]),
            "true_y": float(true_xy[1]),
            "pred_x": float(pred_xy[0]),
            "pred_y": float(pred_xy[1]),
            "error_m": err,
            "true_rp": int(yw_rp[i]),
            "top_rp": int(top_rp),
            "top_prob": float(top_prob),
            "window_idx": int(i),
            "num_packets": int(meta["window_size"]),
        })

    errors = np.array([r["error_m"] for r in rows], dtype=np.float32)
    finite_errors = errors[np.isfinite(errors)]
    if len(finite_errors) == 0:
        raise ValueError("All localisation errors are non-finite.")

    metrics = {
        "num_estimates": int(len(rows)),
        "num_finite_estimates": int(len(finite_errors)),
        "mean_error_m": float(np.mean(finite_errors)),
        "median_error_m": float(np.median(finite_errors)),
        "p90_error_m": float(np.percentile(finite_errors, 90)),
        "within_1m": float(np.mean(finite_errors <= 1.0)),
        "within_2m": float(np.mean(finite_errors <= 2.0)),
        "within_3m": float(np.mean(finite_errors <= 3.0)),
        "classification_accuracy_windowed": float((cls_preds == y_true_class).mean()),
        "aggregation": args.aggregation,
        "window_size": int(meta["window_size"]),
    }
    if f1_score is not None:
        metrics["macro_f1_windowed"] = float(f1_score(y_true_class, cls_preds, average="macro"))
        metrics["weighted_f1_windowed"] = float(f1_score(y_true_class, cls_preds, average="weighted"))

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {out_dir / 'predictions.csv'}")
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


# Define command-line interface.
def build_argparser():
    p = argparse.ArgumentParser(description="Windowed CSI TCN for indoor localisation")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train TCN on processed_dataset.npz")
    p_train.add_argument("--processed", required=True)
    p_train.add_argument("--out-dir", required=True)
    p_train.add_argument("--window-size", type=int, default=10)
    p_train.add_argument("--proj-dim", type=int, default=64)
    p_train.add_argument("--channels", type=int, default=64)
    p_train.add_argument("--levels", type=int, default=4)
    p_train.add_argument("--kernel-size", type=int, default=3)
    p_train.add_argument("--dropout", type=float, default=0.2)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--val-ratio", type=float, default=0.1)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--cpu", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate trained TCN on processed_dataset.npz")
    p_eval.add_argument("--processed", required=True)
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--out-dir", required=True)
    p_eval.add_argument("--batch-size", type=int, default=128)
    p_eval.add_argument("--aggregation", choices=["argmax", "weighted"], default="argmax")
    p_eval.add_argument("--top-k", type=int, default=None)
    p_eval.add_argument("--cpu", action="store_true")
    p_eval.add_argument("--no-progress", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)
    return p


# Program entry point.
def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
