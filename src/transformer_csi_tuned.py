from __future__ import annotations

# Tuned Transformer and hyperparameter search.

import argparse
import csv
import itertools
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass, asdict
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


# Fix random seeds for reproducible runs.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
        return [X[i : i + 1] for i in range(len(X))]
    return [X[i : i + chunk_size] for i in range(0, len(X), chunk_size) if len(X[i : i + chunk_size]) > 0]


# Small terminal progress bar.
class ProgressBar:
    def __init__(self, total: int, desc: str = "Progress", width: int = 30, enabled: bool = True):
        self.total = max(int(total), 1)
        self.desc = desc
        self.width = width
        self.enabled = enabled
        self.start_time: Optional[float] = None
        self.last_len = 0

    def _format_time(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def start(self) -> None:
        if not self.enabled:
            return
        self.start_time = time.time()
        self.update(0)

    def update(self, n: int, extra: str = "") -> None:
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

    def close(self, extra: str = "") -> None:
        if not self.enabled:
            return
        self.update(self.total, extra=extra)
        print(flush=True)


# Convert packet-level CSI into window-level samples.
def build_windows(X: np.ndarray, rp: np.ndarray, y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = sanitise_numeric_array(X)
    windows: List[np.ndarray] = []
    labels: List[int] = []
    coords: List[np.ndarray] = []
    for rp_id in sorted(set(rp.tolist())):
        X_rp = X[rp == rp_id]
        y_rp = y[rp == rp_id]
        for wX, wy in zip(chunk_array(X_rp, window_size), chunk_array(y_rp, window_size)):
            if len(wX) < window_size:

                pad = np.zeros((window_size - len(wX), X.shape[1]), dtype=np.float32)
                wX = np.concatenate([wX, pad], axis=0)
            windows.append(wX.astype(np.float32))
            labels.append(int(rp_id))
            coords.append(wy[0].astype(np.float32))
    return np.stack(windows, axis=0), np.asarray(labels, dtype=np.int64), np.stack(coords, axis=0)


# Create a class-balanced validation split.
def stratified_window_split(Xw: np.ndarray, yw: np.ndarray, coords: np.ndarray, val_ratio: float, seed: int = 42):
    rng = np.random.RandomState(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
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

    def __len__(self) -> int:
        return len(self.yw)

    def __getitem__(self, idx: int):
        return self.Xw[idx], self.yw[idx]


# Prepared arrays and label mappings.
@dataclass
class DataBundle:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xval: np.ndarray
    yval: np.ndarray
    Xtest: np.ndarray
    ytest_cls: np.ndarray
    ytest_rp: np.ndarray
    test_coords: np.ndarray
    rp_ids: List[int]
    rp_to_class: Dict[int, int]
    class_to_rp: Dict[int, int]
    rp_coords: Dict[int, np.ndarray]
    input_dim: int
    num_classes: int
    window_size: int


def load_data_bundle(processed: Path, window_size: int, val_ratio: float, seed: int) -> DataBundle:
    proc = np.load(processed, allow_pickle=True)
    X_train = proc["X_train"].astype(np.float32)
    y_train_xy = proc["y_train"].astype(np.float32)
    rp_train = proc["rp_train"].astype(np.int64)
    X_test = proc["X_test"].astype(np.float32)
    y_test_xy = proc["y_test"].astype(np.float32)
    rp_test = proc["rp_test"].astype(np.int64)
    rp_coords_arr = proc["rp_coords"].astype(np.float32)

    rp_ids = sorted(set(rp_train.tolist()))
    rp_to_class = {int(rp_id): i for i, rp_id in enumerate(rp_ids)}
    class_to_rp = {i: int(rp_id) for rp_id, i in rp_to_class.items()}
    rp_coords = {int(row[0]): np.array([row[1], row[2]], dtype=np.float32) for row in rp_coords_arr}

    Xw_train, yw_rp, coords_train = build_windows(X_train, rp_train, y_train_xy, window_size)
    yw_cls = np.asarray([rp_to_class[int(r)] for r in yw_rp], dtype=np.int64)
    (Xtr, ytr, _), (Xval, yval, _) = stratified_window_split(Xw_train, yw_cls, coords_train, val_ratio, seed=seed)

    Xw_test, ytest_rp, test_coords = build_windows(X_test, rp_test, y_test_xy, window_size)
    ytest_cls = np.asarray([rp_to_class[int(r)] for r in ytest_rp], dtype=np.int64)

    return DataBundle(
        Xtr=Xtr,
        ytr=ytr,
        Xval=Xval,
        yval=yval,
        Xtest=Xw_test,
        ytest_cls=ytest_cls,
        ytest_rp=ytest_rp,
        test_coords=test_coords,
        rp_ids=rp_ids,
        rp_to_class=rp_to_class,
        class_to_rp=class_to_rp,
        rp_coords=rp_coords,
        input_dim=int(Xtr.shape[2]),
        num_classes=len(rp_ids),
        window_size=window_size,
    )


# Transformer classifier for CSI windows.
class CSITransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        window_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        norm_first: bool = False,
        pooling: str = "cls",
        input_dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.window_size = window_size
        self.pooling = pooling
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(input_dropout),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, t, f = x.shape
        if t != self.window_size:
            if t < self.window_size:
                pad = torch.zeros((b, self.window_size - t, f), dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=1)
            else:
                x = x[:, : self.window_size, :]
        tok = self.input_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        tok = tok + self.pos_embed[:, : tok.shape[1], :]
        enc = self.encoder(tok)
        if self.pooling == "cls":
            pooled = enc[:, 0]
        else:
            pooled = enc[:, 1:].mean(dim=1)
        pooled = self.norm(pooled)
        return self.head(pooled)


# Transformer training configuration.
@dataclass
class TrainConfig:
    window_size: int = 10
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    input_dropout: float = 0.0
    norm_first: bool = False
    pooling: str = "cls"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    val_ratio: float = 0.1
    patience: int = 12
    min_delta: float = 1e-5
    seed: int = 42
    cpu: bool = False


# Select CUDA when available unless CPU is requested.
def get_device(cpu: bool = False) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")


# Create a PyTorch DataLoader for CSI windows.
def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


# Run one training or validation epoch.
def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip: float = 0.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += float(loss.item()) * len(xb)
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        gts.append(yb.detach().cpu().numpy())
    preds_np = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.int64)
    gts_np = np.concatenate(gts) if gts else np.zeros((0,), dtype=np.int64)
    acc = float((preds_np == gts_np).mean()) if len(gts_np) else 0.0
    return total_loss / max(len(loader.dataset), 1), acc


# Save model weights and metadata together.
def save_model(path: Path, model: nn.Module, cfg: TrainConfig, data: DataBundle, val_acc: float, val_loss: float) -> None:
    meta = {
        "input_dim": data.input_dim,
        "num_classes": data.num_classes,
        "window_size": cfg.window_size,
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "num_layers": cfg.num_layers,
        "dim_feedforward": cfg.dim_feedforward,
        "dropout": cfg.dropout,
        "input_dropout": cfg.input_dropout,
        "norm_first": cfg.norm_first,
        "pooling": cfg.pooling,
        "rp_to_class": data.rp_to_class,
        "class_to_rp": data.class_to_rp,
        "rp_coords": {int(k): v.tolist() for k, v in data.rp_coords.items()},
        "seed": cfg.seed,
        "val_ratio": cfg.val_ratio,
        "best_val_acc": float(val_acc),
        "best_val_loss": float(val_loss),
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta, "config": asdict(cfg)}, path)


# Train one Transformer configuration.
def train_once(processed: Path, out_dir: Path, cfg: TrainConfig, no_progress: bool = False) -> Dict:
    set_seed(cfg.seed)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "models")
    data = load_data_bundle(processed, cfg.window_size, cfg.val_ratio, cfg.seed)
    device = get_device(cfg.cpu)

    print(f"Using device: {device}")
    print(f"Windows: train={len(data.Xtr)} val={len(data.Xval)} test={len(data.Xtest)} | RPs={data.num_classes} | window={cfg.window_size}")

    train_loader = make_loader(WindowDataset(data.Xtr, data.ytr), cfg.batch_size, True, device)
    val_loader = make_loader(WindowDataset(data.Xval, data.yval), cfg.batch_size, False, device)

    model = CSITransformerClassifier(
        input_dim=data.input_dim,
        num_classes=data.num_classes,
        window_size=cfg.window_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        input_dropout=cfg.input_dropout,
        norm_first=cfg.norm_first,
        pooling=cfg.pooling,
    ).to(device)

    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    except TypeError:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_path = out_dir / "models" / "best_transformer.pt"
    history: List[Dict] = []
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, grad_clip=cfg.grad_clip)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": lr_now,
            "seconds": elapsed,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr_now:.2e} | {elapsed:.1f}s"
        )
        improved = (val_acc > best_val_acc + cfg.min_delta) or (abs(val_acc - best_val_acc) <= cfg.min_delta and val_loss < best_val_loss)
        if improved:
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)
            save_model(best_path, model, cfg, data, best_val_acc, best_val_loss)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if cfg.patience > 0 and epochs_without_improvement >= cfg.patience:
                print(f"Early stopping at epoch {epoch}: no validation improvement for {cfg.patience} epochs.")
                break

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Saved best model to {best_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f} | best validation loss: {best_val_loss:.6f}")
    return {"best_model": str(best_path), "best_val_acc": best_val_acc, "best_val_loss": best_val_loss, "history": history}


# Load a saved neural-network checkpoint.
def load_model(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = CSITransformerClassifier(
        input_dim=meta["input_dim"],
        num_classes=meta["num_classes"],
        window_size=meta["window_size"],
        d_model=meta["d_model"],
        nhead=meta["nhead"],
        num_layers=meta["num_layers"],
        dim_feedforward=meta["dim_feedforward"],
        dropout=meta["dropout"],
        input_dropout=meta.get("input_dropout", 0.0),
        norm_first=meta.get("norm_first", False),
        pooling=meta.get("pooling", "cls"),
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
    probs_sel = probs[order]
    if probs_sel.sum() <= 0:
        probs_sel = np.ones_like(probs_sel) / len(probs_sel)
    else:
        probs_sel = probs_sel / probs_sel.sum()
    rp_ids = [int(class_to_rp[int(i)]) for i in order]
    coords = np.stack([rp_coords[rp] for rp in rp_ids], axis=0)
    pred_xy = (coords * probs_sel[:, None]).sum(axis=0)
    best_rp = rp_ids[int(np.argmax(probs_sel))]
    return pred_xy.astype(np.float32), best_rp, float(np.max(probs_sel))


# Evaluate a saved Transformer model.
def evaluate_model(processed: Path, model_path: Path, out_dir: Optional[Path], aggregation: str, top_k: Optional[int], batch_size: int, cpu: bool, no_progress: bool) -> Dict:
    device = get_device(cpu)
    model, meta = load_model(model_path, device)
    data = load_data_bundle(processed, int(meta["window_size"]), float(meta.get("val_ratio", 0.1)), int(meta.get("seed", 42)))

    loader = make_loader(WindowDataset(data.Xtest, data.ytest_cls), batch_size, False, device)
    rows: List[Dict] = []
    probs_all: List[np.ndarray] = []
    cls_preds: List[np.ndarray] = []

    progress = ProgressBar(len(loader), desc="Evaluating transformer", enabled=not no_progress)
    progress.start()
    with torch.no_grad():
        for step, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
            cls_preds.append(np.argmax(probs, axis=1))
            progress.update(step)
    progress.close(extra="done")

    probs_all_np = np.concatenate(probs_all, axis=0)
    cls_preds_np = np.concatenate(cls_preds, axis=0)
    rp_coords = {int(k): np.array(v, dtype=np.float32) for k, v in meta["rp_coords"].items()}
    class_to_rp = {int(k): int(v) for k, v in meta["class_to_rp"].items()}

    for i, probs in enumerate(probs_all_np):
        pred_xy, top_rp, top_prob = probs_to_prediction(probs, class_to_rp, rp_coords, aggregation, top_k)
        true_xy = data.test_coords[i]
        err = float(np.linalg.norm(pred_xy - true_xy))
        rows.append({
            "true_x": float(true_xy[0]),
            "true_y": float(true_xy[1]),
            "pred_x": float(pred_xy[0]),
            "pred_y": float(pred_xy[1]),
            "error_m": err,
            "true_rp": int(data.ytest_rp[i]),
            "top_rp": int(top_rp),
            "top_prob": float(top_prob),
            "window_idx": int(i),
            "num_packets": int(meta["window_size"]),
        })

    errors = np.array([r["error_m"] for r in rows], dtype=np.float32)
    finite_errors = errors[np.isfinite(errors)]
    if len(finite_errors) == 0:
        raise ValueError("All localisation errors are non-finite.")

    metrics: Dict[str, float | int | str | None] = {
        "num_estimates": int(len(rows)),
        "num_finite_estimates": int(len(finite_errors)),
        "mean_error_m": float(np.mean(finite_errors)),
        "median_error_m": float(np.median(finite_errors)),
        "p90_error_m": float(np.percentile(finite_errors, 90)),
        "within_1m": float(np.mean(finite_errors <= 1.0)),
        "within_2m": float(np.mean(finite_errors <= 2.0)),
        "within_3m": float(np.mean(finite_errors <= 3.0)),
        "classification_accuracy_windowed": float((cls_preds_np == data.ytest_cls).mean()),
        "aggregation": aggregation,
        "top_k": int(top_k) if top_k is not None else None,
        "window_size": int(meta["window_size"]),
    }
    if f1_score is not None:
        metrics["macro_f1_windowed"] = float(f1_score(data.ytest_cls, cls_preds_np, average="macro"))
        metrics["weighted_f1_windowed"] = float(f1_score(data.ytest_cls, cls_preds_np, average="weighted"))

    if out_dir is not None:
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
    return metrics


# Return preset Transformer tuning configurations.
def get_tune_configs(preset: str, base: TrainConfig) -> List[TrainConfig]:

    configs: List[TrainConfig] = []
    if preset == "quick":
        grid = [
            dict(d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.20, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="cls"),
            dict(d_model=96, nhead=4, num_layers=2, dim_feedforward=192, dropout=0.15, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="cls"),
            dict(d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.20, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="cls"),
            dict(d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.10, input_dropout=0.00, lr=1e-3, weight_decay=1e-4, label_smoothing=0.00, pooling="cls"),
        ]
    elif preset == "overnight":
        grid = [
            dict(d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.10, input_dropout=0.00, lr=1e-3, weight_decay=1e-4, label_smoothing=0.00, pooling="cls"),
            dict(d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.20, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="cls"),
            dict(d_model=96, nhead=4, num_layers=2, dim_feedforward=192, dropout=0.15, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="cls"),
            dict(d_model=96, nhead=4, num_layers=3, dim_feedforward=192, dropout=0.20, input_dropout=0.05, lr=3e-4, weight_decay=3e-4, label_smoothing=0.05, pooling="cls"),
            dict(d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.20, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="cls"),
            dict(d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.20, input_dropout=0.05, lr=3e-4, weight_decay=3e-4, label_smoothing=0.05, pooling="cls"),
            dict(d_model=128, nhead=8, num_layers=3, dim_feedforward=384, dropout=0.15, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="cls"),
            dict(d_model=192, nhead=8, num_layers=3, dim_feedforward=384, dropout=0.10, input_dropout=0.00, lr=5e-4, weight_decay=1e-4, label_smoothing=0.00, pooling="cls"),
            dict(d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.20, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="mean"),
        ]
    else:
        grid = [
            dict(d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.20, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="cls"),
            dict(d_model=96, nhead=4, num_layers=2, dim_feedforward=192, dropout=0.15, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="cls"),
            dict(d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.20, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="cls"),
            dict(d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.20, input_dropout=0.05, lr=3e-4, weight_decay=3e-4, label_smoothing=0.05, pooling="cls"),
            dict(d_model=128, nhead=8, num_layers=3, dim_feedforward=384, dropout=0.15, input_dropout=0.05, lr=5e-4, weight_decay=1e-4, label_smoothing=0.02, pooling="cls"),
            dict(d_model=192, nhead=8, num_layers=3, dim_feedforward=384, dropout=0.10, input_dropout=0.00, lr=5e-4, weight_decay=1e-4, label_smoothing=0.00, pooling="cls"),
        ]

    for g in grid:
        cfg = TrainConfig(**asdict(base))
        for k, v in g.items():
            setattr(cfg, k, v)
        configs.append(cfg)
    return configs


# Convert command-line arguments into a training config.
def args_to_config(args) -> TrainConfig:
    return TrainConfig(
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        norm_first=args.norm_first,
        pooling=args.pooling,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        val_ratio=args.val_ratio,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        cpu=args.cpu,
    )


# CLI command: train a model.
def cmd_train(args) -> None:
    cfg = args_to_config(args)
    train_once(Path(args.processed), Path(args.out_dir), cfg, no_progress=False)


# CLI command: evaluate a trained model.
def cmd_evaluate(args) -> None:
    evaluate_model(
        processed=Path(args.processed),
        model_path=Path(args.model),
        out_dir=Path(args.out_dir),
        aggregation=args.aggregation,
        top_k=args.top_k,
        batch_size=args.batch_size,
        cpu=args.cpu,
        no_progress=args.no_progress,
    )


# CLI command: run Transformer tuning experiments.
def cmd_tune(args) -> None:
    base = TrainConfig(
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        patience=args.patience,
        seed=args.seed,
        cpu=args.cpu,
    )
    configs = get_tune_configs(args.preset, base)
    if args.max_runs is not None:
        configs = configs[: args.max_runs]

    root = Path(args.out_dir)
    ensure_dir(root)
    results: List[Dict] = []

    print(f"Running {len(configs)} Transformer tuning runs | preset={args.preset} | processed={args.processed}")
    for i, cfg in enumerate(configs, start=1):
        run_name = (
            f"run_{i:02d}_w{cfg.window_size}_d{cfg.d_model}_h{cfg.nhead}_l{cfg.num_layers}_"
            f"ff{cfg.dim_feedforward}_do{cfg.dropout}_lr{cfg.lr}_wd{cfg.weight_decay}_ls{cfg.label_smoothing}_{cfg.pooling}"
        ).replace(".", "p")
        run_dir = root / run_name
        print("\n" + "=" * 88)
        print(f"TUNE RUN {i}/{len(configs)}: {run_name}")
        print(json.dumps(asdict(cfg), indent=2))
        t0 = time.time()
        train_info = train_once(Path(args.processed), run_dir, cfg, no_progress=True)
        model_path = Path(train_info["best_model"])

        eval_argmax_dir = run_dir / "eval_argmax"
        m_argmax = evaluate_model(Path(args.processed), model_path, eval_argmax_dir, "argmax", None, args.eval_batch_size, cfg.cpu, args.no_progress)
        eval_weighted_dir = run_dir / "eval_weighted_top3"
        m_weighted = evaluate_model(Path(args.processed), model_path, eval_weighted_dir, "weighted", 3, args.eval_batch_size, cfg.cpu, args.no_progress)

        row = {
            "run": i,
            "run_name": run_name,
            "model_path": str(model_path),
            "seconds_total": time.time() - t0,
            "best_val_acc": train_info["best_val_acc"],
            "best_val_loss": train_info["best_val_loss"],
            "argmax_mean_error_m": m_argmax["mean_error_m"],
            "argmax_within_1m": m_argmax["within_1m"],
            "argmax_accuracy": m_argmax["classification_accuracy_windowed"],
            "weighted_top3_mean_error_m": m_weighted["mean_error_m"],
            "weighted_top3_within_1m": m_weighted["within_1m"],
            "weighted_top3_accuracy": m_weighted["classification_accuracy_windowed"],
            **asdict(cfg),
        }
        results.append(row)
        with open(root / "tune_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        write_tune_csv(root / "tune_results.csv", results)
        print(f"Current best by argmax mean error: {min(results, key=lambda x: x['argmax_mean_error_m'])['run_name']}")

    results_sorted = sorted(results, key=lambda x: x["argmax_mean_error_m"])
    with open(root / "tune_results_ranked.json", "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, indent=2)
    write_tune_csv(root / "tune_results_ranked.csv", results_sorted)
    print("\nFinal ranking by argmax mean error:")
    for r in results_sorted[:10]:
        print(f"#{r['run']:02d} {r['run_name']} | argmax_mean={r['argmax_mean_error_m']:.4f} | weighted_top3_mean={r['weighted_top3_mean_error_m']:.4f} | acc={r['argmax_accuracy']:.4f}")
    print(f"Saved tuning table to {root / 'tune_results_ranked.csv'}")


# Save tuning results as a CSV table.
def write_tune_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# Register shared training arguments.
def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--processed", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input-dropout", type=float, default=0.0)
    parser.add_argument("--norm-first", action="store_true", help="Use pre-norm transformer blocks. May show a PyTorch nested tensor warning; harmless.")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")


# Define command-line interface.
def build_argparser():
    p = argparse.ArgumentParser(description="Tuned windowed CSI Transformer for indoor localisation")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train one tuned Transformer configuration")
    add_train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a trained Transformer")
    p_eval.add_argument("--processed", required=True)
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--out-dir", required=True)
    p_eval.add_argument("--batch-size", type=int, default=256)
    p_eval.add_argument("--aggregation", choices=["argmax", "weighted"], default="argmax")
    p_eval.add_argument("--top-k", type=int, default=None)
    p_eval.add_argument("--cpu", action="store_true")
    p_eval.add_argument("--no-progress", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_tune = sub.add_parser("tune", help="Run a Transformer hyperparameter search")
    p_tune.add_argument("--processed", required=True)
    p_tune.add_argument("--out-dir", required=True)
    p_tune.add_argument("--window-size", type=int, default=10)
    p_tune.add_argument("--preset", choices=["quick", "balanced", "overnight"], default="balanced")
    p_tune.add_argument("--epochs", type=int, default=40)
    p_tune.add_argument("--batch-size", type=int, default=64)
    p_tune.add_argument("--eval-batch-size", type=int, default=256)
    p_tune.add_argument("--val-ratio", type=float, default=0.1)
    p_tune.add_argument("--patience", type=int, default=10)
    p_tune.add_argument("--max-runs", type=int, default=None)
    p_tune.add_argument("--seed", type=int, default=42)
    p_tune.add_argument("--cpu", action="store_true")
    p_tune.add_argument("--no-progress", action="store_true")
    p_tune.set_defaults(func=cmd_tune)
    return p


# Program entry point.
def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
