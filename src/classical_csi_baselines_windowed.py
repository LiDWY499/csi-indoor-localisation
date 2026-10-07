from __future__ import annotations

# CSI localisation script

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


# Create output folders when needed.
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Small terminal progress bar.
class ProgressBar:
    def __init__(self, total: int, desc: str = "Progress", width: int = 30):
        self.total = max(int(total), 1)
        self.desc = desc
        self.width = width
        self.start = None
        self.last_len = 0

    def _fmt(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def update(self, n: int, extra: str = "") -> None:
        if self.start is None:
            self.start = time.time()
        n = min(max(int(n), 0), self.total)
        frac = n / self.total
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        rate = n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - n) / rate if rate > 0 else 0.0
        msg = f"\r{self.desc} [{bar}] {n}/{self.total} ({frac*100:5.1f}%) elapsed {self._fmt(elapsed)} eta {self._fmt(eta)}"
        if extra:
            msg += f" | {extra}"
        pad = max(0, self.last_len - len(msg))
        print(msg + " " * pad, end="", flush=True)
        self.last_len = len(msg)
        if n == self.total:
            print(flush=True)


# Load the shared processed_dataset.npz format.
def load_processed_dataset(path: Path):
    data = np.load(path, allow_pickle=True)
    X_train = np.asarray(data["X_train"], dtype=np.float32)
    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.float32)
    rp_train = np.asarray(data["rp_train"], dtype=np.int64)
    rp_test = np.asarray(data["rp_test"], dtype=np.int64)
    rp_coords_arr = np.asarray(data["rp_coords"], dtype=np.float32)
    rp_coords = {int(row[0]): np.array([row[1], row[2]], dtype=np.float32) for row in rp_coords_arr}
    return X_train, X_test, y_train, y_test, rp_train, rp_test, rp_coords


# Split packet streams into fixed-size windows.
def chunk_array(X: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    if chunk_size <= 1:
        return [X[i:i+1] for i in range(len(X))]
    return [X[i:i+chunk_size] for i in range(0, len(X), chunk_size) if len(X[i:i+chunk_size]) > 0]


# Calculate localisation error metrics.
def metrics_from_predictions(true_xy: np.ndarray, pred_xy: np.ndarray) -> Dict[str, float]:
    errors = np.linalg.norm(pred_xy - true_xy, axis=1).astype(np.float32)
    finite = errors[np.isfinite(errors)]
    if len(finite) == 0:
        raise ValueError("No finite localisation errors were produced.")
    return {
        "num_estimates": int(len(errors)),
        "num_finite_estimates": int(len(finite)),
        "mean_error_m": float(np.mean(finite)),
        "median_error_m": float(np.median(finite)),
        "p90_error_m": float(np.percentile(finite, 90)),
        "within_1m": float(np.mean(finite <= 1.0)),
        "within_2m": float(np.mean(finite <= 2.0)),
        "within_3m": float(np.mean(finite <= 3.0)),
    }


# Construct classical baseline models.
def build_models(args) -> Dict[str, object]:
    return {
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=args.knn_k, weights=args.knn_weights)),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                C=args.svm_c,
                kernel=args.svm_kernel,
                gamma=args.svm_gamma,
                probability=True,
                random_state=SEED,
            )),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            n_jobs=args.jobs,
            random_state=SEED,
        ),
    }


# Convert class probabilities to a coordinate estimate.
def posterior_to_prediction(
    posterior: np.ndarray,
    classes_: np.ndarray,
    rp_coords: Dict[int, np.ndarray],
    aggregation: str,
    top_k: Optional[int],
):
    posterior = np.asarray(posterior, dtype=np.float64)
    posterior = np.nan_to_num(posterior, nan=0.0, posinf=0.0, neginf=0.0)
    if posterior.sum() <= 0:
        posterior = np.ones_like(posterior) / len(posterior)
    else:
        posterior = posterior / posterior.sum()

    order = np.argsort(posterior)[::-1]
    if top_k is not None and top_k > 0:
        order = order[:min(top_k, len(order))]

    selected_posterior = posterior[order]
    selected_classes = classes_[order]

    if aggregation == "argmax":
        pred_rp = int(selected_classes[np.argmax(selected_posterior)])
        pred_xy = rp_coords[pred_rp].astype(np.float32)
        top_prob = float(np.max(selected_posterior))
    else:
        weights = selected_posterior / max(selected_posterior.sum(), 1e-12)
        coords = np.stack([rp_coords[int(lbl)] for lbl in selected_classes], axis=0)
        pred_xy = (coords * weights[:, None]).sum(axis=0).astype(np.float32)
        pred_rp = int(selected_classes[np.argmax(weights)])
        top_prob = float(np.max(weights))

    posterior_dict = {int(classes_[i]): float(posterior[i]) for i in range(len(classes_))}
    return pred_xy, pred_rp, top_prob, posterior_dict


# Evaluate classical models with windowed aggregation.
def evaluate_windowed_model(model, X_test, y_test, rp_test, rp_coords, args):
    classes_ = np.asarray(model.classes_)
    pred_rows = []
    true_rps = []
    pred_rps = []

    unique_rps = sorted(set(rp_test.tolist()))
    total_windows = 0
    rp_window_counts = {}
    for rp_id in unique_rps:
        count = len(chunk_array(X_test[rp_test == rp_id], args.window_size))
        rp_window_counts[rp_id] = count
        total_windows += count

    progress = ProgressBar(total_windows, desc="Evaluating windowed baselines")
    done = 0
    infer_start = time.time()

    for rp_idx, rp_id in enumerate(unique_rps, start=1):
        X_rp = X_test[rp_test == rp_id]
        y_rp = y_test[rp_test == rp_id]
        windows = chunk_array(X_rp, args.window_size)
        y_windows = chunk_array(y_rp, args.window_size)

        for window_idx, (wX, wy) in enumerate(zip(windows, y_windows), start=1):
            packet_posteriors = model.predict_proba(wX)
            window_posterior = np.mean(packet_posteriors, axis=0)
            pred_xy, pred_rp, top_prob, posterior = posterior_to_prediction(
                window_posterior,
                classes_,
                rp_coords,
                aggregation=args.aggregation,
                top_k=args.top_k,
            )
            true_xy = wy[0].astype(np.float32)
            err = float(np.linalg.norm(pred_xy - true_xy))
            pred_rows.append({
                "true_x": float(true_xy[0]),
                "true_y": float(true_xy[1]),
                "pred_x": float(pred_xy[0]),
                "pred_y": float(pred_xy[1]),
                "error_m": err,
                "true_rp": int(rp_id),
                "pred_rp": int(pred_rp),
                "top_prob": float(top_prob),
                "window_idx": int(window_idx - 1),
                "num_packets": int(len(wX)),
            })
            true_rps.append(int(rp_id))
            pred_rps.append(int(pred_rp))
            done += 1
            progress.update(done, extra=f"RP {rp_idx}/{len(unique_rps)} (id={rp_id}) | window {window_idx}/{rp_window_counts[rp_id]}")

    infer_seconds = time.time() - infer_start
    y_true_xy = np.array([[r["true_x"], r["true_y"]] for r in pred_rows], dtype=np.float32)
    y_pred_xy = np.array([[r["pred_x"], r["pred_y"]] for r in pred_rows], dtype=np.float32)

    metrics = metrics_from_predictions(y_true_xy, y_pred_xy)
    metrics["classification_accuracy_windowed"] = float(accuracy_score(true_rps, pred_rps))
    report = classification_report(true_rps, pred_rps, output_dict=True, zero_division=0)
    metrics["macro_f1_windowed"] = float(report["macro avg"]["f1-score"])
    metrics["weighted_f1_windowed"] = float(report["weighted avg"]["f1-score"])
    metrics["infer_seconds_total"] = float(infer_seconds)
    metrics["infer_ms_per_estimate"] = float(1000.0 * infer_seconds / max(len(pred_rows), 1))
    return metrics, pred_rows


# Write per-window or per-packet predictions to CSV.
def save_predictions_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Program entry point.
def main():
    p = argparse.ArgumentParser(description="DeepFi-like classical CSI baselines with windowed aggregation")
    p.add_argument("--processed", required=True, help="Path to processed_dataset.npz")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--window-size", type=int, default=10, help="Packets per estimate")
    p.add_argument("--aggregation", choices=["weighted", "argmax"], default="weighted", help="How to convert RP posterior to final location")
    p.add_argument("--top-k", type=int, default=None, help="Use only top-k RPs for weighted averaging")
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--knn-weights", choices=["uniform", "distance"], default="distance")
    p.add_argument("--svm-c", type=float, default=10.0)
    p.add_argument("--svm-kernel", choices=["linear", "rbf", "poly", "sigmoid"], default="rbf")
    p.add_argument("--svm-gamma", default="scale")
    p.add_argument("--rf-trees", type=int, default=500)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--rf-min-samples-leaf", type=int, default=1)
    p.add_argument("--jobs", type=int, default=-1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "models")

    X_train, X_test, y_train, y_test, rp_train, rp_test, rp_coords = load_processed_dataset(Path(args.processed))
    models = build_models(args)
    per_model = {}
    ranking = []

    print(
        f"Loaded processed dataset: train={len(X_train)} test={len(X_test)} features={X_train.shape[1]} "
        f"RPs={len(rp_coords)} window_size={args.window_size} aggregation={args.aggregation} top_k={args.top_k}"
    )

    train_bar = ProgressBar(len(models), desc="Training models")
    for idx, (name, model) in enumerate(models.items(), start=1):
        print(f"\n[{name}] fitting...")
        t0 = time.time()
        model.fit(X_train, rp_train)
        train_seconds = time.time() - t0
        joblib.dump(model, out_dir / "models" / f"{name}_windowed.joblib")

        print(f"[{name}] evaluating with windowed aggregation...")
        metrics, rows = evaluate_windowed_model(model, X_test, y_test, rp_test, rp_coords, args)
        metrics["train_seconds"] = float(train_seconds)
        per_model[name] = metrics
        ranking.append((name, metrics["mean_error_m"]))

        save_predictions_csv(out_dir / f"predictions_{name}.csv", rows)
        train_bar.update(idx, extra=f"{name} done | mean={metrics['mean_error_m']:.3f}m | win_acc={metrics['classification_accuracy_windowed']:.4f}")
        print(json.dumps(metrics, indent=2))

    ranking.sort(key=lambda x: x[1])
    summary = {
        "window_size": int(args.window_size),
        "aggregation": args.aggregation,
        "top_k": args.top_k,
        "best_model_by_mean_error": ranking[0][0],
        "ranking_by_mean_error": [{"model": name, "mean_error_m": float(score)} for name, score in ranking],
    }

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "metrics_per_model.json", "w", encoding="utf-8") as f:
        json.dump(per_model, f, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_dir / 'metrics_summary.json'}")
    print(f"Saved per-model metrics to {out_dir / 'metrics_per_model.json'}")


if __name__ == "__main__":
    main()
