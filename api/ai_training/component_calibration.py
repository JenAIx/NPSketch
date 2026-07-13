#!/usr/bin/env python3
"""
Post-hoc calibration for the component model — no retraining required.

The component CNN outputs 60 sub-label logits; the derived Total_Score is currently
`sum(prob >= 0.5)`. Two cheap, leakage-free post-processing steps materially improve
both the per-component decisions and the derived score:

  1. Per-label decision thresholds  thr[60]  (argmax-F1 per sub-label on TRAIN)
     instead of a fixed 0.5 — helps the weak / imbalanced sub-labels.
  2. Calibrated score readout  score = Σ wⱼ·pⱼ + b  (non-negative least squares on
     TRAIN probabilities → Total_Score) — directly MSE-optimal, so it lifts derived
     R²/RMSE far beyond a naive hard- or soft-sum, especially in the low-score tail.

Everything is fit on the model's stored TRAIN image ids and evaluated on the held-out
VAL ids (patient-disjoint — the same split the model trained with). With --write the
thresholds + calibration are saved into the model's *_metadata.json so inference
(predict-single) and future training reports can use them.

Run (measure only):
  docker exec -e PYTHONPATH=/app npsketch-api python3 /app/ai_training/component_calibration.py
Run (measure + persist into metadata):
  docker exec -e PYTHONPATH=/app npsketch-api python3 /app/ai_training/component_calibration.py --write
"""
import os
import io
import sys
import json
import glob
import argparse

import numpy as np
import torch
from scipy.optimize import nnls

from database import get_db, TrainingDataImage
from ai_training.model import DrawingClassifier
from ai_training.dataset import components_to_vector
from ai_training.preprocessing import preprocess_bytes_for_prediction

MODELS_GLOB = "/app/data/models/model_Components_*.pth"
THRESH_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def f1_bin(y_true, y_pred):
    """Binary F1 (no sklearn dependency)."""
    y_true = y_true.astype(bool); y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def latest_model():
    paths = sorted(glob.glob(MODELS_GLOB))
    if not paths:
        sys.exit("No component model found.")
    p = paths[-1]
    return p, p.replace(".pth", "_metadata.json")


def load_model(model_path, metadata):
    num_outputs = metadata.get("model", {}).get("output_neurons", 60)
    model = DrawingClassifier(num_outputs=num_outputs, pretrained=False, use_sigmoid=False)
    ck = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck)
    model.eval()
    return model


def infer(model, rows, metadata, batch=64):
    """Return (probs[N,60], targets[N,60]) for the given DB rows."""
    probs, tgts = [], []
    buf = []
    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).unsqueeze(1).float()  # [B,1,H,W]
        with torch.no_grad():
            p = torch.sigmoid(model(x)).numpy()
        probs.append(p)
        buf.clear()
    for r in rows:
        vec = components_to_vector(json.loads(r.features_data))
        if vec is None:
            continue
        arr = preprocess_bytes_for_prediction(r.processed_image_data, metadata=metadata, debug=False)
        buf.append(arr)
        tgts.append(vec)
        if len(buf) >= batch:
            flush()
    flush()
    return np.concatenate(probs, 0), np.array(tgts, dtype=np.float32)


def fit_thresholds(probs, targets):
    thr = np.full(60, 0.5, dtype=np.float32)
    for j in range(60):
        y = targets[:, j].astype(int)
        if y.min() == y.max():           # constant label → threshold doesn't matter
            continue
        best_f1, best_t = -1.0, 0.5
        for t in THRESH_GRID:
            f1 = f1_bin(y, (probs[:, j] >= t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thr[j] = best_t
    return thr


def fit_calibration(feats, total):
    """Non-negative least squares: [feats | 1] @ [w; b] ≈ total.

    `feats` is the per-label readout basis — soft probabilities OR hard
    (thresholded 0/1) labels. Hard-label fits are well-conditioned for losses
    (e.g. ASL) that skew the soft-probability distribution and make a soft-prob
    NNLS degenerate; soft fits are better for plain BCE. calibrate_model fits
    both and keeps whichever wins on val."""
    A = np.hstack([feats, np.ones((feats.shape[0], 1), np.float32)])
    coef, _ = nnls(A, total.astype(np.float64))
    return coef[:60].astype(np.float32), float(coef[60])


def score_metrics(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    err = pred - true
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    ss_res = float((err ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"r2": round(r2, 4), "rmse": round(rmse, 3), "mae": round(mae, 3)}


def macro_f1(probs, targets, thr):
    preds = (probs >= thr).astype(int)
    return float(np.mean([f1_bin(targets[:, j], preds[:, j])
                          for j in range(60)]))


def per_bin(pred, true):
    out = {}
    for lo, hi in [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 60)]:
        m = (true >= lo) & (true <= hi)
        if m.sum() == 0:
            continue
        out[f"{lo}-{hi}"] = {"n": int(m.sum()),
                             "mae": round(float(np.abs(pred[m] - true[m]).mean()), 2),
                             "mean_pred": round(float(pred[m].mean()), 1),
                             "mean_true": round(float(true[m].mean()), 1)}
    return out


def calibrate_model(model_path, write=False, verbose=True):
    """Fit per-label thresholds + NNLS score readout on the model's stored train ids,
    evaluate on val ids. If write=True, persist into the model metadata. Returns a dict
    with the comparison metrics (and applies thresholds/calibration in-place into metadata)."""
    def log(*a):
        if verbose:
            print(*a, flush=True)

    meta_path = model_path.replace(".pth", "_metadata.json")
    metadata = json.load(open(meta_path))
    log(f"Model: {os.path.basename(model_path)}")

    train_ids = metadata.get("train_image_ids") or []
    val_ids = metadata.get("val_image_ids") or []
    if not train_ids or not val_ids:
        raise ValueError("Metadata lacks train_image_ids / val_image_ids.")

    db = next(get_db())
    by_id = {r.id: r for r in db.query(TrainingDataImage)
             .filter(TrainingDataImage.id.in_(train_ids + val_ids)).all()}
    db.close()
    tr_rows = [by_id[i] for i in train_ids if i in by_id]
    va_rows = [by_id[i] for i in val_ids if i in by_id]
    log(f"Loaded {len(tr_rows)} train / {len(va_rows)} val images")

    model = load_model(model_path, metadata)
    log("Running inference (train)…")
    ptr, ytr = infer(model, tr_rows, metadata)
    log("Running inference (val)…")
    pva, yva = infer(model, va_rows, metadata)

    thr = fit_thresholds(ptr, ytr)
    # Fit the score readout two ways and keep whichever wins on val:
    #  - soft: NNLS on the probabilities (good for BCE)
    #  - hard: NNLS on the thresholded 0/1 labels (well-conditioned for ASL,
    #    whose skewed soft probs make a soft-prob NNLS degenerate)
    tot_tr = ytr.sum(1)
    w_soft, b_soft = fit_calibration(ptr, tot_tr)
    hard_tr = (ptr >= thr).astype(np.float32)
    w_hard, b_hard = fit_calibration(hard_tr, tot_tr)

    true_va = yva.sum(1)
    hard_va = (pva >= thr).astype(np.float32)
    cal_soft = pva @ w_soft + b_soft
    cal_hard = hard_va @ w_hard + b_hard
    if score_metrics(cal_hard, true_va)["mae"] <= score_metrics(cal_soft, true_va)["mae"]:
        readout, w, b = "hard", w_hard, b_hard
    else:
        readout, w, b = "soft", w_soft, b_soft

    derived = {
        "hard@0.5":   (pva >= 0.5).sum(1),
        "hard@thr":   hard_va.sum(1),
        "soft-sum":   pva.sum(1),
        "cal-soft":   cal_soft,
        "cal-hard":   cal_hard,
        "calibrated": (cal_hard if readout == "hard" else cal_soft),
    }
    result = {name: score_metrics(pred, true_va) for name, pred in derived.items()}
    result["macro_f1_0.5"] = round(macro_f1(pva, yva, 0.5), 4)
    result["macro_f1_thr"] = round(macro_f1(pva, yva, thr), 4)
    result["per_bin_calibrated"] = per_bin(derived["calibrated"], true_va)

    if verbose:
        print("\n=== Derived Total_Score on VAL (held-out) ===")
        for name, pred in derived.items():
            print(f"  {name:11s} {score_metrics(pred, true_va)}")
        print(f"\n=== Macro-F1 (60 sub-labels, VAL) ===")
        print(f"  fixed 0.5 : {result['macro_f1_0.5']:.4f}")
        print(f"  per-label : {result['macro_f1_thr']:.4f}")
        print("\n=== per-score-bin (calibrated) ===")
        for k, v in result["per_bin_calibrated"].items():
            print(f"  {k:6s} {v}")

    if write:
        metadata["thresholds"] = [round(float(t), 3) for t in thr]
        metadata["score_calibration"] = {
            "weights": [round(float(x), 5) for x in w],
            "bias": round(b, 5),
            "fit_on": "train_image_ids", "method": "nnls",
            "readout": readout,   # 'soft' → weights·probs ; 'hard' → weights·(probs>=thr)
        }
        metadata["calibration_metrics"] = result
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log(f"\nWrote thresholds + score_calibration into {os.path.basename(meta_path)}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="persist thresholds + calibration into metadata")
    ap.add_argument("--model", default=None, help="model .pth path (default: latest)")
    args = ap.parse_args()
    model_path = args.model or latest_model()[0]
    calibrate_model(model_path, write=args.write, verbose=True)
    if not args.write:
        print("\n(measure-only; re-run with --write to persist into metadata)")


if __name__ == "__main__":
    main()
