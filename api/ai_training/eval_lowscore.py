#!/usr/bin/env python3
"""Compare component models on the shared real held-out val, per score-bin.

All component models share the same seed-42 patient split, so the baseline
(model_Components_20260613_005214) val_image_ids identify the held-out REAL images for
every model. We evaluate each given model on those same stored images (its own
calibration) and report per-bin derived-score MAE + macro-F1. This is the standard,
fair comparison for the synthetic-data experiments (see MASTER_PLAN_SYN_TRAIN.md).

Note: evaluate on REAL val only — a synthetic test set is biased toward synthetic-trained
models. Low bins have small n (0-9: ~3) so read directional, corroborated by F1.

Usage:
  python3 /app/ai_training/eval_lowscore.py 20260613_005214 20260617_023227 [<stamp> ...]
  (args are the model timestamp stamps, or full model filenames; default compares the
   baseline against all model_Components_* on disk.)
"""
import sys, glob, json, os
import numpy as np
import torch

sys.path.insert(0, "/app")
from database import get_db, TrainingDataImage
from ai_training.component_calibration import load_model, f1_bin
from ai_training.dataset import components_to_vector
from ai_training.preprocessing import preprocess_bytes_for_prediction

MODELS = "/app/data/models"
BASELINE_STAMP = "20260625_055133"   # deployed model — provides the shared val_image_ids
BINS = [(0, 9), (10, 19), (20, 29), (0, 19), (0, 29), (0, 60)]


def resolve(arg):
    """Accept a stamp (20260617_023227), a .pth name, or a full path."""
    if arg.endswith(".pth"):
        return arg if os.path.isabs(arg) else f"{MODELS}/{arg}"
    return f"{MODELS}/model_Components_{arg}.pth"


def infer(model, meta, rows):
    probs = []
    for r in rows:
        arr = preprocess_bytes_for_prediction(r.processed_image_data, metadata=meta, debug=False)
        with torch.no_grad():
            probs.append(torch.sigmoid(model(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float())).numpy()[0])
    return np.array(probs)


def main():
    stamps = sys.argv[1:]
    if not stamps:
        stamps = [os.path.basename(p)[16:-4] for p in sorted(glob.glob(f"{MODELS}/model_Components_*.pth"))]

    base_meta = json.load(open(resolve(BASELINE_STAMP).replace(".pth", "_metadata.json")))
    val_ids = base_meta["val_image_ids"]
    db = next(get_db())
    rows = [db.query(TrainingDataImage).filter_by(id=i).first() for i in val_ids]
    db.close()
    rows = [r for r in rows if r and r.features_data]
    gtv = np.array([components_to_vector(json.loads(r.features_data)) for r in rows])
    gt = gtv.sum(1).astype(int)
    print(f"shared real val: {len(rows)} images\n")

    results = {}
    for s in stamps:
        mp = resolve(s)
        meta = json.load(open(mp.replace(".pth", "_metadata.json")))
        model = load_model(mp, meta)
        probs = infer(model, meta, rows)
        sc = meta.get("score_calibration") or {}
        score = np.clip(probs @ np.array(sc["weights"]) + sc["bias"], 0, 60) if sc.get("weights") else probs.sum(1)
        thr = np.array(meta.get("thresholds", [0.5] * 60), np.float32)
        results[s] = (score, probs, thr)

    # table per bin
    hdr = "bin".ljust(8) + "n".rjust(5) + "".join(f"  {s[-6:]:>13}" for s in stamps)
    print(hdr); print("-" * len(hdr))
    for lo, hi in BINS:
        m = (gt >= lo) & (gt <= hi)
        if m.sum() == 0:
            continue
        cells = []
        for s in stamps:
            score, probs, thr = results[s]
            mae = float(np.abs(score[m] - gt[m]).mean())
            pred = (probs[m] >= thr).astype(int); tgt = gtv[m].astype(int)
            f1 = float(np.mean([f1_bin(tgt[:, j], pred[:, j]) for j in range(60)]))
            cells.append(f"{mae:5.2f}/{f1:.3f}")
        print(f"{f'{lo}-{hi}':8}{int(m.sum()):>5}" + "".join(f"  {c:>13}" for c in cells))
    print("\n(cells = MAE / macro-F1; lower MAE + higher F1 = better)")


if __name__ == "__main__":
    main()
