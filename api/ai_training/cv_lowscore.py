#!/usr/bin/env python3
"""Patient-level 3-fold CV for honest low-score measurement of the component model.

Every real TELEFRED patient is held out in exactly one fold, so each real image gets an
out-of-fold (OOF) prediction → per-score-bin MAE/F1 with FULL coverage (n~179 ≤29 instead
of the single split's 35). Synthetic rows (SYNTH_*) are forced into train every fold (via
the data_loader val_patient_ids override + split_strategy prefix), so this measures the
deployed recipe (real + synthetic). ~3×10h on CPU; resumable via the manifest.

  --smoke   1 fold, 1 epoch, 400 images — validate the fold override end-to-end (~min)
  (default) run all 3 folds (skips folds already in the manifest), then aggregate OOF
"""
import sys, os, json, glob, argparse
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, "/app")
from database import SessionLocal, TrainingDataImage
from routers.ai_training_base import run_training_job
from ai_training.component_calibration import load_model
from ai_training.dataset import components_to_vector
from ai_training.preprocessing import preprocess_bytes_for_prediction

OUT = "/app/data/cv_lowscore"; os.makedirs(OUT, exist_ok=True)
MAN = f"{OUT}/manifest.json"
NFOLD = 3
BINS = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 60)]


def real_rows():
    db = SessionLocal()
    rows = db.query(TrainingDataImage).filter(
        TrainingDataImage.source_format == "TELEFRED",
        TrainingDataImage.features_data.isnot(None)).all()
    out = [r for r in rows if json.loads(r.features_data).get("components")]
    db.close()
    return out


def make_folds(rows):
    pat = defaultdict(list)
    for r in rows:
        pat[r.patient_id].append(json.loads(r.features_data)["Total_Score"])
    med = {p: float(np.median(s)) for p, s in pat.items()}
    order = sorted(med, key=lambda p: (med[p], p))      # stratify by median score, deterministic
    folds = [[] for _ in range(NFOLD)]
    for i, p in enumerate(order):
        folds[i % NFOLD].append(p)
    return folds


def images_data(rows):
    return [{"id": r.id, "patient_id": r.patient_id,
             "processed_image_data": r.processed_image_data, "features_data": r.features_data} for r in rows]


def train_fold(rows, val_patients, smoke=False):
    db = SessionLocal()
    cfg = {"target_feature": "Components", "train_split": 0.8,
           "num_epochs": 1 if smoke else 15, "batch_size": 8,
           "use_augmentation": True, "use_normalization": False,
           "add_synthetic_bad_images": False, "synthetic_n_samples": 0,
           "max_images": 400 if smoke else None,
           "images_data": images_data(rows), "db_session": db,
           "val_patient_ids": set(val_patients)}
    run_training_job(cfg)
    db.close()
    return max(glob.glob("/app/data/models/model_Components_*.pth"), key=os.path.getmtime)


def oof_aggregate(rows, manifest):
    by_pat = defaultdict(list)
    for r in rows:
        by_pat[r.patient_id].append(r)
    gt, pred, P, GV = [], [], [], []
    for k in range(NFOLD):
        mp = manifest[str(k)]["model"]; meta = json.load(open(mp.replace(".pth", "_metadata.json")))
        model = load_model(mp, meta)
        sc = meta["score_calibration"]; w = np.array(sc["weights"]); b = sc["bias"]
        thr = np.array(meta["thresholds"], np.float32)
        for p in manifest[str(k)]["val_patients"]:
            for r in by_pat.get(p, []):
                v = components_to_vector(json.loads(r.features_data))
                if v is None:
                    continue
                arr = preprocess_bytes_for_prediction(r.processed_image_data, metadata=meta, debug=False)
                with torch.no_grad():
                    probs = torch.sigmoid(model(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float())).numpy()[0]
                gt.append(int(sum(v))); pred.append(float(np.clip(probs @ w + b, 0, 60)))
                P.append(probs); GV.append(np.array(v))
    gt = np.array(gt); pred = np.array(pred); P = np.array(P); GV = np.array(GV)
    from ai_training.component_calibration import f1_bin
    thr = np.array(json.load(open(manifest["0"]["model"].replace(".pth", "_metadata.json")))["thresholds"])
    print(f"\n=== OOF 3-fold (n={len(gt)} real images, full coverage) ===")
    print(f"{'bin':8s}{'n':>5}{'MAE':>8}{'macroF1':>9}")
    for lo, hi in BINS:
        m = (gt >= lo) & (gt <= hi)
        if not m.sum():
            continue
        f1 = float(np.mean([f1_bin(GV[m][:, j].astype(int), (P[m][:, j] >= thr[j]).astype(int)) for j in range(60)]))
        print(f"{f'{lo}-{hi}':8s}{int(m.sum()):>5}{np.abs(pred[m]-gt[m]).mean():>8.2f}{f1:>9.3f}")
    for lab, m in [("0-29", gt <= 29), ("ALL", gt >= 0)]:
        print(f"{lab:8s}{int(m.sum()):>5}{np.abs(pred[m]-gt[m]).mean():>8.2f}")
    json.dump({"n": len(gt), "oof_mae_0_29": float(np.abs(pred[gt <= 29] - gt[gt <= 29]).mean()),
               "oof_mae_all": float(np.abs(pred - gt).mean())}, open(f"{OUT}/oof_result.json", "w"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    rows = real_rows(); folds = make_folds(rows)
    print(f"{len(rows)} real images, {sum(len(f) for f in folds)} patients, fold sizes {[len(f) for f in folds]}", flush=True)
    if args.smoke:
        mp = train_fold(rows, folds[0][:60], smoke=True)
        print(f"SMOKE OK -> {mp}", flush=True); return
    manifest = json.load(open(MAN)) if os.path.exists(MAN) else {}
    for k in range(NFOLD):
        if str(k) in manifest:
            print(f"fold {k} already trained: {manifest[str(k)]['model']}", flush=True); continue
        print(f"=== training fold {k} (val {len(folds[k])} patients) ===", flush=True)
        mp = train_fold(rows, folds[k])
        manifest[str(k)] = {"model": mp, "val_patients": folds[k]}
        json.dump(manifest, open(MAN, "w"))
        print(f"fold {k} -> {mp}", flush=True)
    oof_aggregate(rows, manifest)


if __name__ == "__main__":
    main()
