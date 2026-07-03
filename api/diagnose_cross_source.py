#!/usr/bin/env python3
"""Cross-source diagnosis of the DEPLOYED component model.

Runs the pinned components model on real images grouped by source_format, with
the model's own train_image_ids EXCLUDED (so every reported image is held-out).
Reports derived-score MAE/RMSE (vs the real Total_Score) and component macro-F1
where component ground truth exists. Shows how well the model generalizes to
sources it was NOT trained on (OXFORD/ALGORITHM/OCS_MACHINE/DRAWN) vs in-domain
held-out TELEFRED and the held-out low-score LOWSCORER set.

Single-view inference (no TTA) for speed — comparable to eval_lowscore.py.

Usage: docker exec -e PYTHONPATH=/app npsketch-api python3 /app/diagnose_cross_source.py [N_per_source]
"""
import sys, json, random
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, "/app")
from database import get_db, TrainingDataImage
from ai_training.component_calibration import load_model, f1_bin
from ai_training.dataset import components_to_vector
from ai_training.preprocessing import preprocess_bytes_for_prediction

MODELS = "/app/data/models"
DEPLOYED = "model_Components_20260625_055133"
SEED = 42
# component-training pool for this model (data_loader source filter + forced synth)
IN_DOMAIN = {"TELEFRED", "SYNTHETIC"}


def domain_label(src):
    if src == "TELEFRED":
        return "in-dist"
    if src == "LOWSCORER":
        return "lowscore"
    if src == "SYNTHETIC":
        return "synth"
    return "cross"          # OXFORD / ALGORITHM / OCS_MACHINE / DRAWN / …


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    meta = json.load(open(f"{MODELS}/{DEPLOYED}_metadata.json"))
    model = load_model(f"{MODELS}/{DEPLOYED}.pth", meta)
    sc = meta.get("score_calibration") or {}
    w = np.array(sc["weights"], np.float32) if sc.get("weights") else None
    b = float(sc.get("bias", 0.0))
    thr = np.array(meta.get("thresholds", [0.5] * 60), np.float32)
    train_ids = set(meta.get("train_image_ids", []))
    print(f"deployed: {DEPLOYED}")
    print(f"model_input: {meta.get('model_input')} | calibrated readout: {w is not None} | "
          f"train images excluded: {len(train_ids)}")
    print(f"single-view, up to {n_per}/source, held-out only\n")

    db = next(get_db())
    rng = random.Random(SEED)
    ids_by_src = defaultdict(list)
    for r in db.query(TrainingDataImage.id, TrainingDataImage.source_format,
                      TrainingDataImage.features_data).filter(
                      TrainingDataImage.features_data.isnot(None)).all():
        if r.id in train_ids:
            continue
        try:
            feat = json.loads(r.features_data)
        except Exception:
            continue
        if feat.get("Total_Score") is None:
            continue
        ids_by_src[r.source_format].append(r.id)

    hdr = f"{'source':12}{'domain':>9}{'n':>6}{'MAE':>8}{'RMSE':>8}{'compF1':>8}{'score_bias':>12}"
    print(hdr)
    print("-" * len(hdr))
    for src in sorted(ids_by_src, key=lambda s: (domain_label(s), s)):
        ids = ids_by_src[src][:]
        rng.shuffle(ids)
        ids = ids[:n_per]
        gt, derived, gtv, preds = [], [], [], []
        comps = True
        for i in ids:
            r = db.query(TrainingDataImage).filter_by(id=i).first()
            feat = json.loads(r.features_data)
            arr = preprocess_bytes_for_prediction(r.processed_image_data, metadata=meta, debug=False)
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float())).numpy()[0]
            derived.append(float(np.clip(p @ w + b, 0, 60)) if w is not None else float(p.sum()))
            gt.append(float(feat["Total_Score"]))
            if feat.get("components"):
                gtv.append(components_to_vector(feat))
                preds.append((p >= thr).astype(int))
            else:
                comps = False
        if not ids:
            continue
        gt = np.array(gt); derived = np.array(derived)
        mae = float(np.abs(derived - gt).mean())
        rmse = float(np.sqrt(((derived - gt) ** 2).mean()))
        bias = float((derived - gt).mean())
        if comps and gtv:
            gtv = np.array(gtv); preds = np.array(preds)
            f1 = f"{float(np.mean([f1_bin(gtv[:, j], preds[:, j]) for j in range(60)])):.3f}"
        else:
            f1 = "n/a"
        print(f"{src:12}{domain_label(src):>9}{len(ids):>6}{mae:>8.2f}{rmse:>8.2f}{f1:>8}{bias:>+12.2f}")
    db.close()
    print("\ndomain: in-dist=TELEFRED held-out · cross=source unseen in training · "
          "lowscore=held-out LOWSCORER · score_bias=mean(pred-real)")


if __name__ == "__main__":
    main()
