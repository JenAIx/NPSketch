#!/usr/bin/env python3
"""Precompute the best model's prediction for each TELEFRED image and store it in the
`model_prediction` column (in parallel with the human `features_data`). Powers the
review/labeling tool: real-vs-model score + a model component suggestion.

Stored JSON per image: {model, Total_Score (calibrated), score_hard, components:{presence,
accuracy, position}} — components are the model's hard@threshold per-element decisions.

Run:  docker exec -e PYTHONPATH=/app npsketch-api python3 /app/ai_training/precompute_predictions.py
"""
import sys, json, glob, os
import numpy as np
import torch

sys.path.insert(0, "/app")
from database import SessionLocal, TrainingDataImage
from ai_training.component_calibration import load_model
from ai_training.preprocessing import preprocess_bytes_for_prediction


def main():
    # prefer the pinned "current" components model; else newest on disk
    cur = None
    try:
        cm = json.load(open("/app/data/models/current_models.json")).get("components")
        fn = cm.get("filename") if isinstance(cm, dict) else cm
        if fn and os.path.exists(f"/app/data/models/{fn}"):
            cur = f"/app/data/models/{fn}"
    except Exception:
        pass
    mp = cur or sorted(glob.glob("/app/data/models/model_Components_*.pth"))[-1]
    meta = json.load(open(mp.replace(".pth", "_metadata.json")))
    model = load_model(mp, meta)
    name = os.path.basename(mp)
    thr = np.array(meta.get("thresholds", [0.5] * 60), np.float32)
    sc = meta.get("score_calibration") or {}
    w = np.array(sc.get("weights", [0] * 60)); b = float(sc.get("bias", 0))
    print(f"model: {name}", flush=True)

    db = SessionLocal()
    rows = db.query(TrainingDataImage).filter(TrainingDataImage.source_format == "TELEFRED").all()
    n = 0
    for r in rows:
        if not r.processed_image_data:
            continue
        arr = preprocess_bytes_for_prediction(r.processed_image_data, metadata=meta, debug=False)
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float())).numpy()[0]
        hard = (probs >= thr).astype(int)
        cal = float(np.clip(probs @ w + b, 0, 60)) if sc.get("weights") else float(hard.sum())
        r.model_prediction = json.dumps({
            "model": name,
            "Total_Score": round(cal, 1),
            "score_hard": int(hard.sum()),
            "components": {"presence": hard[0::3].tolist(),
                           "accuracy": hard[1::3].tolist(),
                           "position": hard[2::3].tolist()},
        })
        n += 1
        if n % 500 == 0:
            db.commit(); print(f"  {n} predictions...", flush=True)
    db.commit()
    total = db.query(TrainingDataImage).filter(TrainingDataImage.model_prediction.isnot(None)).count()
    db.close()
    print(f"stored {n} predictions; rows with model_prediction now {total}", flush=True)


if __name__ == "__main__":
    main()
