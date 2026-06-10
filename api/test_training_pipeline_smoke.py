#!/usr/bin/env python3
"""
Smoke test for the training pipeline fixes (2026-06):
  - patient-level (group) train/val split, zero patient overlap
  - linear regression head (use_sigmoid=False) + post-denorm clamping
  - WeightedRandomSampler against score imbalance
  - per-score-bin metrics in train/val metrics
  - torch seeding

Runs two short trainings on small subsets (non-augmented and augmented path)
and asserts the invariants on the produced model metadata.

Run inside the container:
    docker exec -e PYTHONPATH=/app npsketch-api python3 /app/test_training_pipeline_smoke.py

NOTE: creates real model files (model_Total_Score_*.pth) - they are deleted
at the end of the test.
"""
import sys
import json
import glob
import os
import random

sys.path.insert(0, "/app")

from database import SessionLocal, TrainingDataImage
from routers import ai_training_base
from routers.ai_training_base import run_training_job


def load_sample(db, n):
    rows = (
        db.query(TrainingDataImage)
        .filter(TrainingDataImage.features_data.isnot(None))
        .all()
    )
    labeled = []
    for img in rows:
        try:
            features = json.loads(img.features_data)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(features.get("Total_Score"), (int, float)):
            labeled.append(img)
    random.seed(42)
    sample = random.sample(labeled, min(n, len(labeled)))
    return [
        {
            "id": img.id,
            "patient_id": img.patient_id,
            "processed_image_data": img.processed_image_data,
            "features_data": img.features_data,
        }
        for img in sample
    ]


def patient_overlap(db, train_ids, val_ids):
    """Return patients appearing in both sets (must be empty)."""
    def patients(ids):
        if not ids:
            return set()
        rows = (
            db.query(TrainingDataImage.patient_id)
            .filter(TrainingDataImage.id.in_(ids))
            .all()
        )
        return {r[0] for r in rows if r[0]}
    return patients(train_ids) & patients(val_ids)


def check_metadata(db, label):
    files = sorted(
        glob.glob("/app/data/models/model_Total_Score_*_metadata.json"),
        key=os.path.getmtime,
    )
    assert files, "no model metadata found"
    meta_file = files[-1]
    with open(meta_file) as f:
        meta = json.load(f)

    errors = []

    if meta.get("use_sigmoid") is not False:
        errors.append(f"use_sigmoid expected False, got {meta.get('use_sigmoid')}")
    if meta.get("model", {}).get("use_sigmoid") is not False:
        errors.append("model.use_sigmoid expected False")

    sq = meta.get("split_quality", {})
    if sq.get("group_overlap") != 0:
        errors.append(f"split_quality.group_overlap expected 0, got {sq.get('group_overlap')}")
    if "group" not in str(sq.get("method", "")):
        errors.append(f"split method not group-based: {sq.get('method')}")

    overlap = patient_overlap(db, meta.get("train_image_ids", []), meta.get("val_image_ids", []))
    if overlap:
        errors.append(f"PATIENT LEAKAGE: {len(overlap)} patients in both sets, e.g. {sorted(overlap)[:5]}")

    if not meta.get("imbalance_sampler", {}).get("enabled"):
        errors.append("imbalance_sampler not enabled in metadata")

    for key in ("train_metrics", "val_metrics"):
        if "per_score_bin" not in meta.get(key, {}):
            errors.append(f"{key} missing per_score_bin")

    vm = meta.get("val_metrics", {})
    print(f"\n[{label}] model: {os.path.basename(meta_file)}")
    print(f"[{label}] split: {sq.get('method')} | "
          f"groups train/val: {sq.get('n_groups_train')}/{sq.get('n_groups_val')} | "
          f"patient overlap: {len(overlap)}")
    print(f"[{label}] sampler bins: {meta.get('imbalance_sampler', {}).get('bin_counts')}")
    print(f"[{label}] val R2={vm.get('r2_score'):.3f} RMSE={vm.get('rmse'):.2f} MAE={vm.get('mae'):.2f} "
          f"(short smoke run - quality not meaningful)")

    if errors:
        for e in errors:
            print(f"[{label}] FAIL: {e}")
        return meta_file, False
    print(f"[{label}] all checks PASSED")
    return meta_file, True


def base_config(images_data, db, use_augmentation, num_epochs):
    return {
        "target_feature": "Total_Score",
        "train_split": 0.8,
        "num_epochs": num_epochs,
        "learning_rate": None,
        "batch_size": 8,
        "use_augmentation": use_augmentation,
        "use_normalization": True,
        "add_synthetic_bad_images": False,
        "synthetic_n_samples": 0,
        "use_lr_scheduling": None,
        "use_differential_lr": None,
        "backbone_lr_multiplier": None,
        "dropout": None,
        "weight_decay": None,
        "early_stopping_patience": None,
        "early_stopping_min_delta": None,
        "images_data": images_data,
        "db_session": db,
    }


def main():
    db = SessionLocal()
    created_files = []
    ok = True
    try:
        # Phase 1: non-augmented path (create_dataloaders group split)
        print("=" * 60)
        print("SMOKE 1/2: non-augmented path (200 images, 1 epoch)")
        print("=" * 60)
        run_training_job(base_config(load_sample(db, 200), db, use_augmentation=False, num_epochs=1))
        if ai_training_base.training_state.get("status") == "error":
            print(f"FAIL: training errored: {ai_training_base.training_state.get('error')}")
            ok = False
        else:
            meta_file, passed = check_metadata(db, "non-aug")
            created_files.append(meta_file)
            ok = ok and passed

        # Phase 2: augmented path (data_loader group split + disk sampler).
        # prepare_augmented_training_data loads from the DB itself; max_images
        # caps it to a small random subset for the smoke run.
        print("\n" + "=" * 60)
        print("SMOKE 2/2: augmented path (300 images, 1 epoch)")
        print("=" * 60)
        cfg2 = base_config(load_sample(db, 300), db, use_augmentation=True, num_epochs=1)
        cfg2["max_images"] = 300
        run_training_job(cfg2)
        if ai_training_base.training_state.get("status") == "error":
            print(f"FAIL: training errored: {ai_training_base.training_state.get('error')}")
            ok = False
        else:
            meta_file, passed = check_metadata(db, "augmented")
            created_files.append(meta_file)
            ok = ok and passed
    finally:
        db.close()
        # Clean up smoke models so they don't pollute the model list / pollers
        for meta_file in created_files:
            model_file = meta_file.replace("_metadata.json", ".pth")
            for f in (meta_file, model_file):
                try:
                    os.remove(f)
                    print(f"cleaned up {os.path.basename(f)}")
                except OSError:
                    pass

    print("\n" + "=" * 60)
    print("SMOKE RESULT:", "PASSED" if ok else "FAILED")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
