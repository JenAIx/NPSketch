#!/usr/bin/env python3
"""
Quick grid search for Custom_Class_4 with short runs.
Runs a small set of configs for 5 epochs each and ranks by val macro F1.
"""
import sys
import json
import time
import random
from pathlib import Path

sys.path.insert(0, "/app")

from database import SessionLocal, TrainingDataImage
from routers.ai_training_base import run_training_job


MODEL_DIR = Path("/app/data/models")


def find_new_metadata(after_ts: float):
    candidates = sorted(
        MODEL_DIR.glob("model_Custom_Class_4_*_metadata.json"),
        key=lambda p: p.stat().st_mtime,
    )
    for path in reversed(candidates):
        if path.stat().st_mtime > after_ts:
            return path
    return None


def load_custom_class_images(limit: int = 500, seed: int = 42):
    db = SessionLocal()
    try:
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        valid_images = []
        for img in images:
            try:
                features = json.loads(img.features_data)
                if "Custom_Class" in features and "4" in features["Custom_Class"]:
                    valid_images.append(img)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        random.Random(seed).shuffle(valid_images)
        valid_images = valid_images[:limit]

        images_data = [
            {
                "id": img.id,
                "processed_image_data": img.processed_image_data,
                "features_data": img.features_data,
            }
            for img in valid_images
        ]
        return images_data
    finally:
        db.close()


def run_trial(config_name: str, images_data, overrides: dict):
    start_ts = time.time()

    config = {
        "target_feature": "Custom_Class_4",
        "train_split": 0.8,
        "num_epochs": 5,
        "learning_rate": None,
        "batch_size": 8,
        "use_augmentation": False,
        "use_normalization": False,
        "add_synthetic_bad_images": False,
        "synthetic_n_samples": 0,
        "use_lr_scheduling": True,
        "lr_scheduling_patience": 2,
        "lr_scheduling_threshold": 0.01,
        "use_differential_lr": True,
        "backbone_lr_multiplier": 0.1,
        "dropout": None,
        "weight_decay": None,
        "label_smoothing": None,
        "warmup_enabled": True,
        "warmup_epochs": 1,
        "early_stopping_patience": 8,
        "early_stopping_min_delta": 0.01,
        "early_stopping_min_epochs": 2,
        "images_data": images_data,
        "db_session": SessionLocal(),
    }

    config.update(overrides)

    try:
        run_training_job(config)
    finally:
        config["db_session"].close()

    # Find the metadata created after this run started
    metadata_path = find_new_metadata(start_ts)
    if not metadata_path:
        return None

    data = json.loads(metadata_path.read_text())
    val_metrics = data.get("val_metrics", {})
    result = {
        "name": config_name,
        "metadata": str(metadata_path),
        "val_accuracy": val_metrics.get("accuracy", 0.0),
        "val_macro_f1": val_metrics.get("macro_f1", 0.0),
        "best_epoch": data.get("early_stopping", {}).get("best_epoch"),
        "best_val_loss": data.get("early_stopping", {}).get("best_val_loss"),
        "config": data.get("training_config", {}),
    }
    return result


def main():
    images_data = load_custom_class_images(limit=500, seed=42)
    if not images_data:
        print("No Custom_Class_4 images found.")
        return 1

    trials = [
        ("baseline_new", {"learning_rate": 0.0005, "dropout": 0.6, "weight_decay": 0.0005, "label_smoothing": 0.1}),
        ("less_reg", {"learning_rate": 0.0005, "dropout": 0.5, "weight_decay": 0.0001, "label_smoothing": 0.0}),
        ("mid_reg", {"learning_rate": 0.0005, "dropout": 0.5, "weight_decay": 0.0005, "label_smoothing": 0.05}),
        ("higher_lr", {"learning_rate": 0.001, "dropout": 0.5, "weight_decay": 0.0001, "label_smoothing": 0.0}),
        ("higher_lr_mid", {"learning_rate": 0.001, "dropout": 0.5, "weight_decay": 0.0005, "label_smoothing": 0.05}),
        ("low_wd", {"learning_rate": 0.0005, "dropout": 0.6, "weight_decay": 0.0001, "label_smoothing": 0.05}),
    ]

    results = []
    for name, overrides in trials:
        print(f"\n=== Running trial: {name} ===")
        print(f"Overrides: {overrides}")
        result = run_trial(name, images_data, overrides)
        if result is None:
            print("No metadata produced.")
            continue
        print(
            f"Result: val_acc={result['val_accuracy']:.4f}, "
            f"val_macro_f1={result['val_macro_f1']:.4f}, "
            f"best_epoch={result['best_epoch']}, best_val_loss={result['best_val_loss']:.4f}"
        )
        results.append(result)

    results.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    print("\n=== Summary (sorted by val_macro_f1) ===")
    for r in results:
        print(
            f"{r['name']}: val_macro_f1={r['val_macro_f1']:.4f}, "
            f"val_acc={r['val_accuracy']:.4f}, metadata={r['metadata']}"
        )

    if results:
        best = results[0]
        print("\n=== Best Trial ===")
        print(f"Name: {best['name']}")
        print(f"Val Macro F1: {best['val_macro_f1']:.4f}")
        print(f"Val Acc: {best['val_accuracy']:.4f}")
        print(f"Metadata: {best['metadata']}")
        print(f"Config: {best['config']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
