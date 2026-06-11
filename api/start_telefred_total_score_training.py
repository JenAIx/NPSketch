#!/usr/bin/env python3
"""
Full Total_Score regression training run on the expanded dataset
(OXFORD + ALGORITHM + OCS + MAT + TELEFRED 202606 union).

Regression target: Total_Score (continuous 0-60).
Augmentation: ON (~6x). Epochs: 30 with early stopping.

Designed to run detached overnight inside the npsketch-api container:

    docker exec -d npsketch-api bash -c \
      'cd /app && PYTHONPATH=/app nohup python3 start_telefred_total_score_training.py \
       > /app/data/logs/telefred_total_score_train.log 2>&1'
"""
import sys
import json
import time

sys.path.insert(0, "/app")

from database import SessionLocal, TrainingDataImage
from routers.ai_training_base import run_training_job


def main():
    print("=" * 60)
    print("FULL Total_Score Regression Training (expanded dataset)")
    print("=" * 60)

    db = SessionLocal()
    try:
        rows = (
            db.query(TrainingDataImage)
            .filter(TrainingDataImage.features_data.isnot(None))
            .all()
        )

        images_data = []
        by_source = {}
        for img in rows:
            try:
                features = json.loads(img.features_data)
            except (json.JSONDecodeError, TypeError):
                continue
            ts = features.get("Total_Score")
            if not isinstance(ts, (int, float)):
                continue
            images_data.append(
                {
                    "id": img.id,
                    "patient_id": img.patient_id,
                    "processed_image_data": img.processed_image_data,
                    "features_data": img.features_data,
                }
            )
            by_source[img.source_format] = by_source.get(img.source_format, 0) + 1

        print(f"\nImages with Total_Score: {len(images_data)}")
        print(f"By source_format: {by_source}")

        if len(images_data) < 50:
            print("Not enough labeled images for training.")
            return False

        config = {
            "target_feature": "Total_Score",   # regression (does not start with Custom_Class_)
            "train_split": 0.8,
            "num_epochs": 15,                  # CPU run; early stopping may end earlier
            "learning_rate": None,             # use regression default from training_config.yaml
            "batch_size": 8,
            "use_augmentation": True,          # full run: augmentation ON
            "use_normalization": True,         # regression uses min-max target normalization
            "add_synthetic_bad_images": False,
            "synthetic_n_samples": 0,
            "use_lr_scheduling": None,         # default True
            "use_differential_lr": None,       # default True
            "backbone_lr_multiplier": None,    # default 0.1
            "dropout": None,                   # regression default from YAML
            "weight_decay": None,              # regression default from YAML
            "early_stopping_patience": None,   # regression default from YAML
            "early_stopping_min_delta": None,  # regression default from YAML
            "images_data": images_data,
            "db_session": db,
        }

        print("\nConfiguration:")
        print("  Target:        Total_Score (regression)")
        print("  Epochs:        15 (+ early stopping)")
        print("  Augmentation:  ON (~6x)")
        print("  Normalization: ON (min-max target)")
        print("  Batch size:    8")
        print("  Using regression task defaults from training_config.yaml")
        print("\n" + "=" * 60)
        print("Starting training (this will take hours on CPU)...")
        print("=" * 60 + "\n", flush=True)

        start = time.time()
        run_training_job(config)
        elapsed = time.time() - start

        print("\n" + "=" * 60)
        print(f"DONE. Training finished in {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
        print("=" * 60, flush=True)
        return True

    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
