#!/usr/bin/env python3
"""
ABLATION: component-head training WITHOUT the synthetic images.

Same as start_telefred_component_training.py, but passes exclude_sources=['SYNTHETIC']
so the SYNTHETIC rows are dropped from the training/val set *without deleting them from
the DB*. Trains on TELEFRED + LOWSCORER + any other human-validated row (OXFORD/DRAWN).
Purpose: measure whether synthetic still earns its place now that LOWSCORER provides
real low-score data (and whether it drives the 20-29 band regression).

Detached run inside the container:

    docker exec -d npsketch-api bash -c \
      'cd /app && PYTHONPATH=/app nohup python3 start_component_training_nosynth.py \
       > /app/data/logs/component_train_nosynth.log 2>&1'
"""
import sys
import time

sys.path.insert(0, "/app")

from database import SessionLocal, TrainingDataImage
from routers.ai_training_base import run_training_job


def main():
    print("=" * 60)
    print("COMPONENT-HEAD Training — NO SYNTHETIC (ablation)")
    print("=" * 60)

    db = SessionLocal()
    try:
        rows = (
            db.query(TrainingDataImage)
            .filter(TrainingDataImage.source_format == "TELEFRED",
                    TrainingDataImage.features_data.isnot(None))
            .all()
        )
        images_data = [
            {
                "id": img.id,
                "patient_id": img.patient_id,
                "processed_image_data": img.processed_image_data,
                "features_data": img.features_data,
            }
            for img in rows
        ]
        print(f"\nTELEFRED rows available (metadata path): {len(images_data)}")

        config = {
            "target_feature": "Components",
            "train_split": 0.8,
            "num_epochs": 15,
            "learning_rate": None,
            "batch_size": 8,
            "use_augmentation": True,
            "use_normalization": False,
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
            "exclude_sources": ["SYNTHETIC"],   # <-- the ablation: keep rows in DB, drop from training
        }

        print("\nConfiguration:")
        print("  Target:        Components (60 sub-labels, BCE)")
        print("  Source:        TELEFRED + any validated row (LOWSCORER/OXFORD/DRAWN) — SYNTHETIC EXCLUDED")
        print("  Epochs:        15 (+ early stopping)")
        print("\n" + "=" * 60)
        print("Starting training (hours on CPU)...")
        print("=" * 60 + "\n", flush=True)

        start = time.time()
        run_training_job(config)
        print("\n" + "=" * 60)
        print(f"DONE. Training finished in {(time.time()-start)/60:.1f} min")
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
    sys.exit(0 if main() else 1)
