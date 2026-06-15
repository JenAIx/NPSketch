#!/usr/bin/env python3
"""
Component-head training on TELEFRED: predict the 60 OCS-Plus sub-labels
(20 elements x PRES/ACC/POS), Total_Score = their sum.

Rationale: the holistic regressor fails in the sparse low-score range because
each low-score drawing is a near-unique constellation. Decomposing into the 60
binary sub-labels gives dense supervision everywhere. Trained on TELEFRED only
(human-rated, internally consistent); ALGORITHM/OCS_MACHINE use differently
calibrated machine labels and are excluded for v1.

Detached overnight run inside the npsketch-api container:

    docker exec -d npsketch-api bash -c \
      'cd /app && PYTHONPATH=/app nohup python3 start_telefred_component_training.py \
       > /app/data/logs/telefred_component_train.log 2>&1'
"""
import sys
import time

sys.path.insert(0, "/app")

from database import SessionLocal, TrainingDataImage
from routers.ai_training_base import run_training_job


def main():
    print("=" * 60)
    print("COMPONENT-HEAD Training on TELEFRED (60 sub-labels)")
    print("=" * 60)

    db = SessionLocal()
    try:
        # images_data is used for metadata / the non-augmented path; the augmented
        # path reloads TELEFRED-with-components from the DB itself (source_filter).
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
        print(f"\nTELEFRED rows available: {len(images_data)}")

        config = {
            "target_feature": "Components",     # 60-sub-label component head
            "train_split": 0.8,
            "num_epochs": 15,
            "learning_rate": None,              # components default from YAML
            "batch_size": 8,
            "use_augmentation": True,           # required for component mode
            "use_normalization": False,         # BCE on 0/1, no target normalization
            "add_synthetic_bad_images": False,
            "synthetic_n_samples": 0,
            "use_lr_scheduling": None,
            "use_differential_lr": None,
            "backbone_lr_multiplier": None,
            "dropout": None,
            "weight_decay": None,
            "early_stopping_patience": None,    # components default (4) from YAML
            "early_stopping_min_delta": None,
            "images_data": images_data,
            "db_session": db,
        }

        print("\nConfiguration:")
        print("  Target:        Components (60 sub-labels, BCE)")
        print("  Source:        TELEFRED only")
        print("  Epochs:        15 (+ early stopping)")
        print("  Augmentation:  ON (~6x)")
        print("  Input:         284x137 (downscaled)")
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
