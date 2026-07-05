#!/usr/bin/env python3
"""
Component-head RETRAIN: ASL loss + 400x193 input, on the LOWSCORER-enlarged set.

Same pipeline as start_telefred_component_training.py, but:
  - reads the new config (training.components.loss=asl, model_input 400x193,
    consistency 0) automatically from training_config.yaml;
  - FIXES the validation fold to the DEPLOYED model's val patients
    (model_Components_20260625_055133) so both models are evaluated on the
    identical held-out real images → fair A/B via eval_lowscore.py.

Detached run inside the container:

    docker exec -d npsketch-api bash -c \
      'cd /app && PYTHONPATH=/app nohup python3 start_component_retrain_asl400.py \
       > /app/data/logs/retrain_asl400.log 2>&1'
"""
import sys
import time
import json

sys.path.insert(0, "/app")

from database import SessionLocal, TrainingDataImage
from routers.ai_training_base import run_training_job

DEPLOYED_META = "/app/data/models/model_Components_20260625_055133_metadata.json"


def main():
    print("=" * 64)
    print("COMPONENT RETRAIN — ASL + 400x193 (fixed val = deployed model's fold)")
    print("=" * 64)

    db = SessionLocal()
    try:
        # Fixed validation fold: the deployed model's held-out patients, so the
        # new model and the deployed model share the exact held-out real images.
        dep = json.load(open(DEPLOYED_META))
        val_ids = dep["val_image_ids"]
        val_patient_ids = set()
        for i in val_ids:
            r = db.query(TrainingDataImage.patient_id).filter(TrainingDataImage.id == i).first()
            if r and r.patient_id:
                val_patient_ids.add(r.patient_id)
        print(f"Fixed val fold: {len(val_patient_ids)} patients "
              f"(from {len(val_ids)} deployed val images)")

        # images_data is used for metadata / non-augmented path; the augmented
        # component path re-queries the DB (TELEFRED+SYNTHETIC OR validated).
        rows = (db.query(TrainingDataImage)
                .filter(TrainingDataImage.features_data.isnot(None))
                .all())
        images_data = [{"id": r.id, "patient_id": r.patient_id,
                        "processed_image_data": r.processed_image_data,
                        "features_data": r.features_data} for r in rows]

        config = {
            "target_feature": "Components",
            "train_split": 0.8,                 # ignored when val_patient_ids is set
            "num_epochs": 20,                   # + early stopping (components patience 4)
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
            "val_patient_ids": val_patient_ids,
        }

        print("\nConfiguration:")
        print("  Target:        Components (60 sub-labels)")
        print("  Loss:          ASL (from YAML)   Input: 400x193 (from YAML)")
        print("  Source:        TELEFRED + SYNTHETIC + validated (incl. 712 LOWSCORER)")
        print("  Val fold:      FIXED to deployed model's patients (fair A/B)")
        print("  Epochs:        20 (+ early stopping)")
        print("\n" + "=" * 64)
        print("Starting training (hours on CPU — expect ~24-28h at 400x193)...")
        print("=" * 64 + "\n", flush=True)

        start = time.time()
        run_training_job(config)
        elapsed = time.time() - start
        print("\n" + "=" * 64)
        print(f"DONE in {elapsed/3600:.2f} h", flush=True)
        print("=" * 64, flush=True)
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
