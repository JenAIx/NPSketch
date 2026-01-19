#!/usr/bin/env python3
"""
Start a faster Custom_Class_4 training run (reduced data/augmentation).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, "/app")

from database import SessionLocal, TrainingDataImage
from routers.ai_training_base import run_training_job


def start_training():
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

        # Limit to speed up the run
        valid_images = valid_images[:500]

        images_data = [
            {
                "id": img.id,
                "processed_image_data": img.processed_image_data,
                "features_data": img.features_data,
            }
            for img in valid_images
        ]

        config = {
            "target_feature": "Custom_Class_4",
            "train_split": 0.8,
            "num_epochs": 10,
            "learning_rate": None,
            "batch_size": 8,
            "use_augmentation": False,
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
        }

        run_training_job(config)
        return True
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(0 if start_training() else 1)
