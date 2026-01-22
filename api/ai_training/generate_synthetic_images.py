#!/usr/bin/env python3
"""
Generate synthetic "bad" images and save them to disk.

Default: 5 images per complexity level, output to ./data/tmp/synthetic_images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.logger import get_logger
from database import SessionLocal
from config import get_config
from ai_training.synthetic_bad_images import generate_synthetic_bad_images


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic bad images for low-score augmentation."
    )
    parser.add_argument(
        "--output-dir",
        default="./data/tmp/synthetic_images",
        help="Directory to write PNGs to (default: ./data/tmp/synthetic_images)",
    )
    parser.add_argument(
        "--samples-per-level",
        type=int,
        default=5,
        help="Number of images per complexity level (default: 5)",
    )
    parser.add_argument(
        "--complexity-levels",
        type=int,
        default=5,
        help="Number of complexity levels (default: 5)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Base random seed for reproducible generation (default: 42)",
    )
    parser.add_argument(
        "--target-feature",
        default="Total_Score",
        help="Target feature name to include in per-image JSON (default: Total_Score)",
    )
    parser.add_argument(
        "--target-value",
        type=float,
        default=0.0,
        help="Target value to include in per-image JSON (default: 0.0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean output directory before each run
    removed = 0
    for path in output_dir.glob("*.png"):
        path.unlink(missing_ok=True)
        removed += 1
    for path in output_dir.glob("*.json"):
        path.unlink(missing_ok=True)
        removed += 1
    if removed:
        logger.info("Cleaned %s existing files from %s", removed, output_dir)

    config = get_config()
    synthetic_cfg = config.get("synthetic", {})
    defaults_cfg = synthetic_cfg.get("defaults", {})
    generation_cfg = synthetic_cfg.get("generation", {})

    score_threshold = float(defaults_cfg.get("score_threshold", 20.0))
    image_size = generation_cfg.get("image_size", [568, 274])
    image_size = (int(image_size[0]), int(image_size[1]))

    n_samples = args.samples_per_level * args.complexity_levels

    logger.info("Generating synthetic bad images...")
    logger.info(
        "Params: samples_per_level=%s, complexity_levels=%s, total=%s, score_threshold=%s, image_size=%s",
        args.samples_per_level,
        args.complexity_levels,
        n_samples,
        score_threshold,
        image_size,
    )
    logger.info("Output directory: %s", output_dir)

    db = SessionLocal()
    try:
        synthetic_images = generate_synthetic_bad_images(
            db=db,
            n_samples=n_samples,
            complexity_levels=args.complexity_levels,
            score_threshold=score_threshold,
            image_size=image_size,
            random_seed=args.random_seed,
        )
    finally:
        db.close()

    manifest = {
        "output_dir": str(output_dir),
        "samples_per_level": args.samples_per_level,
        "complexity_levels": args.complexity_levels,
        "total_samples": n_samples,
        "score_threshold": score_threshold,
        "image_size": list(image_size),
        "random_seed": args.random_seed,
        "target_feature": args.target_feature,
        "target_value": args.target_value,
        "images": [],
    }

    # Write files
    for idx, synth in enumerate(synthetic_images):
        level = synth.get("complexity_level", -1)
        filename = f"synthetic_bad_L{level:02d}_{idx:04d}.png"
        out_path = output_dir / filename
        out_path.write_bytes(synth["image_data"])
        json_name = f"synthetic_bad_L{level:02d}_{idx:04d}.json"
        json_path = output_dir / json_name

        image_id = f"synthetic_bad_{idx}"
        patient_id = f"SYNTHETIC_BAD_L{level}"
        per_image_payload = {
            "image_id": image_id,
            "patient_id": patient_id,
            "target_feature": args.target_feature,
            "target_value": args.target_value,
            "target_value_original": args.target_value,
            "augmentation": None,
            "synthetic": {
                "complexity_level": level,
                "complexity_value": synth.get("complexity_value"),
            },
        }
        json_path.write_text(json.dumps(per_image_payload, indent=2, sort_keys=True))
        manifest["images"].append(
            {
                "filename": filename,
                "json_filename": json_name,
                "image_id": image_id,
                "patient_id": patient_id,
                "complexity_level": level,
                "complexity_value": synth.get("complexity_value"),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    logger.info("Wrote %s images to %s", len(synthetic_images), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
