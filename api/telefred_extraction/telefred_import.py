#!/usr/bin/env python3
"""
Import TeleFred PNGs into training_data_images using OCS-style red-pixel extraction.

Only images with >=1 red pixel are imported. Duplicates (by SHA256 of the original
PNG bytes) are skipped, so a re-export of the same drawing is never imported twice.

Multiple --base directories can be given. They are processed in the order listed, and
because duplicate detection consults the database, the FIRST base that contains a given
(byte-identical) image wins. This is how a union of several deliveries is built with a
defined label precedence: list the newest/most-authoritative dataset first.

Usage:
    # single dataset
    python3 telefred_import.py --base /app/templates/training_data_telefred_202606 --limit 0

    # union of two deliveries, 202606 labels take precedence over 20260119
    python3 telefred_import.py \
        --base /app/templates/training_data_telefred_202606 \
        --base /app/templates/training_data_telefred_20260119 \
        --limit 0
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure /app (container) or repo root/api is on sys.path for imports
if Path("/app").exists():
    sys.path.insert(0, "/app")
else:
    script_path = Path(__file__).resolve()
    api_root = script_path.parents[1]  # api/telefred_extraction -> api
    sys.path.insert(0, str(api_root))

from database import get_db, TrainingDataImage
from ocs_extraction.ocs_extractor import (
    load_config,
    extract_red_pixels,
    calculate_red_bbox,
    render_red_pixels_to_image,
)


DEFAULT_RED_THRESHOLD = {"r_min": 200, "g_max": 100, "b_max": 100}


def load_csv_rows(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def select_images_with_red(
    image_dir: Path,
    csv_rows,
    red_threshold: dict,
    min_score: int,
    limit: int,
):
    selected = []
    missing_files = []
    zero_red_files = []
    unreadable_files = []

    for row in csv_rows:
        filename = row.get("FileName")
        if not filename:
            continue
        image_path = image_dir / filename
        if not image_path.exists():
            missing_files.append(filename)
            continue

        total_score_value = row.get("TotalScore")
        if total_score_value not in (None, ""):
            try:
                total_score = int(total_score_value)
            except ValueError:
                total_score = None
        else:
            total_score = None

        if min_score > 0 and (total_score is None or total_score < min_score):
            continue

        try:
            red_mask, _ = extract_red_pixels(str(image_path), red_threshold)
            red_count = int(red_mask.sum())
        except Exception:
            unreadable_files.append(filename)
            continue

        if red_count == 0:
            zero_red_files.append(filename)
            continue

        selected.append(filename)
        if limit > 0 and len(selected) >= limit:
            break

    return selected, missing_files, zero_red_files, unreadable_files


def import_images(
    folder_name: str,
    image_dir: Path,
    csv_rows,
    red_threshold: dict,
    padding: int,
    min_score: int,
    limit: int,
    session_id: str,
    output_dir: Path,
    dataset_name: str,
):
    db = next(get_db())
    selected, missing_files, zero_red_files, unreadable_files = select_images_with_red(
        image_dir, csv_rows, red_threshold, min_score, limit
    )

    print("\n" + "=" * 80)
    print(f"Dataset: {dataset_name}  |  Folder: {folder_name}")
    print("=" * 80)
    limit_label = "no limit" if limit <= 0 else str(limit)
    print(f"Requested limit:           {limit_label}")
    print(f"Min TotalScore:            {min_score}")
    print(f"Selected (>=1 red pixel):  {len(selected)}")
    print(f"Missing files in folder:   {len(missing_files)}")
    print(f"Zero-red files:            {len(zero_red_files)}")
    print(f"Unreadable files:          {len(unreadable_files)}")

    stats = {
        "processed": 0,
        "imported": 0,
        "duplicates": 0,
        "errors": 0,
        "skipped_zero_red": 0,
        "skipped_missing": len(missing_files),
        "skipped_unreadable": len(unreadable_files),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, filename in enumerate(selected, start=1):
        row = next((r for r in csv_rows if r.get("FileName") == filename), None)
        if row is None:
            stats["errors"] += 1
            print(f"[{idx:4d}/{len(selected)}] {filename} - CSV row not found")
            continue

        image_path = image_dir / filename
        try:
            original_bytes = image_path.read_bytes()
            image_hash = hashlib.sha256(original_bytes).hexdigest()

            existing = (
                db.query(TrainingDataImage)
                .filter(TrainingDataImage.image_hash == image_hash)
                .first()
            )
            if existing:
                stats["duplicates"] += 1
                print(
                    f"[{idx:4d}/{len(selected)}] {filename} - duplicate of ID {existing.id}"
                )
                continue

            red_mask, original_shape = extract_red_pixels(
                str(image_path), red_threshold
            )
            red_count = int(red_mask.sum())
            if red_count == 0:
                stats["skipped_zero_red"] += 1
                print(f"[{idx:4d}/{len(selected)}] {filename} - zero red")
                continue

            bbox = calculate_red_bbox(red_mask, padding=padding)
            if bbox is not None:
                bbox = [int(value) for value in bbox]

            processed_path = output_dir / filename
            canvas_size = (568, 274)
            success = render_red_pixels_to_image(
                red_mask, bbox, str(processed_path), canvas_size=canvas_size
            )
            if not success or not processed_path.exists():
                stats["errors"] += 1
                print(f"[{idx:4d}/{len(selected)}] {filename} - render failed")
                continue

            processed_bytes = processed_path.read_bytes()

            total_score_value = row.get("TotalScore")
            total_score = int(total_score_value) if total_score_value not in (None, "") else None

            features_data = None
            if total_score is not None and total_score != 0:
                features_data = json.dumps({"Total_Score": total_score})

            patient_id = str(row.get("ID") or "")
            task_type = folder_name.upper()

            metadata = {
                "source": "telefred",
                "source_dataset": dataset_name,
                "source_folder": folder_name,
                "original_resolution": [int(original_shape[1]), int(original_shape[0])],
                "red_pixel_count": int(red_count),
                "bbox": bbox,
                "padding_px": padding,
                "red_threshold": red_threshold,
            }

            entry = TrainingDataImage(
                patient_id=patient_id,
                task_type=task_type,
                source_format="TELEFRED",
                original_filename=filename,
                original_file_data=original_bytes,
                processed_image_data=processed_bytes,
                image_hash=image_hash,
                extraction_metadata=json.dumps(metadata),
                features_data=features_data,
                session_id=session_id,
            )

            db.add(entry)
            db.commit()
            stats["imported"] += 1
            stats["processed"] += 1
            print(f"[{idx:4d}/{len(selected)}] {filename} - imported ID {entry.id}")
        except Exception as e:
            db.rollback()
            stats["errors"] += 1
            print(f"[{idx:4d}/{len(selected)}] {filename} - error: {e}")

    db.close()

    print("\nSummary:")
    print(f"  Imported:    {stats['imported']}")
    print(f"  Duplicates:  {stats['duplicates']}")
    print(f"  Errors:      {stats['errors']}")
    print(f"  Zero-red:    {stats['skipped_zero_red']}")
    print(f"  Missing:     {stats['skipped_missing']}")
    print(f"  Unreadable:  {stats['skipped_unreadable']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Import TeleFred images (red pixels only)")
    parser.add_argument(
        "--base",
        action="append",
        required=True,
        dest="bases",
        help="Base directory with fc0/fc1 folders + csv files. "
        "Repeatable; earlier entries win on duplicate detection (label precedence).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max images per folder per base (<=0 means no limit)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum TotalScore to include (default: 0)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to ocs_extractor.conf (optional)",
    )
    parser.add_argument(
        "--output",
        default="/app/data/tmp/telefred_processed",
        help="Temporary output directory for processed images",
    )
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = "/app/ocs_extraction/ocs_extractor.conf"
        if not os.path.exists(config_path):
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "ocs_extractor.conf"
            )

    config = load_config(config_path) if os.path.exists(config_path) else None
    red_threshold = DEFAULT_RED_THRESHOLD
    padding = 5
    if config:
        red_threshold = config.get("red_threshold", red_threshold)
        padding = int(config.get("padding_px", padding))

    session_id = f"telefred_{datetime.now().strftime('%Y%m%d')}"
    output_root = Path(args.output) / session_id

    print("=" * 80)
    print("TeleFred Import")
    print("=" * 80)
    print(f"Bases (precedence order): {args.bases}")
    print(f"Session ID:     {session_id}")
    print(f"Limit/folder:   {args.limit}")
    print(
        f"Red threshold:  r>={red_threshold['r_min']}, g<={red_threshold['g_max']}, b<={red_threshold['b_max']}"
    )
    print(f"Padding:        {padding}px")
    print("=" * 80)

    grand_total = {"imported": 0, "duplicates": 0, "errors": 0}

    for base in args.bases:
        base_dir = Path(base)
        dataset_name = base_dir.name
        for folder_name in ["fc0", "fc1"]:
            csv_path = base_dir / f"{folder_name}.csv"
            image_dir = base_dir / folder_name
            if not csv_path.exists() or not image_dir.exists():
                print(f"\n⚠ Missing {folder_name} data in {dataset_name}: {csv_path} or {image_dir}")
                continue

            csv_rows = load_csv_rows(csv_path)
            stats = import_images(
                folder_name,
                image_dir,
                csv_rows,
                red_threshold,
                padding,
                args.min_score,
                args.limit,
                session_id,
                output_root / dataset_name / folder_name,
                dataset_name,
            )
            grand_total["imported"] += stats["imported"]
            grand_total["duplicates"] += stats["duplicates"]
            grand_total["errors"] += stats["errors"]

    print("\n" + "=" * 80)
    print("GRAND TOTAL (all bases / folders)")
    print("=" * 80)
    print(f"  Imported:    {grand_total['imported']}")
    print(f"  Duplicates:  {grand_total['duplicates']}")
    print(f"  Errors:      {grand_total['errors']}")
    print(f"  Session ID:  {session_id}")


if __name__ == "__main__":
    main()
