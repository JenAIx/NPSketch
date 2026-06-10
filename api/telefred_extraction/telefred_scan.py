#!/usr/bin/env python3
"""
Scan TeleFred training images for resolution + red-pixel extractability.

Pre-import sanity check: reports how many PNGs in fc0/ and fc1/ have red pixels,
how many are blank (zero red), how many fail to read, and the resolution mix.

Usage:
    python3 telefred_scan.py --input-base /app/templates/training_data_telefred_202606
"""

import argparse
import os
import sys
from collections import Counter
from statistics import median
from pathlib import Path

import numpy as np

# Ensure /app/ocs_extraction (container) or api/ocs_extraction is on sys.path
if Path("/app").exists():
    sys.path.insert(0, "/app/ocs_extraction")
else:
    script_path = Path(__file__).resolve()
    api_root = script_path.parents[1]  # api/telefred_extraction -> api
    sys.path.insert(0, str(api_root / "ocs_extraction"))

from ocs_extractor import load_config, extract_red_pixels


DEFAULT_RED_THRESHOLD = {"r_min": 200, "g_max": 100, "b_max": 100}


def find_png_files(directory):
    png_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(".png"):
                png_files.append(os.path.join(root, filename))
    return sorted(png_files)


def scan_folder(folder_path, red_threshold):
    png_files = find_png_files(folder_path)
    resolution_counts = Counter()
    red_pixel_counts = []
    zero_red_files = []
    failed_files = []

    for image_path in png_files:
        try:
            red_mask, original_shape = extract_red_pixels(image_path, red_threshold)
            height, width = original_shape[0], original_shape[1]
            resolution_counts[(width, height)] += 1
            red_count = int(np.sum(red_mask))
            red_pixel_counts.append(red_count)
            if red_count == 0:
                zero_red_files.append(image_path)
        except Exception:
            failed_files.append(image_path)

    return {
        "total_files": len(png_files),
        "resolution_counts": resolution_counts,
        "red_pixel_counts": red_pixel_counts,
        "zero_red_files": zero_red_files,
        "failed_files": failed_files,
    }


def format_resolution_counts(resolution_counts):
    if not resolution_counts:
        return "  (none)"
    lines = []
    for (width, height), count in sorted(resolution_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  - {width}x{height}: {count}")
    return "\n".join(lines)


def summarize_red_counts(red_pixel_counts):
    if not red_pixel_counts:
        return {"min": 0, "median": 0, "max": 0}
    return {
        "min": min(red_pixel_counts),
        "median": int(median(red_pixel_counts)),
        "max": max(red_pixel_counts),
    }


def print_summary(folder_name, results):
    total_files = results["total_files"]
    red_pixel_counts = results["red_pixel_counts"]
    zero_red_files = results["zero_red_files"]
    failed_files = results["failed_files"]

    red_positive = total_files - len(zero_red_files) - len(failed_files)
    red_stats = summarize_red_counts(red_pixel_counts)

    print("\n" + "=" * 80)
    print(f"Folder: {folder_name}")
    print("=" * 80)
    print(f"Total PNG files:           {total_files}")
    print(f"Files with red pixels:     {red_positive}")
    print(f"Files with zero red:       {len(zero_red_files)}")
    print(f"Failed to read:            {len(failed_files)}")
    print("\nResolution overview:")
    print(format_resolution_counts(results["resolution_counts"]))

    print("\nRed pixel count stats (all files):")
    print(f"  min: {red_stats['min']}")
    print(f"  median: {red_stats['median']}")
    print(f"  max: {red_stats['max']}")

    if zero_red_files:
        print("\nExample files with zero red pixels (up to 10):")
        for path in zero_red_files[:10]:
            print(f"  - {path}")

    if failed_files:
        print("\nExample files that failed to read (up to 10):")
        for path in failed_files[:10]:
            print(f"  - {path}")


def main():
    parser = argparse.ArgumentParser(description="Scan TeleFred PNGs for resolution and red pixels")
    parser.add_argument(
        "--input-base",
        required=True,
        help="Base directory containing fc0/ and fc1/ folders",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to ocs_extractor.conf (optional)",
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
    if config and "red_threshold" in config:
        red_threshold = config["red_threshold"]

    base_dir = args.input_base
    folders = [
        ("fc0", os.path.join(base_dir, "fc0")),
        ("fc1", os.path.join(base_dir, "fc1")),
    ]

    print("=" * 80)
    print("TeleFred PNG Scan")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Red threshold: r>={red_threshold['r_min']}, g<={red_threshold['g_max']}, b<={red_threshold['b_max']}")
    print("=" * 80)

    for folder_name, folder_path in folders:
        if not os.path.isdir(folder_path):
            print(f"\n⚠ Missing folder: {folder_path}")
            continue
        results = scan_folder(folder_path, red_threshold)
        print_summary(folder_name, results)


if __name__ == "__main__":
    main()
