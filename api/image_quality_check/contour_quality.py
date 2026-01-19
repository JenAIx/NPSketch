#!/usr/bin/env python3
"""
Quality check for multi-structure drawings (e.g., two drawings stacked vertically).

What it does (per DB image):
1) Load image bytes from the database.
   - If --use-original:
     a) If red pixels exist (thresholded by --red-*), keep only red pixels,
        render them as black on white.
     b) If no red pixels exist, keep the original grayscale.
     c) Normalize line thickness to 2px using normalize_line_thickness().
   - Else: use processed_image_data as grayscale directly.
2) Binarize: background = white (255), ink = black (0).
3) Compute connected components, keep “large” components.
4) Compute:
   - Gap between large components (vertical separation).
   - Outside-ink ratio (ink outside the padded main component bbox).
   - External contour count (optionally filtered by min area).
5) Flag suspicious images based on the configured logic:
   - --require-gap-and-outside: gap AND outside-ink must match.
   - --require-contours: also require contour_count >= --min-contour-count.
   - If --contour-only: only contour_count decides.

Outputs a CSV containing only flagged images, plus diagnostics such as
gap size, outside ratio, contour count, and ink totals.

Usage examples:
  # Check all images (processed):
  python3 contour_quality.py --source ALL --min-gap 80 --require-gap-and-outside --require-contours \
    --min-contour-area 1 --min-contour-count 2 --output /app/data/tmp/quality_flags_current.csv

  # Check all images using originals with red extraction:
  python3 contour_quality.py --source ALL --use-original --min-gap 80 --require-gap-and-outside \
    --require-contours --min-contour-area 1 --min-contour-count 2 \
    --output /app/data/tmp/quality_flags_current.csv
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure /app (container) or repo root is on sys.path for imports
if Path("/app").exists():
    sys.path.insert(0, "/app")
else:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    sys.path.insert(0, str(repo_root))

from database import get_db, TrainingDataImage
from ocs_extraction.ocs_extractor import normalize_line_thickness


DEFAULT_RED_THRESHOLD = {"r_min": 200, "g_max": 100, "b_max": 100}
DEFAULT_PARAMS = {
    "min_component_area": 200,
    "min_component_ratio": 0.05,
    "min_gap_px": 40,
    "merge_kernel": 3,
    "merge_iterations": 1,
    "peak_threshold_ratio": 0.35,
    "min_peak_separation": 40,
    "outside_margin_ratio": 0.15,
    "outside_ink_ratio": 0.08,
    "min_contour_area": 200,
    "contour_only": False,
    "require_gap_and_outside": False,
    "min_component_height_ratio": 0.1,
    "require_contours": False,
    "min_contour_count": 2,
}


def preprocess_original_image(image_bytes: bytes, red_threshold: dict) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    red_mask = (r >= red_threshold["r_min"]) & (g <= red_threshold["g_max"]) & (b <= red_threshold["b_max"])
    red_count = int(red_mask.sum())

    if red_count > 0:
        # Use only red pixels, render them as black on white background
        rendered = np.ones_like(img_rgb, dtype=np.uint8) * 255
        rendered[red_mask] = [0, 0, 0]
        normalized = normalize_line_thickness(rendered, target_thickness=2)
        return cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)

    # No red pixels: use original grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    normalized = normalize_line_thickness(gray, target_thickness=2)
    if len(normalized.shape) == 3:
        return cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
    return normalized


def analyze_image(
    image_bytes: bytes,
    use_original: bool,
    red_threshold: dict,
    min_component_area: int,
    min_component_ratio: float,
    min_gap_px: int,
    merge_kernel: int,
    merge_iterations: int,
    peak_threshold_ratio: float,
    min_peak_separation: int,
    outside_margin_ratio: float,
    outside_ink_ratio: float,
    min_contour_area: int,
    contour_only: bool,
    require_gap_and_outside: bool,
    min_component_height_ratio: float,
    require_contours: bool,
    min_contour_count: int,
):
    if use_original:
        img = preprocess_original_image(image_bytes, red_threshold)
    else:
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    ink = (binary == 0).astype(np.uint8)
    if merge_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (merge_kernel, merge_kernel)
        )
        ink = cv2.morphologyEx(
            ink, cv2.MORPH_CLOSE, kernel, iterations=merge_iterations
        )
    ink_total = int(ink.sum())
    if ink_total == 0:
        return {
            "flagged": False,
            "reason": "no_ink",
            "ink_total": 0,
            "components": [],
            "gap_px": None,
        }

    # Projection-based check: look for two strong horizontal peaks separated by a gap
    row_sums = ink.sum(axis=1).astype(np.float32)
    if row_sums.max() > 0:
        row_sums_norm = row_sums / row_sums.max()
    else:
        row_sums_norm = row_sums

    peak_threshold = peak_threshold_ratio
    peaks = row_sums_norm >= peak_threshold
    # Find contiguous peak segments
    peak_segments = []
    in_seg = False
    start = 0
    for i, val in enumerate(peaks):
        if val and not in_seg:
            start = i
            in_seg = True
        elif not val and in_seg:
            peak_segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        peak_segments.append((start, len(peaks) - 1))

    # Keep only "substantial" segments (at least 3 rows)
    peak_segments = [seg for seg in peak_segments if (seg[1] - seg[0] + 1) >= 3]

    projection_gap = None
    if len(peak_segments) >= 2:
        # Compute largest separation between consecutive peak segments
        peak_segments = sorted(peak_segments, key=lambda s: s[0])
        for a, b in zip(peak_segments, peak_segments[1:]):
            gap = b[0] - a[1] - 1
            if projection_gap is None or gap > projection_gap:
                projection_gap = gap

    projection_flag = (
        projection_gap is not None and projection_gap >= min_peak_separation
    )

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        ink, connectivity=8
    )
    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cy = centroids[i][1]
        components.append(
            {
                "i": int(i),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "cy": float(cy),
            }
        )

    components = sorted(components, key=lambda c: c["area"], reverse=True)
    large_threshold = max(min_component_area, int(ink_total * min_component_ratio))
    large = [c for c in components if c["area"] >= large_threshold]
    min_height_px = int(ink.shape[0] * min_component_height_ratio)
    large_for_gap = [c for c in large if c["h"] >= min_height_px]

    # Compute "outside ink" relative to main component bbox
    outside_flag = False
    outside_ratio = 0.0
    if components:
        main = components[0]
        pad_x = int(main["w"] * outside_margin_ratio)
        pad_y = int(main["h"] * outside_margin_ratio)
        x0 = max(0, main["x"] - pad_x)
        y0 = max(0, main["y"] - pad_y)
        x1 = min(ink.shape[1] - 1, main["x"] + main["w"] + pad_x)
        y1 = min(ink.shape[0] - 1, main["y"] + main["h"] + pad_y)
        main_mask = np.zeros_like(ink, dtype=np.uint8)
        main_mask[y0 : y1 + 1, x0 : x1 + 1] = 1
        outside_ink = int((ink * (1 - main_mask)).sum())
        outside_ratio = outside_ink / ink_total if ink_total else 0.0
        outside_flag = outside_ratio >= outside_ink_ratio

    # Contour-based element count (more tolerant of broken frames)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_contour_area:
            contour_count += 1

    if contour_only:
        return {
            "flagged": contour_count >= min_contour_count,
            "reason": "contours" if contour_count >= min_contour_count else "single_contour",
            "ink_total": ink_total,
            "components": large,
            "gap_px": None,
            "outside_ratio": outside_ratio,
            "projection_gap": projection_gap,
            "contours": contour_count,
        }

    if len(large_for_gap) < 2 and not projection_flag and not outside_flag:
        return {
            "flagged": False,
            "reason": "single_component",
            "ink_total": ink_total,
            "components": large,
            "gap_px": None,
            "outside_ratio": outside_ratio,
            "projection_gap": projection_gap,
            "contours": contour_count,
        }

    large_sorted = sorted(large_for_gap, key=lambda c: c["y"])
    max_gap = 0
    for a, b in zip(large_sorted, large_sorted[1:]):
        gap = b["y"] - (a["y"] + a["h"])
        if gap > max_gap:
            max_gap = gap

    if require_gap_and_outside:
        flagged = (max_gap >= min_gap_px) and outside_flag
        reason = "gap_and_outside" if flagged else "no_gap_outside"
    else:
        flagged = max_gap >= min_gap_px or projection_flag or outside_flag
        reason = (
            "gap"
            if max_gap >= min_gap_px
            else ("projection_gap" if projection_flag else "outside_ink")
        )

    if require_contours:
        flagged = flagged and (contour_count >= min_contour_count)
        if flagged:
            reason = f"{reason}_and_contours"
        else:
            reason = "no_contours"

    return {
        "flagged": flagged,
        "reason": reason,
        "ink_total": ink_total,
        "components": large,
        "gap_px": int(max_gap),
        "outside_ratio": outside_ratio,
        "projection_gap": projection_gap,
        "contours": contour_count,
    }


def run_quality_check(
    db,
    source: str = None,
    limit: int = None,
    use_original: bool = True,
    red_threshold: dict = None,
    params: dict = None,
):
    """Run quality check across DB entries and return list of flagged results."""
    if red_threshold is None:
        red_threshold = DEFAULT_RED_THRESHOLD
    merged_params = DEFAULT_PARAMS.copy()
    if params:
        merged_params.update(params)

    query = db.query(TrainingDataImage)
    if source and source.upper() != "ALL":
        query = query.filter(TrainingDataImage.source_format == source)
    if limit and limit > 0:
        query = query.limit(limit)
    entries = query.all()

    flagged = []
    for entry in entries:
        image_bytes = (
            entry.original_file_data if use_original else entry.processed_image_data
        )
        result = analyze_image(
            image_bytes,
            use_original=use_original,
            red_threshold=red_threshold,
            min_component_area=merged_params["min_component_area"],
            min_component_ratio=merged_params["min_component_ratio"],
            min_gap_px=merged_params["min_gap_px"],
            merge_kernel=merged_params["merge_kernel"],
            merge_iterations=merged_params["merge_iterations"],
            peak_threshold_ratio=merged_params["peak_threshold_ratio"],
            min_peak_separation=merged_params["min_peak_separation"],
            outside_margin_ratio=merged_params["outside_margin_ratio"],
            outside_ink_ratio=merged_params["outside_ink_ratio"],
            min_contour_area=merged_params["min_contour_area"],
            contour_only=merged_params["contour_only"],
            require_gap_and_outside=merged_params["require_gap_and_outside"],
            min_component_height_ratio=merged_params["min_component_height_ratio"],
            require_contours=merged_params["require_contours"],
            min_contour_count=merged_params["min_contour_count"],
        )
        if result and result.get("flagged"):
            flagged.append(
                {
                    "id": entry.id,
                    "source_format": entry.source_format,
                    "task_type": entry.task_type,
                    "original_filename": entry.original_filename,
                    "reason": result.get("reason"),
                    "gap_px": result.get("gap_px"),
                    "projection_gap": result.get("projection_gap"),
                    "outside_ratio": result.get("outside_ratio"),
                    "ink_total": result.get("ink_total"),
                    "components": len(result.get("components", [])),
                    "contours": result.get("contours"),
                }
            )

    return flagged


def main():
    parser = argparse.ArgumentParser(description="TeleFred quality check for stacked drawings")
    parser.add_argument(
        "--source",
        default="TELEFRED",
        help="Source format to scan or ALL (default: TELEFRED)",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=40,
        help="Min vertical gap in pixels between large components (default: 40)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="Min area in pixels for a component to be considered large (default: 200)",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.05,
        help="Min fraction of total ink to be considered large (default: 0.05)",
    )
    parser.add_argument(
        "--merge-kernel",
        type=int,
        default=3,
        help="Morphological closing kernel size to merge nearby components (default: 3, 0 disables)",
    )
    parser.add_argument(
        "--merge-iterations",
        type=int,
        default=1,
        help="Iterations for morphological closing (default: 1)",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=0.35,
        help="Row projection threshold ratio (default: 0.35)",
    )
    parser.add_argument(
        "--min-peak-gap",
        type=int,
        default=40,
        help="Min vertical gap between projection peaks (default: 40)",
    )
    parser.add_argument(
        "--outside-margin-ratio",
        type=float,
        default=0.15,
        help="Padding around main component for outside-ink check (default: 0.15)",
    )
    parser.add_argument(
        "--outside-ink-ratio",
        type=float,
        default=0.08,
        help="Min fraction of ink outside main bbox to flag (default: 0.08)",
    )
    parser.add_argument(
        "--min-component-height-ratio",
        type=float,
        default=0.1,
        help="Min component height ratio for gap check (default: 0.1)",
    )
    parser.add_argument(
        "--require-contours",
        action="store_true",
        help="Require contour count to be >= min-contour-count",
    )
    parser.add_argument(
        "--min-contour-count",
        type=int,
        default=2,
        help="Min contour count for contour-based checks (default: 2)",
    )
    parser.add_argument(
        "--min-contour-area",
        type=int,
        default=200,
        help="Min contour area for contour counting (default: 200)",
    )
    parser.add_argument(
        "--contour-only",
        action="store_true",
        help="Flag based only on contour count (>=2)",
    )
    parser.add_argument(
        "--require-gap-and-outside",
        action="store_true",
        help="Flag only when BOTH gap and outside-ink criteria match",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images scanned (<=0 means no limit)",
    )
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="Analyze original_file_data instead of processed_image_data",
    )
    parser.add_argument(
        "--red-r-min",
        type=int,
        default=DEFAULT_RED_THRESHOLD["r_min"],
        help="Red detection: minimum R value (default: 200)",
    )
    parser.add_argument(
        "--red-g-max",
        type=int,
        default=DEFAULT_RED_THRESHOLD["g_max"],
        help="Red detection: maximum G value (default: 100)",
    )
    parser.add_argument(
        "--red-b-max",
        type=int,
        default=DEFAULT_RED_THRESHOLD["b_max"],
        help="Red detection: maximum B value (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="telefred_quality_flags.csv",
        help="Output CSV filename (default: telefred_quality_flags.csv)",
    )
    args = parser.parse_args()

    db = next(get_db())
    query = db.query(TrainingDataImage)
    if args.source.upper() != "ALL":
        query = query.filter(TrainingDataImage.source_format == args.source)
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    entries = query.all()
    red_threshold = {
        "r_min": args.red_r_min,
        "g_max": args.red_g_max,
        "b_max": args.red_b_max,
    }

    flagged = []
    scanned = 0

    for entry in entries:
        scanned += 1
        image_bytes = (
            entry.original_file_data if args.use_original else entry.processed_image_data
        )
        result = analyze_image(
            image_bytes,
            use_original=args.use_original,
            red_threshold=red_threshold,
            min_component_area=args.min_area,
            min_component_ratio=args.min_area_ratio,
            min_gap_px=args.min_gap,
            merge_kernel=args.merge_kernel,
            merge_iterations=args.merge_iterations,
            peak_threshold_ratio=args.peak_threshold,
            min_peak_separation=args.min_peak_gap,
            outside_margin_ratio=args.outside_margin_ratio,
            outside_ink_ratio=args.outside_ink_ratio,
            min_contour_area=args.min_contour_area,
            contour_only=args.contour_only,
            require_gap_and_outside=args.require_gap_and_outside,
            min_component_height_ratio=args.min_component_height_ratio,
            require_contours=args.require_contours,
            min_contour_count=args.min_contour_count,
        )
        if result is None:
            continue
        if result["flagged"]:
            flagged.append(
                {
                    "id": entry.id,
                    "source_format": entry.source_format,
                    "task_type": entry.task_type,
                    "original_filename": entry.original_filename,
                    "gap_px": result["gap_px"],
                    "ink_total": result["ink_total"],
                    "components": len(result["components"]),
                    "outside_ratio": result.get("outside_ratio"),
                    "projection_gap": result.get("projection_gap"),
                    "reason": result.get("reason"),
                    "contours": result.get("contours"),
                }
            )

    db.close()

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("id,source_format,task_type,original_filename,reason,gap_px,projection_gap,outside_ratio,ink_total,components,contours\n")
        for row in flagged:
            f.write(
                f"{row['id']},{row['source_format']},{row['task_type']},{row['original_filename']},{row['reason']},{row['gap_px']},{row['projection_gap']},{row['outside_ratio']},{row['ink_total']},{row['components']},{row['contours']}\n"
            )

    print("Quality check complete")
    print(f"Scanned: {scanned}")
    print(f"Flagged: {len(flagged)}")
    print(f"Output:  {output_path}")


if __name__ == "__main__":
    main()
