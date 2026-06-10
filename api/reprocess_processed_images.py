#!/usr/bin/env python3
"""
Re-render processed_image_data for all training images from their stored
originals, using the fixed rendering pipeline (2026-06):

  - aspect-ratio-PRESERVING fit to 568x274, centered on white
    (previous imports stretched the content bbox to the full canvas,
    distorting every drawing by a content-dependent factor)
  - true 2px line thickness (line_normalizer kernel fix)

Per source:
  TELEFRED / OCS      red-pixel extraction from the original scan, then
                      AR-preserving render (ocs_extractor)
  OXFORD / ALGORITHM  content-bbox crop + AR-preserving fit + 2px line norm
                      (oxford_normalizer)
  MAT / DRAWN / *     fallback: 2px line re-normalization of the existing
                      processed image (original is not a renderable PNG /
                      AR cannot be reconstructed) - logged

BACK UP data/npsketch.db BEFORE RUNNING.

Usage (inside the container):
    docker exec -e PYTHONPATH=/app npsketch-api python3 \
        /app/reprocess_processed_images.py [--limit N] [--sources TELEFRED,OXFORD,...]
"""
import argparse
import io
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, "/app")

from database import SessionLocal, TrainingDataImage
from line_normalizer import normalize_line_thickness
from ocs_extraction.ocs_extractor import (
    extract_red_pixels,
    calculate_red_bbox,
    render_red_pixels_to_image,
)
from oxford_extraction.oxford_normalizer import normalize_oxford_image

RED_THRESHOLD = {"r_min": 200, "g_max": 100, "b_max": 100}  # matches import defaults
CANVAS = (568, 274)
TMP_DIR = "/app/data/tmp/reprocess"


def rerender_red(original_bytes: bytes) -> bytes:
    """TELEFRED/OCS: red-pixel extraction + AR-preserving render."""
    with tempfile.NamedTemporaryFile(suffix=".png", dir=TMP_DIR, delete=False) as f_in:
        f_in.write(original_bytes)
        in_path = f_in.name
    out_path = in_path + "_out.png"
    try:
        red_mask, _ = extract_red_pixels(in_path, RED_THRESHOLD)
        if int(red_mask.sum()) == 0:
            raise ValueError("no red pixels in original")
        bbox = calculate_red_bbox(red_mask, padding=5)
        if not render_red_pixels_to_image(red_mask, bbox, out_path, canvas_size=CANVAS):
            raise ValueError("render failed")
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass


def rerender_scan(original_bytes: bytes) -> bytes:
    """OXFORD/ALGORITHM: bbox crop + AR-preserving fit + 2px line norm."""
    with tempfile.NamedTemporaryFile(suffix=".png", dir=TMP_DIR, delete=False) as f_in:
        f_in.write(original_bytes)
        in_path = f_in.name
    out_path = in_path + "_out.png"
    try:
        ok = normalize_oxford_image(
            in_path, out_path, target_size=CANVAS,
            auto_crop=True, padding=5, target_thickness=2, verbose=False,
        )
        if not ok:
            raise ValueError("normalize_oxford_image failed")
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass


def renorm_only(processed_bytes: bytes) -> bytes:
    """Fallback: re-normalize line thickness of the existing processed image."""
    img = Image.open(io.BytesIO(processed_bytes)).convert("RGB")
    normalized = normalize_line_thickness(np.array(img), target_thickness=2)
    out = io.BytesIO()
    Image.fromarray(normalized, mode="RGB").save(out, format="PNG")
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="TELEFRED,OCS,OXFORD,ALGORITHM,MAT,DRAWN")
    parser.add_argument("--limit", type=int, default=0, help="max rows per source (0 = all)")
    parser.add_argument("--batch-commit", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(TMP_DIR, exist_ok=True)
    sources = [s.strip().upper() for s in args.sources.split(",") if s.strip()]

    db = SessionLocal()
    totals = {}
    try:
        for src in sources:
            q = db.query(TrainingDataImage).filter(TrainingDataImage.source_format == src)
            if args.limit > 0:
                q = q.limit(args.limit)
            rows = q.all()
            ok = failed = fallback = 0
            print(f"\n=== {src}: {len(rows)} rows ===", flush=True)

            for i, row in enumerate(rows):
                try:
                    if src in ("TELEFRED", "OCS"):
                        new_bytes = rerender_red(row.original_file_data)
                    elif src in ("OXFORD", "ALGORITHM"):
                        new_bytes = rerender_scan(row.original_file_data)
                    else:
                        new_bytes = renorm_only(row.processed_image_data)
                        fallback += 1
                    row.processed_image_data = new_bytes
                    ok += 1
                except Exception as e:
                    # Fallback to line re-normalization of the existing image
                    try:
                        row.processed_image_data = renorm_only(row.processed_image_data)
                        fallback += 1
                        ok += 1
                        print(f"  [{src} id={row.id}] re-render failed ({e}) -> renorm-only fallback", flush=True)
                    except Exception as e2:
                        failed += 1
                        print(f"  [{src} id={row.id}] FAILED entirely: {e} / {e2}", flush=True)

                if (i + 1) % args.batch_commit == 0:
                    db.commit()
                    print(f"  {src}: {i+1}/{len(rows)} committed", flush=True)

            db.commit()
            totals[src] = (ok, failed, fallback)
            print(f"=== {src} done: {ok} ok ({fallback} renorm-only), {failed} failed ===", flush=True)
    finally:
        db.close()

    print("\nSUMMARY:")
    for src, (ok, failed, fb) in totals.items():
        print(f"  {src:10} ok={ok} failed={failed} renorm_only={fb}")
    return all(f == 0 for _, f, _ in totals.values())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
