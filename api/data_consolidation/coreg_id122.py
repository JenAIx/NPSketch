#!/usr/bin/env python3
"""Clean, from-scratch coregistration of a single drawing (default id 122).

Pipeline, one stage at a time, each dumped for visual inspection:
  1. original scan            (raw original_file_data, red ink on printed template)
  2. red extraction + content (process_red: red-pixel mask -> content bbox ->
                               render 568x274 -> 2px line-normalize)
  3. pystackreg alignment      (anisotropic bbox prealign -> SCALED_ROTATION refine,
                               shear-free so straight verticals stay vertical)
  4. overlay                   (reference = blue, coregistered = red, overlap = purple)

Writes to data/tmp/pystackreg/ and prints verification metrics so the result can
be double-checked numerically, not just by eye.
"""
import os, sys, argparse
import numpy as np
import cv2
from PIL import Image
import io

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/data_consolidation")

from database import get_db, TrainingDataImage
from import_unified import process_red
from coregistration import (
    W, H, gray, ink, bbox, align, overlap, overlay_blue_red, load_reference,
)

OUT = "/app/data/tmp/pystackreg"
os.makedirs(OUT, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=122)
    args = ap.parse_args()

    db = next(get_db())
    row = db.query(TrainingDataImage).filter(TrainingDataImage.id == args.id).first()
    db.close()
    if row is None:
        print(f"id {args.id} not found"); return

    # 1. original scan (red ink on printed template)
    orig = np.array(Image.open(io.BytesIO(row.original_file_data)).convert("RGB"))
    cv2.imwrite(f"{OUT}/01_id{args.id}_original.png", cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))

    # 2. red extraction + content detection (bbox) + render + 2px line-normalize
    red_bytes = process_red(row.original_file_data)
    if red_bytes is None:
        print("no red ink extracted"); return
    content = gray(red_bytes)
    cv2.imwrite(f"{OUT}/02_id{args.id}_content.png", content)

    # reference, used AS-IS (never distorted)
    ref = load_reference()

    # 3. align
    aligned, status = align(content, ref)
    cv2.imwrite(f"{OUT}/03_id{args.id}_aligned.png", aligned)

    # 4. overlay
    ov = overlay_blue_red(ref, aligned)
    cv2.imwrite(f"{OUT}/04_id{args.id}_overlay.png", ov)

    # ---- verification (double-check the result numerically) ----
    bb = bbox(aligned)
    col = ink(aligned).sum(0)
    # rightmost strong vertical: the rectangle's right edge. Look in the right third.
    right = col[2 * W // 3:]
    print(f"id{args.id}  status={status}")
    print(f"  content ink px : {int(ink(content).sum())}")
    print(f"  aligned ink px : {int(ink(aligned).sum())}  (retention {ink(aligned).sum()/max(1,ink(content).sum()):.2f})")
    print(f"  aligned bbox   : x[{bb[0]}..{bb[2]}] y[{bb[1]}..{bb[3]}]  (margins L{bb[0]} R{W-1-bb[2]} T{bb[1]} B{H-1-bb[3]})")
    print(f"  overlap content->ref : {overlap(content, ref):.2f}")
    print(f"  overlap aligned->ref : {overlap(aligned, ref):.2f}")
    print(f"  right-third max col-ink: {int(right.max())} at x={2*W//3 + int(right.argmax())}  (a clean vertical ~ figure height)")
    print(f"  outputs in {OUT}: 01_original 02_content 03_aligned 04_overlay")


if __name__ == "__main__":
    main()
