#!/usr/bin/env python3
"""
THE single batch importer: consolidated base (templates/labels.csv + img/) -> DB.

Replaces the per-source CLI populators. Reads the unified labels, preprocesses
each raw original to the standard 568x274 / 2px format, and inserts one
`training_data_images` row per (curated) image.

Image-style is AUTO-DETECTED per image via a priority cascade (see detect_style):
  1. red ink present                          -> red extraction (TeleFred-style)
  2. no red + dark content + black-source name -> bbox-crop (black lines)
  3. otherwise heuristic (red-template w/o ink = blank; else by dark content)
The source name is only a consistency anchor for steps 2-3 — pixel detection
decides. A blank TeleFred scan still shows the printed black template (lots of
dark) but has no red ink, so it is correctly treated as blank.
Detected-vs-expected style mismatches are logged as a QA signal.

Curation:
  - blanks (no ink) skipped
  - SHA256-dedup on the original bytes; a hash group carrying >1 distinct
    nonzero score is corrupt -> whole group dropped
  - negative score -> dropped
  - |total_score - sublabel sum| > 5 -> kept, flagged label_warning

Run in the container:
    docker exec -e PYTHONPATH=/app npsketch-api python3 /app/data_consolidation/import_unified.py
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, "/app")

from database import SessionLocal, TrainingDataImage, init_database
from ocs_extraction.ocs_extractor import (
    extract_red_pixels, calculate_red_bbox, render_red_pixels_to_image,
)
from oxford_extraction.oxford_normalizer import normalize_oxford_image

RED_THRESHOLD = {"r_min": 200, "g_max": 100, "b_max": 100}
RED_MIN = 50      # >= this many red px -> patient drew in red ink
DARK_MIN = 50     # >= this many dark px -> has black-line content
TMP_DIR = "/app/data/tmp/import_unified"

# Expected style per source (consistency anchor; detection still decides).
RED_SOURCES = {"TELEFRED"}                          # red ink on a printed (black) template
BLACK_SOURCES = {"OXFORD", "ALGORITHM", "OCS_MACHINE"}  # black lines on white

ELEM_TRIPLETS = [(f"ELEM{n:02d}PRES", f"ELEM{n:02d}ACC", f"ELEM{n:02d}POS") for n in range(1, 21)]


def detect_style(orig_bytes, source):
    """
    Auto-detect the image style. Clean priority cascade:
      1. red ink present                         -> "red"   (red extraction)
      2. no red, dark content + black-source name -> "black" (bbox-crop)
      3. otherwise                                -> heuristic:
           - red-template source w/o red ink = blank (dark is the printed template)
           - unknown source: dark content -> black, else blank

    Returns (style, red_count, dark_count); style in {red, black, blank}.
    """
    arr = np.array(Image.open(io.BytesIO(orig_bytes)).convert("RGB"))
    r = arr[:, :, 0].astype(int); g = arr[:, :, 1].astype(int); b = arr[:, :, 2].astype(int)
    red_count = int(((r >= RED_THRESHOLD["r_min"]) &
                     (g <= RED_THRESHOLD["g_max"]) &
                     (b <= RED_THRESHOLD["b_max"])).sum())
    dark_count = int(((0.299 * r + 0.587 * g + 0.114 * b) < 100).sum())

    # 1. red ink -> red, regardless of the source label
    if red_count >= RED_MIN:
        return "red", red_count, dark_count
    # 2. no red, has black content, and the name says black-line source
    if source in BLACK_SOURCES and dark_count >= DARK_MIN:
        return "black", red_count, dark_count
    # 3. heuristic fallback
    if source in RED_SOURCES:
        return "blank", red_count, dark_count   # red-template scan, no ink = blank
    return ("black" if dark_count >= DARK_MIN else "blank"), red_count, dark_count


def process_red(orig_bytes):
    """Red-pixel extraction -> processed PNG bytes, or None if no red bbox."""
    with tempfile.NamedTemporaryFile(suffix=".png", dir=TMP_DIR, delete=False) as f:
        f.write(orig_bytes); in_path = f.name
    out_path = in_path + "_out.png"
    try:
        red_mask, _ = extract_red_pixels(in_path, RED_THRESHOLD)
        if int(red_mask.sum()) == 0:
            return None
        bbox = calculate_red_bbox(red_mask, padding=5)
        if not render_red_pixels_to_image(red_mask, bbox, out_path, canvas_size=(568, 274)):
            return None
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try: os.remove(p)
            except OSError: pass


def process_black(orig_bytes):
    """bbox-crop + AR-fit + 2px normalize -> processed PNG bytes."""
    with tempfile.NamedTemporaryFile(suffix=".png", dir=TMP_DIR, delete=False) as f:
        # normalize_oxford_image opens by path; PIL handles png/jpg via suffix-agnostic open
        f.write(orig_bytes); in_path = f.name
    out_path = in_path + "_out.png"
    try:
        if not normalize_oxford_image(in_path, out_path, target_size=(568, 274),
                                      auto_crop=True, padding=5, target_thickness=2):
            return None
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try: os.remove(p)
            except OSError: pass


def build_features(row):
    """features_data JSON: Total_Score + 60 sub-labels (or null components)."""
    ts = row["total_score"]
    feats = {"Total_Score": int(ts) if ts not in ("", None) else None}
    pres, acc, pos = [], [], []
    complete = True
    for cp, ca, co in ELEM_TRIPLETS:
        if row[cp] == "" or row[ca] == "" or row[co] == "":
            complete = False
            break
        pres.append(int(row[cp])); acc.append(int(row[ca])); pos.append(int(row[co]))
    feats["components"] = ({"presence": pres, "accuracy": acc, "position": pos}
                           if complete else None)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default="/app/templates")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="run detection/dedup/curation + report, but do not write the DB")
    ap.add_argument("--coregister", action="store_true",
                    help="coregister each rendered image to the OCS-Plus reference "
                         "(gated RIGID_BODY pystackreg) before storing")
    args = ap.parse_args()
    os.makedirs(TMP_DIR, exist_ok=True)

    coreg_ref = None
    if args.coregister:
        from coregistration import coregister_processed, load_reference
        coreg_ref = load_reference()
        print("Coregistration: ON (gated RIGID_BODY -> OCS-Plus reference)")

    labels = os.path.join(args.templates, "labels.csv")
    img_dir = os.path.join(args.templates, "img")
    with open(labels, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    # --- pass 1: hash + dedup + corrupt-group detection ---
    by_hash = defaultdict(list)
    for r in rows:
        p = os.path.join(img_dir, r["filename"])
        try:
            r["_sha"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        except OSError:
            r["_sha"] = None
    for r in rows:
        if r["_sha"]:
            by_hash[r["_sha"]].append(r)

    reps, corrupt_drop, dup_drop = [], 0, 0
    for h, grp in by_hash.items():
        nonzero = {r["total_score"] for r in grp if r["total_score"] not in ("", "0")}
        if len(nonzero) > 1:
            corrupt_drop += len(grp)
            continue
        reps.append(grp[0])
        dup_drop += len(grp) - 1

    # --- pass 2: curate + preprocess + insert ---
    db = None if args.dry_run else SessionLocal()
    # Write-protect: never overwrite human-validated rows on reimport (preserve corrections).
    validated_uids = set()
    if db is not None:
        validated_uids = {u for (u,) in db.query(TrainingDataImage.uid).filter(
            TrainingDataImage.validated == True).all() if u}
        if validated_uids:
            print(f"preserving {len(validated_uids)} human-validated rows (skipped on import)")
    stats = defaultdict(int)
    style_by_source = defaultdict(lambda: defaultdict(int))
    mismatches = []
    inserted = 0
    try:
        for r in reps:
            src = r["source"]
            if r["uid"] in validated_uids:        # preserve human-validated rows (write-protect)
                stats["skip_validated"] += 1; continue
            # Unscored placeholders: label_status='zero' marks rows that were never
            # actually scored (all components 0, total_score 0) — they are NOT genuine
            # zero-score drawings (e.g. id 6006 is a near-complete figure). We KEEP the
            # image but import it WITHOUT features (features_data=NULL), so it is excluded
            # from training (queries filter features_data IS NOT NULL) yet still appears
            # under "Only Missing" in the UI for later labelling.
            unscored = (r.get("label_status") or "").strip().lower() == "zero"
            # invalid score
            if not unscored and r["total_score"] not in ("", None):
                try:
                    if int(r["total_score"]) < 0:
                        stats["drop_negative"] += 1; continue
                except ValueError:
                    stats["drop_bad_score"] += 1; continue

            orig = open(os.path.join(img_dir, r["filename"]), "rb").read()
            style, red_c, dark_c = detect_style(orig, src)
            style_by_source[src][style] += 1
            expected = "red" if src in RED_SOURCES else "black"
            if style != "blank" and style != expected:
                mismatches.append((r["uid"], src, f"expected {expected}, detected {style}",
                                   red_c, dark_c))

            if style == "blank":
                stats["skip_blank"] += 1; continue

            if args.dry_run:
                # count what would be inserted, skip actual preprocessing/insert
                inserted += 1; stats[f"src_{src}"] += 1
                if unscored:
                    stats["unscored_no_features"] += 1
                continue

            proc = process_red(orig) if style == "red" else process_black(orig)
            if proc is None:
                stats["skip_no_content"] += 1; continue

            if coreg_ref is not None:
                coreg = coregister_processed(proc, coreg_ref)
                if coreg is not None:
                    proc = coreg
                    stats["coregistered"] += 1
                else:
                    stats["coreg_failed"] += 1

            # unscored placeholders are kept as images but get NO features (NULL)
            feats = None if unscored else build_features(r)
            if unscored:
                stats["unscored_no_features"] += 1
            label_warning = None
            if not unscored and r["total_score_sum"] != "" and r["total_score"] not in ("", None):
                if abs(int(r["total_score"]) - int(r["total_score_sum"])) > 5:
                    label_warning = f"total={r['total_score']} vs sum={r['total_score_sum']}"
                    stats["flagged_mismatch"] += 1

            meta = {
                "uid": r["uid"], "source": src, "orig_filename": r["orig_filename"],
                "style_detected": style, "red_count": red_c, "dark_count": dark_c,
            }
            if label_warning:
                meta["label_warning"] = label_warning

            db.add(TrainingDataImage(
                uid=r["uid"], patient_id=r["patient_id"], task_type=r["cond"],
                source_format=src, original_filename=r["orig_filename"],
                original_file_data=orig, processed_image_data=proc,
                image_hash=r["_sha"], extraction_metadata=json.dumps(meta),
                features_data=(json.dumps(feats) if feats is not None else None),
                session_id=f"unified_{datetime.utcnow():%Y%m%d}",
            ))
            inserted += 1
            stats[f"src_{src}"] += 1
            if inserted % 500 == 0:
                db.commit(); print(f"  ...{inserted} inserted", flush=True)
        if db is not None:
            db.commit()
    finally:
        if db is not None:
            db.close()

    # --- report ---
    print("=" * 60); print("UNIFIED IMPORT REPORT"); print("=" * 60)
    print(f"labels.csv rows:        {len(rows)}")
    print(f"corrupt rows dropped:   {corrupt_drop} (hash groups with conflicting scores)")
    print(f"exact-dup rows dropped: {dup_drop}")
    print(f"representatives:        {len(reps)}")
    print(f"  unscored→no features: {stats['unscored_no_features']} (kept as images, NULL features → 'Only Missing')")
    print(f"  skip blank:           {stats['skip_blank']}")
    print(f"  skip no content:      {stats['skip_no_content']}")
    print(f"  drop negative score:  {stats['drop_negative']}")
    print(f"INSERTED:               {inserted}")
    print("\nper source inserted:")
    for k in sorted(stats):
        if k.startswith("src_"):
            print(f"  {k[4:]:13} {stats[k]}")
    print(f"\nflagged label mismatches (kept): {stats['flagged_mismatch']}")
    print("\ndetected style per source:")
    for src in sorted(style_by_source):
        print(f"  {src:13} {dict(style_by_source[src])}")
    print(f"\nstyle/source mismatches: {len(mismatches)}")
    for m in mismatches[:10]:
        print("   ", m)
    if len(mismatches) > 10:
        print(f"    ... +{len(mismatches) - 10} more")

    # --- automatic alignment QA (skipped on dry-run; needs the inserted rows) ---
    if not args.dry_run:
        try:
            from check_alignment import audit, print_report
            db2 = SessionLocal()
            try:
                summary = audit(db2.query(TrainingDataImage).all())
            finally:
                db2.close()
            print("\n" + "=" * 60)
            print("ALIGNMENT QA  (clip=ink at edge · loose=cropped too small · offc=off-center)")
            print("=" * 60)
            print_report(summary, list_outliers=True)
            print("\nReview the flagged ids in ai_training_data_view.html (search by ID).")
        except Exception as e:
            print(f"\n(alignment QA skipped: {e})")


if __name__ == "__main__":
    main()
