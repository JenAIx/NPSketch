#!/usr/bin/env python3
"""Apply the gated pystackreg coregistration to a random sample of every charge
(source_format) for human review.

For each source_format, pick N random rows (seeded -> reproducible) and write, per
example, into data/tmp/coreg_review/<SOURCE>/:
  ex_NN_id<ID>_original.png   raw original scan (red ink for TELEFRED, black else)
  ex_NN_id<ID>_overlay.png    reference = blue, coregistered = red, overlap = purple
plus a per-source contact_sheet.png of all overlays, and a summary line.

Content = the stored processed_image_data (already the normalized 568x274, 2px image
for every source). Alignment = coreg_id122.align (bbox prealign -> overlap-gated
RIGID_BODY refine -> 2px renorm).
"""
import io, os, sys, json, random, argparse
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/data_consolidation")

from database import get_db, TrainingDataImage
from coregistration import gray, ink, align, overlap, overlay_blue_red, load_reference, W, H

OUT = "/app/data/tmp/coreg_review"
SOURCES = ["TELEFRED", "OXFORD", "ALGORITHM", "OCS_MACHINE"]


def label(img, txt):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    cv2.rectangle(img, (0, 0), (W, 20), (245, 245, 245), -1)
    cv2.putText(img, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def score_of(row):
    try:
        return json.loads(row.features_data).get("Total_Score", "")
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    ref = load_reference()

    db = next(get_db())
    grand = []
    for sf in SOURCES:
        rows = db.query(TrainingDataImage).filter(TrainingDataImage.source_format == sf).all()
        if not rows:
            print(f"{sf}: no rows"); continue
        pick = random.sample(rows, min(args.n, len(rows)))
        sdir = f"{OUT}/{sf}"
        os.makedirs(sdir, exist_ok=True)
        # clear any previous run for this source
        for f in os.listdir(sdir):
            os.remove(os.path.join(sdir, f))

        overlays, din, dout, n_ref = [], [], [], 0
        for i, r in enumerate(sorted(pick, key=lambda x: x.id)):
            try:
                content = gray(r.processed_image_data)
                aligned, status = align(content, ref)
            except Exception as e:
                print(f"  {sf} id{r.id}: FAILED {e}"); continue
            oin, oout = overlap(content, ref), overlap(aligned, ref)
            din.append(oin); dout.append(oout); n_ref += int(status == "refined")
            sc = score_of(r)

            # raw original scan, native size
            try:
                orig = cv2.cvtColor(np.array(Image.open(io.BytesIO(r.original_file_data)).convert("RGB")), cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{sdir}/ex_{i:02d}_id{r.id}_original.png", orig)
            except Exception:
                pass
            ov = overlay_blue_red(ref, aligned)
            cv2.imwrite(f"{sdir}/ex_{i:02d}_id{r.id}_overlay.png", ov)
            overlays.append(label(ov, f"id{r.id} {r.task_type} sc{sc}  {oin:.2f}->{oout:.2f} [{status}]"))

        if overlays:
            cv2.imwrite(f"{sdir}/contact_sheet.png", np.vstack(overlays))
        msg = (f"{sf:12s} n={len(din):3d}  overlap {np.mean(din):.3f} -> {np.mean(dout):.3f} "
               f"(Δ{np.mean(dout)-np.mean(din):+.3f})  refined {n_ref}/{len(din)}  -> {sdir}")
        print(msg); grand.append(msg)
    db.close()
    print("\n=== summary ===")
    for m in grand:
        print(m)


if __name__ == "__main__":
    main()
