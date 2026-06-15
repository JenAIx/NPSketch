#!/usr/bin/env python3
"""Render StackReg before/after examples to data/tmp/pystackreg/ for visual review.
Each strip: [drawing | StackReg-aligned | reference | aligned(red) over reference]."""
import io, os, json
import numpy as np
import cv2
from PIL import Image
from pystackreg import StackReg
from database import get_db, TrainingDataImage
from ai_training.preprocessing import preprocess_bytes_for_prediction

W, H = 568, 274
OUT = "/app/data/tmp/pystackreg"
os.makedirs(OUT, exist_ok=True)


def gray(b):
    return np.array(Image.open(io.BytesIO(b)).convert("L").resize((W, H)), np.uint8)


def ink(g):
    return (g < 128).astype(np.uint8)


def blur(g):
    return cv2.GaussianBlur((255 - g).astype(np.float32) / 255, (0, 0), 3)


def overlap(a, b, tol=5):
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    return 0.5 * ((a & cv2.dilate(b, k)).sum() / a.sum() + (b & cv2.dilate(a, k)).sum() / b.sum())


def stackreg(d, r, mode):
    sr = StackReg(mode)
    sr.register(blur(r), blur(d))
    aligned = sr.transform((255 - d).astype(float) / 255.0)
    return (255 - np.clip(aligned, 0, 1) * 255).astype(np.uint8)


def label(img_gray, txt):
    bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bgr, (0, 0), (W, 22), (245, 245, 245), -1)
    cv2.putText(bgr, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    return bgr


def overlay_on_ref(aligned, ref):
    bgr = cv2.cvtColor(255 - (255 - ref) // 3, cv2.COLOR_GRAY2BGR)  # faded ref
    am = ink(aligned).astype(bool)
    bgr[am] = (0, 0, 230)  # aligned drawing in red
    cv2.rectangle(bgr, (0, 0), (W, 22), (245, 245, 245), -1)
    cv2.putText(bgr, "bilinear (red) over reference", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    return bgr.astype(np.uint8)


def main():
    refg = cv2.resize((preprocess_bytes_for_prediction(open("/app/templates/reference_image.png", "rb").read(),
                       metadata={}) * 255).astype(np.uint8), (W, H))
    db = next(get_db())
    rows = db.query(TrainingDataImage).filter(
        TrainingDataImage.source_format == "TELEFRED", TrainingDataImage.features_data.isnot(None)).all()
    db.close()
    def sc(r):
        try: return json.loads(r.features_data).get("Total_Score", 0)
        except: return 0
    picks = [r for r in rows if sc(r) >= 55][:5] + [r for r in rows if sc(r) <= 35][:5]

    rink = ink(refg)
    for i, r in enumerate(picks):
        d = gray(r.processed_image_data)
        aff = stackreg(d, refg, StackReg.AFFINE)
        bil = stackreg(d, refg, StackReg.BILINEAR)
        ob, oa, obl = overlap(ink(d), rink), overlap(ink(aff), rink), overlap(ink(bil), rink)
        strip = np.hstack([
            label(d, f"drawing (score {sc(r)})  overlap {ob:.2f}"),
            label(aff, f"stackreg affine  {oa:.2f}"),
            label(bil, f"stackreg bilinear  {obl:.2f}"),
            label(refg, "reference"),
            overlay_on_ref(bil, refg),
        ])
        cv2.imwrite(f"{OUT}/ex_{i:02d}_id{r.id}.png", strip)
        print(f"ex_{i:02d}_id{r.id}: score {sc(r)} | overlap drawing {ob:.2f} → affine {oa:.2f} / bilinear {obl:.2f}", flush=True)
    print(f"\nwrote {len(picks)} strips to {OUT}")


if __name__ == "__main__":
    main()
