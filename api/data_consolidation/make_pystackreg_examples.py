#!/usr/bin/env python3
"""Render StackReg coregistration overlays for visual review.

Reference is used AS-IS (never distorted). Each drawing is aligned to it via
bbox-fill prealign (drawing ink-bbox -> reference ink-bbox) + StackReg affine refine,
gated: the refine is kept only if it doesn't touch the frame border or lose ink,
else we fall back to the prealign (which can't clip by construction).

Outputs to data/tmp/pystackreg/:
  ex_NN_idX.png        [ drawing | aligned | overlay ]
  ex_NN_idX_overlay.png   overlay only: reference BLUE, coregistered drawing RED
  contact_sheet.png    all 10 overlays stacked, labelled
"""
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


def warp_aff(g, M):
    return cv2.warpAffine(g, M.astype(np.float32), (W, H), flags=cv2.INTER_LINEAR, borderValue=255)


def bbox_affine(src_g, dst_g):
    """non-uniform scale + translate mapping src ink-bbox -> dst ink-bbox."""
    def bb(g):
        ys, xs = np.where(ink(g) > 0)
        return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
    s, d = bb(src_g), bb(dst_g)
    if not s or not d:
        return None
    sw, sh = s[2] - s[0] + 1, s[3] - s[1] + 1
    dw, dh = d[2] - d[0] + 1, d[3] - d[1] + 1
    A = np.array([[dw / sw, 0], [0, dh / sh]], float)
    scx, scy = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
    dcx, dcy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
    t = np.array([dcx, dcy]) - A @ np.array([scx, scy])
    return np.hstack([A, t.reshape(2, 1)])


def align(d, ref, mode=StackReg.AFFINE):
    """bbox-fill prealign (drawing bbox -> reference bbox) + StackReg refine.
    Keep the refine unless it loses ink off-frame (a whole line clipped); lines
    sitting AT the border are fine, so we gate on ink retention only."""
    A1 = bbox_affine(d, ref)
    if A1 is None:
        return d
    pre = warp_aff(d, A1)
    sr = StackReg(mode)
    sr.register(blur(ref), blur(pre))
    out = sr.transform((255 - pre).astype(float) / 255.0)
    out = (255 - np.clip(out, 0, 1) * 255).astype(np.uint8)
    return pre if ink(out).sum() < 0.93 * ink(pre).sum() else out


def label(img, txt):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    cv2.rectangle(img, (0, 0), (W, 22), (245, 245, 245), -1)
    cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    return img


def overlay_blue_red(ref, aligned):
    """reference = blue, coregistered drawing = red, overlap = purple, on white."""
    b, r = ink(ref).astype(bool), ink(aligned).astype(bool)
    out = np.full((H, W, 3), 255, np.uint8)           # BGR
    out[b] = (235, 120, 0)                             # blue (reference)
    out[r & ~b] = (0, 0, 235)                          # red (drawing only)
    out[r & b] = (180, 0, 180)                         # purple (overlap)
    return out


def main():
    ref = cv2.resize((preprocess_bytes_for_prediction(open("/app/templates/reference_image.png", "rb").read(),
                      metadata={}) * 255).astype(np.uint8), (W, H))   # AS-IS, never distorted
    rink = ink(ref)
    db = next(get_db())
    rows = db.query(TrainingDataImage).filter(
        TrainingDataImage.source_format == "TELEFRED", TrainingDataImage.features_data.isnot(None)).all()
    db.close()
    def sc(r):
        try: return json.loads(r.features_data).get("Total_Score", 0)
        except: return 0
    picks = [r for r in rows if sc(r) >= 55][:5] + [r for r in rows if sc(r) <= 35][:5]

    overlays = []
    for i, r in enumerate(picks):
        # also dump the raw original scan (e.g. red ink for TELEFRED) at native size
        orig = np.array(Image.open(io.BytesIO(r.original_file_data)).convert("RGB"))
        cv2.imwrite(f"{OUT}/ex_{i:02d}_id{r.id}_original.png", cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
        d = gray(r.processed_image_data)
        al = align(d, ref)
        ob, oa = overlap(ink(d), rink), overlap(ink(al), rink)
        ov = overlay_blue_red(ref, al)
        cv2.imwrite(f"{OUT}/ex_{i:02d}_id{r.id}.png", np.hstack([
            label(d, f"drawing (score {sc(r)})  overlap {ob:.2f}"),
            label(al, f"coregistered  {oa:.2f}"),
            label(ov, "reference=blue  coreg=red  overlap=purple"),
        ]))
        cv2.imwrite(f"{OUT}/ex_{i:02d}_id{r.id}_overlay.png", ov)
        overlays.append(label(ov, f"ex_{i:02d} id{r.id}  score {sc(r)}  overlap {ob:.2f}->{oa:.2f}"))
        print(f"ex_{i:02d}_id{r.id}: score {sc(r)} | overlap {ob:.2f} -> {oa:.2f}", flush=True)

    sheet = np.vstack(overlays)
    cv2.imwrite(f"{OUT}/contact_sheet.png", sheet)
    print(f"\noverlays + contact_sheet.png in {OUT}  (reference=blue, coreg=red)")


if __name__ == "__main__":
    main()
