#!/usr/bin/env python3
"""
EXPERIMENT: which method best aligns a drawing onto the OCS-Plus reference figure?

Tries several registration/normalization methods on a sample of TELEFRED images and
measures alignment to the reference (symmetric ink overlap within a tolerance, +
blurred-ink SSIM), split by complete (score>=55) vs partial (score<=35). Writes a few
visual comparison strips to data/tmp/regexp_*.png.

Methods:
  baseline      stored crop-to-ink + AR resize (current)
  moments       affine from image moments (centroid + orientation + scale)
  ecc_euclid    ECC rotation+translation, moment-initialised
  ecc_affine    ECC full affine, moment-initialised
  orb_affine    ORB keypoints + RANSAC partial-affine
  flow          Farneback dense optical flow → nonlinear remap (moment pre-align)
  rect_persp    detect the outer rectangle (4 corners) → perspective to canonical

Run: docker exec -e PYTHONPATH=/app npsketch-api python3 /app/data_consolidation/registration_pipeline.py
"""
import io, json, warnings
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from database import get_db, TrainingDataImage
from ai_training.preprocessing import preprocess_bytes_for_prediction

W, H = 568, 274
warnings.filterwarnings("ignore")


def gray(b):
    return np.array(Image.open(io.BytesIO(b)).convert("L").resize((W, H)), np.uint8)


def ink(g):
    return (g < 128).astype(np.uint8)


def overlap(a, b, tol=5):
    """symmetric fraction of each ink within `tol` px of the other (1=perfect)."""
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    ad, bd = cv2.dilate(a, k), cv2.dilate(b, k)
    return 0.5 * ((a & bd).sum() / a.sum() + (b & ad).sum() / b.sum())


def blur(g):
    return cv2.GaussianBlur((255 - g).astype(np.float32) / 255, (0, 0), 3)


def moment_affine(src_g, dst_g):
    """affine mapping src moments → dst moments (translation+rotation+scale)."""
    def feats(g):
        m = cv2.moments(ink(g).astype(np.float32))
        if m["m00"] == 0:
            return None
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        scale = np.sqrt(m["m00"])
        return np.array([cx, cy]), theta, scale
    fs, fd = feats(src_g), feats(dst_g)
    if fs is None or fd is None:
        return None
    (cs, ts, ss), (cd, td, sd) = fs, fd
    s = sd / ss if ss > 0 else 1.0
    dth = td - ts
    R = np.array([[np.cos(dth), -np.sin(dth)], [np.sin(dth), np.cos(dth)]]) * s
    t = cd - R @ cs
    return np.hstack([R, t.reshape(2, 1)]).astype(np.float32)


def warp_aff(g, M):
    return cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)


def m_moments(d, r):
    M = moment_affine(d, r)
    return warp_aff(d, M) if M is not None else None


def m_ecc(d, r, motion):
    init = moment_affine(d, r)
    warp = init if init is not None else np.eye(2, 3, np.float32)
    if motion == cv2.MOTION_EUCLIDEAN:                 # ECC euclidean needs 2x3 w/o scale skew
        warp = np.eye(2, 3, np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
    try:
        _, warp = cv2.findTransformECC(blur(r), blur(d), warp, motion, crit, None, 5)
    except cv2.error:
        return None
    return cv2.warpAffine(d, warp, (W, H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderValue=255)


def m_orb(d, r):
    orb = cv2.ORB_create(800)
    k1, des1 = orb.detectAndCompute(255 - ink(d) * 255, None)
    k2, des2 = orb.detectAndCompute(255 - ink(r) * 255, None)
    if des1 is None or des2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:60]
    if len(matches) < 6:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    return warp_aff(d, M) if M is not None else None


def m_flow(d, r):
    pre = m_moments(d, r)                                # coarse affine first
    if pre is None:
        pre = d
    flow = cv2.calcOpticalFlowFarneback(pre, r, None, 0.5, 3, 25, 3, 5, 1.2, 0)
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    mapx = (gx + flow[..., 0]).astype(np.float32)
    mapy = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(pre, mapx, mapy, cv2.INTER_LINEAR, borderValue=255)


def _order_quad(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1); diff = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], np.float32)


def m_rect(d, r):
    cnts, _ = cv2.findContours(ink(d), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) != 4 or cv2.contourArea(c) < 0.15 * W * H:
        return None
    m = 6
    dst = np.float32([[m, m], [W - m, m], [W - m, H - m], [m, H - m]])
    M = cv2.getPerspectiveTransform(_order_quad(approx), dst)
    return cv2.warpPerspective(d, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)


METHODS = {
    "baseline":   lambda d, r: d,
    "moments":    m_moments,
    "ecc_euclid": lambda d, r: m_ecc(d, r, cv2.MOTION_EUCLIDEAN),
    "ecc_affine": lambda d, r: m_ecc(d, r, cv2.MOTION_AFFINE),
    "orb_affine": m_orb,
    "flow":       m_flow,
    "rect_persp": m_rect,
}


def main():
    refg = cv2.resize((preprocess_bytes_for_prediction(open("/app/templates/reference_image.png", "rb").read(),
                       metadata={}) * 255).astype(np.uint8), (W, H))
    rink, rblur = ink(refg), blur(refg)

    db = next(get_db())
    rows = [r for r in db.query(TrainingDataImage).filter(
        TrainingDataImage.source_format == "TELEFRED", TrainingDataImage.features_data.isnot(None)).all()]
    db.close()
    def sc(r):
        try: return json.loads(r.features_data).get("Total_Score", 0)
        except: return 0
    complete = [r for r in rows if sc(r) >= 55][:25]
    partial = [r for r in rows if sc(r) <= 35][:15]

    agg = defaultdict(lambda: defaultdict(list))
    fails = defaultdict(lambda: defaultdict(int))
    strips = []
    for grp, sample in [("complete", complete), ("partial", partial)]:
        for r in sample:
            d = gray(r.processed_image_data)
            row_imgs = [refg, d]
            for name, fn in METHODS.items():
                try:
                    a = fn(d, refg)
                except Exception:
                    a = None
                if a is None:
                    fails[grp][name] += 1
                    continue
                agg[grp][name].append(overlap(ink(a), rink))
                agg[grp][name + "__ssim"].append(float(ssim(rblur, blur(a), data_range=1.0)))
                if r in (complete[:1] + partial[:1]) and name != "baseline":
                    row_imgs.append(a)
            if r in (complete[:1] + partial[:1]):
                strips.append(np.hstack([cv2.resize(x, (W // 2, H // 2)) for x in row_imgs]))

    print(f"{'method':12} | {'complete overlap':>16} {'ssim':>6} | {'partial overlap':>15} {'ssim':>6} | fails(c/p)")
    for name in METHODS:
        co = np.mean(agg['complete'][name]) if agg['complete'][name] else float('nan')
        cs = np.mean(agg['complete'][name + '__ssim']) if agg['complete'][name + '__ssim'] else float('nan')
        po = np.mean(agg['partial'][name]) if agg['partial'][name] else float('nan')
        ps = np.mean(agg['partial'][name + '__ssim']) if agg['partial'][name + '__ssim'] else float('nan')
        print(f"{name:12} | {co:16.3f} {cs:6.3f} | {po:15.3f} {ps:6.3f} | {fails['complete'][name]}/{fails['partial'][name]}")

    for i, strip in enumerate(strips):
        cv2.imwrite(f"/app/data/tmp/regexp_{i}.png", strip)
    print(f"\nstrips: data/tmp/regexp_*.png  (cols: reference, drawing, " +
          ", ".join([m for m in METHODS if m != 'baseline']) + ")")
    print(f"sample: {len(complete)} complete (score>=55), {len(partial)} partial (score<=35)")


if __name__ == "__main__":
    main()
