#!/usr/bin/env python3
"""Shared coregistration core: align a normalized 568x274 drawing to the OCS-Plus
reference figure.

Pipeline: anisotropic bbox prealign (drawing ink-bbox -> reference ink-bbox, no
shear) -> overlap-gated RIGID_BODY StackReg refine -> margin + 2px line-renormalize.

Two deliberate constraints, both learned from id122:

* RIGID_BODY only (rotation + translation, scale == 1). The bbox prealign already
  matches scale; any mode with scale < 1 bilinearly resamples the 2px lines,
  spatially spreading thin near-vertical edges (the rectangle's right side) until
  they fragment and vanish — and a higher overlap score does NOT mean the line
  survived, so overlap can't police scaling.

* Overlap-gate. StackReg readily invents a spurious 1-2 degrees rotation that
  *lowers* the true ink-overlap (it over-fits internal structure). Keep the refine
  only when symmetric ink-overlap actually improves vs the prealign.

This module is import-safe (no side effects, no circular imports); CLI/demo code
lives in coreg_id122.py and coreg_batch.py.
"""
import io
import numpy as np
import cv2
from PIL import Image
from pystackreg import StackReg

from ai_training.preprocessing import preprocess_bytes_for_prediction
from line_normalizer import normalize_line_thickness

W, H = 568, 274
REFERENCE_PATH = "/app/templates/reference_image.png"


def gray(b):
    """PNG/JPEG bytes -> 568x274 grayscale uint8."""
    return np.array(Image.open(io.BytesIO(b)).convert("L").resize((W, H)), np.uint8)


def to_png_bytes(g):
    """grayscale uint8 -> RGB PNG bytes (matches the stored processed_image_data)."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    return buf.tobytes() if ok else None


def ink(g):
    return (g < 128).astype(np.uint8)


def blur(g, sigma=4):
    return cv2.GaussianBlur((255 - g).astype(np.float32) / 255, (0, 0), sigma)


def warp_aff(g, M):
    return cv2.warpAffine(g, M.astype(np.float32), (W, H), flags=cv2.INTER_LINEAR, borderValue=255)


def bbox(g):
    ys, xs = np.where(ink(g) > 0)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None


def bbox_affine(src_g, dst_g):
    """anisotropic scale + translate mapping src ink-bbox -> dst ink-bbox (no shear)."""
    s, d = bbox(src_g), bbox(dst_g)
    if not s or not d:
        return None
    sw, sh = s[2] - s[0] + 1, s[3] - s[1] + 1
    dw, dh = d[2] - d[0] + 1, d[3] - d[1] + 1
    A = np.array([[dw / sw, 0], [0, dh / sh]], float)
    t = np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2]) - A @ np.array([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2])
    return np.hstack([A, t.reshape(2, 1)])


def fit_margin(g, m=12):
    """uniformly shrink (if needed) AND recenter so ink keeps >= m px margin on all
    sides. Always recenters: a wide figure that already fits can still sit flush
    against an edge (StackReg can shove it sideways), clipping it."""
    bb = bbox(g)
    if not bb:
        return g
    x0, y0, x1, y1 = bb
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    s = min(1.0, (W - 2 * m) / bw, (H - 2 * m) / bh)
    if s > 0.999 and min(x0, W - 1 - x1, y0, H - 1 - y1) >= m:
        return g
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    M = np.array([[s, 0, W / 2.0 - s * cx], [0, s, H / 2.0 - s * cy]], np.float32)
    return cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)


def renorm(g):
    """re-binarize @175 + restore 2px line thickness (matches the import pipeline;
    repairs any line-thinning from bilinear warping)."""
    b = np.where(g < 175, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(normalize_line_thickness(rgb, target_thickness=2), cv2.COLOR_RGB2GRAY)


def overlap(a, b, tol=5):
    """symmetric ink overlap within tol px (dilate-based)."""
    a, b = ink(a), ink(b)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    return 0.5 * ((a & cv2.dilate(b, k)).sum() / a.sum() + (b & cv2.dilate(a, k)).sum() / b.sum())


def align(content, ref):
    """bbox prealign -> overlap-gated RIGID_BODY refine -> margin + 2px renorm.
    Returns (aligned_gray_uint8, status)."""
    A1 = bbox_affine(content, ref)
    if A1 is None:
        return renorm(content), "no-bbox"
    pre = warp_aff(content, A1)
    pre_out = renorm(fit_margin(pre))
    sr = StackReg(StackReg.RIGID_BODY)
    sr.register(blur(ref), blur(pre))
    out = sr.transform((255 - pre).astype(float) / 255.0)
    out = (255 - np.clip(out, 0, 1) * 255).astype(np.uint8)
    if ink(out).sum() < 0.93 * ink(pre).sum():
        return pre_out, "prealign (ink-gate)"
    ref_out = renorm(fit_margin(out))
    if overlap(ref_out, ref) > overlap(pre_out, ref):
        return ref_out, "refined"
    return pre_out, "prealign (overlap-gate)"


def overlay_blue_red(ref, aligned):
    """reference = blue, coregistered = red, overlap = purple, on white (BGR)."""
    b, r = ink(ref).astype(bool), ink(aligned).astype(bool)
    out = np.full((H, W, 3), 255, np.uint8)
    out[b] = (235, 120, 0)
    out[r & ~b] = (0, 0, 235)
    out[r & b] = (180, 0, 180)
    return out


def load_reference(path=REFERENCE_PATH):
    """the OCS-Plus reference figure, normalized to 568x274 grayscale uint8 (AS-IS)."""
    return cv2.resize((preprocess_bytes_for_prediction(open(path, "rb").read(), metadata={}) * 255).astype(np.uint8), (W, H))


def coregister_processed(proc_bytes, ref):
    """processed_image_data PNG bytes -> coregistered PNG bytes (or None on failure)."""
    try:
        aligned, _ = align(gray(proc_bytes), ref)
        return to_png_bytes(aligned)
    except Exception:
        return None
