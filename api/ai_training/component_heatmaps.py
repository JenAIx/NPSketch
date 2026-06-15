#!/usr/bin/env python3
"""
Generate per-element heatmaps over the OCS-Plus reference figure.

For each of the 20 elements we localize its **presence** sub-label two ways:

  1. data-driven  : mean(ink | presence[e]=1) - mean(ink | presence[e]=0) over all
                    TELEFRED images. Model-free; shows where the element's ink lives
                    in the (aligned 568x274) data.
  2. grad-cam     : Grad-CAM of the component CNN's presence logit (output index e*3)
                    computed on the reference figure. Shows where the trained network
                    *looks* to decide that sub-label (explainable AI).

Outputs to /app/data/visualizations/element_map/ (served at
/api/visualizations/element_map/<file>):
  data_e01.png .. data_e20.png        per-element data-driven overlays
  cam_e01.png  .. cam_e20.png         per-element Grad-CAM overlays
  data_composite.png / cam_composite.png   reference with E01..E20 numbered at centroids
  manifest.json                       counts, centroids, model + timestamp

Run:
  docker exec -e PYTHONPATH=/app npsketch-api python3 /app/ai_training/component_heatmaps.py
"""
import os
import io
import json
import glob
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
import torch

from database import get_db, TrainingDataImage
from ai_training.model import DrawingClassifier
from ai_training.preprocessing import preprocess_bytes_for_prediction

REF_PATH = "/app/templates/reference_image.png"
OUT_DIR = "/app/data/visualizations/element_map"
MODELS_GLOB = "/app/data/models/model_Components_*.pth"
W, H = 568, 274


# --------------------------------------------------------------------------- utils
def load_reference_gray():
    img = Image.open(REF_PATH).convert("L").resize((W, H), Image.LANCZOS)
    return np.array(img).astype(np.uint8)


def refine(heat01, gamma, blur=5):
    """Smooth + gamma-suppress diffuse background so the peak region stands out."""
    h = cv2.GaussianBlur(heat01, (blur, blur), 0) if blur else heat01
    m = h.max()
    if m > 0:
        h = (h / m) ** gamma
    return h.astype(np.float32)


def overlay(ref_gray, heat01, alpha=0.65):
    """Blend a [0,1] heatmap (JET) onto the faded reference figure."""
    bg = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    bg = bg * 0.55 + 255 * 0.45  # fade the figure so colour reads clearly
    heat_u8 = np.clip(heat01 * 255, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET).astype(np.float32)
    a = (heat01[..., None] * alpha)
    out = bg * (1 - a) + heat_color * a
    return out.clip(0, 255).astype(np.uint8)


def peak_centroid(heat01, thr=0.6):
    """Weighted centroid of the strongest region (>= thr*max)."""
    m = heat01.max()
    if m <= 0:
        return (W // 2, H // 2)
    mask = (heat01 >= thr * m).astype(np.float32) * heat01
    s = mask.sum()
    if s <= 0:
        return (W // 2, H // 2)
    Y, X = np.mgrid[0:H, 0:W]
    return (int((X * mask).sum() / s), int((Y * mask).sum() / s))


def composite(ref_gray, centroids):
    bg = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    bg = (bg * 0.6 + 255 * 0.4).astype(np.uint8)
    for e, (cx, cy) in enumerate(centroids):
        cv2.circle(bg, (cx, cy), 12, (0, 0, 200), -1)
        cv2.circle(bg, (cx, cy), 12, (255, 255, 255), 1, cv2.LINE_AA)
        label = str(e + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(bg, label, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return bg


# ------------------------------------------------------------------ data-driven
# Score-stratified difference-of-averages. Comparing presence=1 vs presence=0
# *within the same Total_Score band* removes the completeness confound (absent
# images otherwise tend to be globally sparser low-score drawings), so the
# difference isolates element e's own strokes rather than the central cluster.
SCORE_BINS = [(1, 15), (16, 25), (26, 35), (36, 45), (46, 53), (54, 60)]


def _bin_of(score):
    for i, (lo, hi) in enumerate(SCORE_BINS):
        if lo <= score <= hi:
            return i
    return None


def data_driven():
    db = next(get_db())
    rows = (db.query(TrainingDataImage)
            .filter(TrainingDataImage.source_format == "TELEFRED",
                    TrainingDataImage.features_data.isnot(None))
            .all())
    nb = len(SCORE_BINS)
    sum_p = [[np.zeros((H, W), np.float32) for _ in range(nb)] for _ in range(20)]
    sum_a = [[np.zeros((H, W), np.float32) for _ in range(nb)] for _ in range(20)]
    np_b = [[0] * nb for _ in range(20)]
    na_b = [[0] * nb for _ in range(20)]
    n_p = [0] * 20
    n_a = [0] * 20
    used = 0
    for r in rows:
        try:
            fd = json.loads(r.features_data)
            comp = fd.get("components")
            score = fd.get("Total_Score")
        except (ValueError, TypeError):
            comp = None
        if not comp or not comp.get("presence") or not r.processed_image_data or score is None:
            continue
        b = _bin_of(int(score))
        if b is None:
            continue
        img = Image.open(io.BytesIO(r.processed_image_data)).convert("L").resize((W, H))
        ink = 1.0 - (np.array(img, np.float32) / 255.0)  # black lines -> 1
        pres = comp["presence"]
        for e in range(20):
            if pres[e] == 1:
                sum_p[e][b] += ink; np_b[e][b] += 1; n_p[e] += 1
            else:
                sum_a[e][b] += ink; na_b[e][b] += 1; n_a[e] += 1
        used += 1
        if used % 1000 == 0:
            print(f"  data-driven: {used} images...", flush=True)
    db.close()

    heats = []
    for e in range(20):
        acc = np.zeros((H, W), np.float32)
        wsum = 0.0
        for b in range(nb):
            npb, nab = np_b[e][b], na_b[e][b]
            if npb == 0 or nab == 0:
                continue  # element doesn't vary in this band -> no signal
            mp = sum_p[e][b] / npb
            ma = sum_a[e][b] / nab
            w = (npb * nab) / (npb + nab)  # balanced bands weigh most
            acc += w * (mp - ma)
            wsum += w
        diff = np.clip(acc / wsum, 0, None) if wsum > 0 else acc
        m = diff.max()
        heats.append((diff / m).astype(np.float32) if m > 0 else diff)
    print(f"  data-driven done over {used} TELEFRED images (score-stratified)", flush=True)
    return heats, n_p, n_a, used


# ---------------------------------------------------------------------- grad-cam
def grad_cam():
    model_path = sorted(glob.glob(MODELS_GLOB))[-1]
    meta_path = model_path.replace(".pth", "_metadata.json")
    metadata = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    num_outputs = metadata.get("model", {}).get("output_neurons", 60)

    model = DrawingClassifier(num_outputs=num_outputs, pretrained=False, use_sigmoid=False)
    ck = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck)
    model.eval()

    ref_bytes = open(REF_PATH, "rb").read()
    img_array = preprocess_bytes_for_prediction(ref_bytes, metadata=metadata, debug=False)
    x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()

    acts = {}
    handle = model.backbone.layer4.register_forward_hook(
        lambda m, i, o: acts.__setitem__("a", o))
    out = model(x)                      # [1, 60] logits
    A = acts["a"]                       # [1, 512, h, w]

    heats = []
    for e in range(20):
        idx = e * 3                     # presence logit for element e
        g = torch.autograd.grad(out[0, idx], A, retain_graph=True)[0]
        weights = g.mean(dim=(2, 3), keepdim=True)         # [1,512,1,1]
        cam = torch.relu((weights * A).sum(dim=1)).squeeze(0).detach().numpy()
        cam = cv2.resize(cam, (W, H))
        m = cam.max()
        heats.append((cam / m).astype(np.float32) if m > 0 else cam.astype(np.float32))
    handle.remove()
    print(f"  grad-cam done from {os.path.basename(model_path)}", flush=True)
    return heats, os.path.basename(model_path)


# --------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ref = load_reference_gray()

    print("Computing data-driven heatmaps...", flush=True)
    data_heats, n_p, n_a, used = data_driven()
    print("Computing Grad-CAM heatmaps...", flush=True)
    cam_heats, model_name = grad_cam()

    elements = []
    data_centroids, cam_centroids = [], []
    for e in range(20):
        dh = refine(data_heats[e], gamma=2.2)   # data-driven: strong noise suppression
        ch = refine(cam_heats[e], gamma=1.3)     # grad-cam: light cleanup
        cv2.imwrite(os.path.join(OUT_DIR, f"data_e{e+1:02d}.png"), overlay(ref, dh))
        cv2.imwrite(os.path.join(OUT_DIR, f"cam_e{e+1:02d}.png"), overlay(ref, ch))
        dc = peak_centroid(dh)
        cc = peak_centroid(ch)
        data_centroids.append(dc)
        cam_centroids.append(cc)
        elements.append({
            "element": e + 1,
            "n_present": n_p[e],
            "n_absent": n_a[e],
            "data_centroid": dc,
            "cam_centroid": cc,
        })

    cv2.imwrite(os.path.join(OUT_DIR, "data_composite.png"), composite(ref, data_centroids))
    cv2.imwrite(os.path.join(OUT_DIR, "cam_composite.png"), composite(ref, cam_centroids))

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "TELEFRED",
        "n_images": used,
        "model": model_name,
        "methods": {
            "data": "mean(ink | presence=1) - mean(ink | presence=0) over TELEFRED images",
            "cam": "Grad-CAM of the component CNN presence logit on the reference figure",
        },
        "elements": elements,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote 42 PNGs + manifest.json to {OUT_DIR}")
    print(f"  TELEFRED images used: {used} | model: {model_name}")


if __name__ == "__main__":
    main()
