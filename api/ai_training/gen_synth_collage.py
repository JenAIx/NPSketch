#!/usr/bin/env python3
"""Generator v3 — REAL-STROKE COLLAGE: maximally realistic low-score synthetic drawings.

Instead of degrading perfect template strokes (which the diagnostic showed can't close the
realism gap), this recombines ACTUAL patient strokes:

  build-library: coregister each real TELEFRED drawing to the reference, then per element
    crop = element_region ∩ coregistered_real_ink → a real rendition of that element, tagged
    with the drawing's real (accuracy, position) and score band. Cached to data/stroke_library.pkl.
  generate: sample which elements are present from the empirical priors (per band), then paste
    a random REAL crop of each present element (sampled from the target band). The crop's real
    (acc,pos) BECOMES the label → labels stay exact, strokes/quality are real, score emerges.

CLI:
  --build-library [--limit N] [--max-per-elem M]   build + cache the real-stroke library
  --preview N [--scores ...]                       generate collages + contact sheet (no DB)
  --insert N [--score-min A --score-max B --seed S]  insert SYNTHETIC rows (train-only)
  --purge                                          delete SYNTHETIC rows
"""
import os, sys, json, argparse, hashlib, pickle
import numpy as np
import cv2

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/data_consolidation")

from coregistration import load_reference, align, ink, to_png_bytes, renorm, gray, W, H
from gen_synth_image import rasterize_region, load_priors, band_of, BANDS, DEFS, OUT, features_json

LIB_PATH = "/app/data/stroke_library.pkl"
MIN_CROP_INK = 12          # ignore near-empty crops
REGION_DILATE = 6          # px; widen element regions to catch strokes despite imperfect coreg


def element_regions():
    """20 dilated element region masks (bool) from the hand/manual element definitions."""
    defs = json.load(open(DEFS))
    by_e = {int(el["element"]): el.get("strokes", []) for el in defs["elements"]}
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * REGION_DILATE + 1, 2 * REGION_DILATE + 1))
    regions = []
    for e in range(1, 21):
        r = (rasterize_region(by_e.get(e, [])) > 0).astype(np.uint8)
        regions.append(cv2.dilate(r, k) > 0)
    return regions


# --------------------------------------------------------------- build library
def _load_verifier():
    """deployed (latest) component model, used to verify each extracted element is recognizable."""
    import glob, torch
    from ai_training.component_calibration import load_model
    mp = sorted(glob.glob("/app/data/models/model_Components_*.pth"))[-1]
    meta = json.load(open(mp.replace(".pth", "_metadata.json")))
    print(f"verifier model: {os.path.basename(mp)}", flush=True)
    return load_model(mp, meta), meta


def build_library(limit=0, max_per_elem=400, verify=True):
    """Curate a persistent per-element real-stroke library. A crop is kept only if the human
    E1-20 label says the element is present AND (if verify) the component model also reads it as
    present in the coregistered drawing — i.e. the extraction is confirmed correct. Each crop is
    tagged with the real (accuracy, position) and score band, so generation can reuse correct/
    poor, well-placed/misplaced renditions on demand."""
    import torch
    from database import get_db, TrainingDataImage
    from ai_training.preprocessing import preprocess_bytes_for_prediction
    ref = load_reference(); regions = element_regions()
    model = meta = pthr = None
    if verify:
        model, meta = _load_verifier()
        thr = meta.get("thresholds") or [0.5] * 60
        pthr = np.array([thr[e * 3] for e in range(20)], np.float32)   # per-element presence threshold
    db = next(get_db())
    rows = db.query(TrainingDataImage).filter(TrainingDataImage.source_format == "TELEFRED",
                                              TrainingDataImage.features_data.isnot(None)).all()
    db.close()
    rng = np.random.default_rng(0); rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    lib = {e: [] for e in range(20)}
    n = kept = rej_label = rej_verify = rej_ink = 0
    for r in rows:
        try:
            f = json.loads(r.features_data); c = f.get("components")
            if not c:
                continue
            band = band_of(f["Total_Score"])
            aligned, _ = align(gray(r.processed_image_data), ref)     # coregister real ink -> ref grid
            aink = ink(aligned)
            mpres = None
            if verify:
                arr = preprocess_bytes_for_prediction(to_png_bytes(aligned), metadata=meta, debug=False)
                with torch.no_grad():
                    probs = torch.sigmoid(model(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float())).numpy()[0]
                mpres = probs[0::3]                                    # model presence prob per element
        except Exception:
            continue
        for e in range(20):
            if len(lib[e]) >= max_per_elem * len(BANDS):
                continue
            if c["presence"][e] != 1:
                rej_label += 1; continue
            if verify and mpres[e] < pthr[e]:                         # model must confirm the element
                rej_verify += 1; continue
            crop = regions[e] & aink
            if crop.sum() < MIN_CROP_INK:
                rej_ink += 1; continue
            ys, xs = np.where(crop)
            lib[e].append((np.stack([ys, xs], 1).astype(np.int16),
                           int(c["accuracy"][e]), int(c["position"][e]), band))
            kept += 1
        n += 1
        if n % 500 == 0:
            print(f"  processed {n} drawings (kept {kept} crops)...", flush=True)
    pickle.dump(lib, open(LIB_PATH, "wb"))
    sizes = {e: len(v) for e, v in lib.items()}
    print(f"built VERIFIED stroke library from {n} drawings -> {LIB_PATH}", flush=True)
    print(f"  kept {kept} crops | rejected: not-present {rej_label}, model-unconfirmed {rej_verify}, too-little-ink {rej_ink}", flush=True)
    print(f"  crops/element: min {min(sizes.values())} med {int(np.median(list(sizes.values())))} max {max(sizes.values())}", flush=True)
    return lib


def load_library():
    if not os.path.exists(LIB_PATH):
        raise SystemExit(f"{LIB_PATH} missing — run --build-library first")
    return pickle.load(open(LIB_PATH, "rb"))


# --------------------------------------------------------------- generate
def generate_collage(lib, priors, rng, target):
    """Paste real crops of the elements the priors say are present at this band. The crops'
    real (acc,pos) become the labels → exact labels, real strokes, emergent score."""
    b = band_of(target)
    pres = np.array(priors["pres"][b])
    present = np.where(rng.random(20) < pres)[0]
    canvas = np.zeros((H, W), bool)
    vec = np.zeros(60, np.float32)
    for e in present:
        crops = lib[e]
        if not crops:
            continue
        # prefer a crop from the target band (realistic quality for this score); else any
        band_crops = [c for c in crops if c[3] == b] or crops
        coords, acc, pos, _ = band_crops[rng.integers(len(band_crops))]
        canvas[coords[:, 0], coords[:, 1]] = True
        vec[e * 3 + 0] = 1; vec[e * 3 + 1] = acc; vec[e * 3 + 2] = pos
    img = renorm(np.where(canvas, 0, 255).astype(np.uint8))
    return img, vec, int(vec.sum())


# --------------------------------------------------------------- CLI
def do_preview(n, scores):
    os.makedirs(OUT, exist_ok=True)
    lib = load_library(); priors = load_priors()
    rng = np.random.default_rng(11)
    targets = scores if scores else [int(t) for t in np.linspace(4, 34, n)]
    tiles = []
    print(f"{'#':>2} {'target':>6} {'SCORE':>5} {'#present':>8}")
    for i, t in enumerate(targets):
        img, vec, score = generate_collage(lib, priors, rng, t)
        cv2.imwrite(f"{OUT}/collage_{i:02d}_t{t}_s{score}.png", img)
        tiles.append(cv2.putText(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), f"target {t} / score {score}",
                                 (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA))
        print(f"{i:>2} {t:>6} {score:>5} {int(vec[0::3].sum()):>8}", flush=True)
    rowimgs = [np.hstack(tiles[j:j+3]) for j in range(0, len(tiles), 3) if len(tiles[j:j+3]) == 3]
    if rowimgs:
        wmax = max(r.shape[1] for r in rowimgs)
        rowimgs = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rowimgs]
        cv2.imwrite(f"{OUT}/collage_contact.png", np.vstack(rowimgs))
    print(f"\noutputs in {OUT} (collage_*.png + collage_contact.png)", flush=True)


def do_insert(n, smin, smax, seed):
    from database import SessionLocal, TrainingDataImage
    from datetime import datetime
    lib = load_library(); priors = load_priors()
    rng = np.random.default_rng(seed)
    db = SessionLocal(); added = 0
    for i in range(n):
        target = int(rng.integers(smin, smax + 1))
        img, vec, score = generate_collage(lib, priors, rng, target)
        png = to_png_bytes(img)
        db.add(TrainingDataImage(
            uid=f"SYNTHC-{seed}-{i}", patient_id=f"SYNTH_{seed}_{i}", task_type="COPY",
            source_format="SYNTHETIC", original_filename=f"collage_{seed}_{i}.png",
            original_file_data=png, processed_image_data=png,
            image_hash=hashlib.sha256(png + str(i).encode()).hexdigest(),
            extraction_metadata=json.dumps({"synthetic": True, "method": "collage", "seed": seed}),
            features_data=features_json(vec, score),
            session_id=f"synthc_{datetime.utcnow():%Y%m%d}"))
        added += 1
    db.commit()
    total = db.query(TrainingDataImage).filter(TrainingDataImage.source_format == "SYNTHETIC").count()
    db.close()
    print(f"inserted {added} collage rows; total SYNTHETIC now {total}", flush=True)


def do_purge():
    from database import SessionLocal, TrainingDataImage
    from sqlalchemy import or_
    db = SessionLocal()
    nrm = db.query(TrainingDataImage).filter(
        TrainingDataImage.source_format == "SYNTHETIC",
        or_(TrainingDataImage.validated == False, TrainingDataImage.validated.is_(None))
    ).delete(synchronize_session=False)
    db.commit(); db.close()
    print(f"purged {nrm} SYNTHETIC rows (validated rows preserved)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-library", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-per-elem", type=int, default=400)
    ap.add_argument("--no-verify", action="store_true", help="skip the component-model verification gate")
    ap.add_argument("--preview", type=int, metavar="N")
    ap.add_argument("--scores", type=str)
    ap.add_argument("--insert", type=int, metavar="N")
    ap.add_argument("--score-min", type=int, default=0)
    ap.add_argument("--score-max", type=int, default=35)
    ap.add_argument("--seed", type=int, default=401)
    ap.add_argument("--purge", action="store_true")
    args = ap.parse_args()
    if args.build_library:
        build_library(args.limit, args.max_per_elem, verify=not args.no_verify)
    elif args.purge:
        do_purge()
    elif args.insert:
        do_insert(args.insert, args.score_min, args.score_max, args.seed)
    else:
        scores = [int(x) for x in args.scores.split(",")] if args.scores else None
        do_preview(args.preview or 9, scores)


if __name__ == "__main__":
    main()
