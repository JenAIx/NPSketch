#!/usr/bin/env python3
"""Element-grounded synthetic OCS-Plus image generator (low-score focus).

Uses the canonical 20-element geometry hand/manual-derived into
/app/data/element_definitions.json. For each element we recover its TRUE strokes as
  element_ink[e] = rasterize(element e's painted region) ∩ reference figure ink
so stamped elements are the real figure strokes, not random lines.

A drawing is composed by choosing which elements are PRESENT and optionally degrading
each present element's POSITION (shift > 25 px) and/or ACCURACY (elastic tremor). The
60-component label is therefore EXACT by construction (per scoring manual: presence /
accuracy / position, index e*3+{0,1,2}; see docs/SCORING_CRITERIA.md).

CLI:
  --preview N [--scores 5,10,...]   generate, save PNGs + labels + CNN read, contact
                                    sheet -> data/tmp/synth_gen/   (no DB write)
  --insert N --score-min A --score-max B   insert synthetic rows into the DB
                                    (source_format=SYNTHETIC, patient_id=SYNTH_*,
                                    forced into train; picked up by component training)
  --purge                           delete all source_format=='SYNTHETIC' rows
"""
import io, os, sys, json, argparse, hashlib
import numpy as np
import cv2

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/data_consolidation")

from coregistration import load_reference, renorm, to_png_bytes, ink, W, H

DEFS = "/app/data/element_definitions.json"
PRIORS = "/app/data/element_priors.json"
OUT = "/app/data/tmp/synth_gen"
BASELINE = "/app/data/models/model_Components_20260613_005214"
POS_TOL = 25  # px; shift beyond this loses the position point (scoring manual)
BANDS = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 60)]


# --------------------------------------------------------------- element recipe
def rasterize_region(strokes):
    """Replay brush strokes ({width,erase,points}) into a region mask (uint8 0/255)."""
    m = np.zeros((H, W), np.uint8)
    for s in strokes:
        r = max(1, int(round(s.get("width", 8) / 2)))
        val = 0 if s.get("erase") else 255
        pts = s.get("points", [])
        for i, p in enumerate(pts):
            cv2.circle(m, (int(p[0]), int(p[1])), r, val, -1)
            if i > 0:
                cv2.line(m, (int(pts[i-1][0]), int(pts[i-1][1])), (int(p[0]), int(p[1])), val, 2 * r)
    return m


def build_element_ink():
    """element_ink[e] (e=0..19) = bool strokes of element e = its region ∩ reference ink."""
    ref = load_reference()
    ref_ink = ink(ref)
    defs = json.load(open(DEFS))
    by_e = {int(el["element"]): el.get("strokes", []) for el in defs["elements"]}
    masks = []
    for e in range(1, 21):
        region = rasterize_region(by_e.get(e, [])) > 0
        masks.append(region & ref_ink)
    return ref, ref_ink, masks


# --------------------------------------------------------------- degradations
# v2: realistic, LABEL-PRESERVING degradations. Real impaired drawings show tremor,
# incomplete/gapped strokes, pen overshoot, and distorted detail shapes — not a single
# sinusoidal wobble. Variety here reduces synthetic-style overfit. Everything is applied
# per element on its binary mask, then composited; labels stay exact by construction
# (presence=drawn, accuracy=0 iff an accuracy-degradation was applied, position=0 iff
# shifted > POS_TOL). Detail elements (circle/star/cross) degrade by shape, lines by stroke.
DETAIL = {16, 17, 18}   # 0-indexed ELEM17/18/19 = circle / star / cross


def _bbox(m):
    ys, xs = np.where(m > 127)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None


def jitter_affine(m, rng, rot=4.0, smin=0.92, smax=1.08, tmax=7):
    """small label-preserving affine (every element, for natural hand-drawn variation)."""
    bb = _bbox(m)
    if not bb:
        return m
    cx, cy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), rng.uniform(-rot, rot), rng.uniform(smin, smax))
    M[0, 2] += rng.uniform(-tmax, tmax); M[1, 2] += rng.uniform(-tmax, tmax)
    return cv2.warpAffine(m, M, (W, H), borderValue=0)


def tremor_sin(m, rng, amp):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(m, xs + amp * np.sin(ys / 11.0 + rng.uniform(0, 6)),
                     ys + amp * np.sin(xs / 11.0 + rng.uniform(0, 6)), cv2.INTER_NEAREST, borderValue=0)


def tremor_jitter(m, rng, amp):
    """smoothed random displacement field — irregular hand tremor."""
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), 7) * amp * 14
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (H, W)).astype(np.float32), (0, 0), 7) * amp * 14
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(m, xs + dx, ys + dy, cv2.INTER_NEAREST, borderValue=0)


def partial_cut(m, rng):
    """remove a contiguous band — incomplete stroke (line stops short / missing segment)."""
    bb = _bbox(m)
    if not bb:
        return m
    out = m.copy()
    if rng.random() < 0.5:
        cx = int(rng.integers(bb[0], bb[2] + 1)); w = int(rng.integers(10, 28))
        out[:, max(0, cx - w):cx + w] = 0
    else:
        cy = int(rng.integers(bb[1], bb[3] + 1)); h = int(rng.integers(8, 22))
        out[max(0, cy - h):cy + h, :] = 0
    return out if (out > 127).sum() > 0.25 * (m > 127).sum() else m  # don't erase the whole thing


def add_gaps(m, rng, n):
    """small pen-lift gaps along the strokes (kept subtle; does not flip accuracy alone)."""
    ys, xs = np.where(m > 127)
    if len(xs) < 20:
        return m
    out = m.copy()
    for i in rng.choice(len(xs), min(n, len(xs)), replace=False):
        cv2.circle(out, (int(xs[i]), int(ys[i])), int(rng.integers(3, 7)), 0, -1)
    return out


def distort_detail(m, rng):
    """anisotropic scale + rotation about the shape centroid, with an occasional dropped
    sector (open circle / missing star ray / broken cross arm)."""
    bb = _bbox(m)
    if not bb:
        return m
    cx, cy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
    sx, sy = rng.uniform(0.6, 1.4), rng.uniform(0.6, 1.4)
    S = np.float32([[sx, 0, cx * (1 - sx)], [0, sy, cy * (1 - sy)]])
    m = cv2.warpAffine(m, S, (W, H), borderValue=0)
    m = cv2.warpAffine(m, cv2.getRotationMatrix2D((cx, cy), rng.uniform(-25, 25), 1.0), (W, H), borderValue=0)
    if rng.random() < 0.4:
        ys, xs = np.mgrid[0:H, 0:W]
        ang = (np.arctan2(ys - cy, xs - cx) - rng.uniform(0, 2 * np.pi)) % (2 * np.pi)
        m = np.where(ang < rng.uniform(0.6, 1.4), 0, m).astype(np.uint8)
    return m


def shift(m, rng):
    ox = int(rng.integers(-45, 46)); oy = int(rng.integers(-22, 23))
    if abs(ox) + abs(oy) <= POS_TOL:
        ox += POS_TOL + 5
    return cv2.warpAffine(m, np.float32([[1, 0, ox], [0, 1, oy]]), (W, H), borderValue=0)


def degrade_accuracy(m, e, rng, style="full"):
    """apply an accuracy-breaking degradation. style='gentle' = E1 behavior (mild
    sinusoidal tremor only, all element types); 'full' = v2 type-aware (tremor modes /
    partial for lines, shape distortion for details)."""
    if style == "gentle":
        return tremor_sin(m, rng, rng.uniform(2.8, 4.5))
    if e in DETAIL:
        return distort_detail(m, rng)
    mode = rng.choice(["sin", "jitter", "partial"], p=[0.4, 0.35, 0.25])
    if mode == "sin":
        return tremor_sin(m, rng, rng.uniform(3.0, 4.8))
    if mode == "jitter":
        return tremor_jitter(m, rng, rng.uniform(1.4, 2.6))
    return partial_cut(m, rng)


# --------------------------------------------------------------- empirical priors
def build_priors():
    """Per score-band empirical conditionals from REAL TELEFRED labels:
    P(present_e | band), and given present, P(accuracy=1 | e,band), P(position=1 | e,band).
    Real low-score drawings keep certain elements (frame/circle) and drop others — sampling
    from these makes synthetic structurally realistic (vs a uniform random subset)."""
    sys.path.insert(0, "/app")
    from database import get_db, TrainingDataImage
    from ai_training.dataset import components_to_vector
    db = get_db().__next__()
    rows = db.query(TrainingDataImage).filter(
        TrainingDataImage.source_format == "TELEFRED",
        TrainingDataImage.features_data.isnot(None)).all()
    import json as _j
    pres_c = [np.zeros(20) for _ in BANDS]; acc_c = [np.zeros(20) for _ in BANDS]
    pos_c = [np.zeros(20) for _ in BANDS]; n_band = [0] * len(BANDS); npres = [np.zeros(20) for _ in BANDS]
    for r in rows:
        f = _j.loads(r.features_data); c = f.get("components")
        if not c:
            continue
        s = f["Total_Score"]; b = next((i for i, (lo, hi) in enumerate(BANDS) if lo <= s <= hi), None)
        if b is None:
            continue
        pr = np.array(c["presence"]); ac = np.array(c["accuracy"]); po = np.array(c["position"])
        n_band[b] += 1; pres_c[b] += pr; npres[b] += pr
        acc_c[b] += ac * pr; pos_c[b] += po * pr   # accuracy/position only count where present
    db.close()
    pres, acc, pos = [], [], []
    for b in range(len(BANDS)):
        nb = max(1, n_band[b]); npr = npres[b] + 1.0   # Laplace smoothing
        pres.append(((pres_c[b] + 0.5) / (nb + 1)).tolist())
        acc.append(((acc_c[b] + 0.5) / npr).tolist())
        pos.append(((pos_c[b] + 0.5) / npr).tolist())
    out = {"bands": BANDS, "n_per_band": n_band, "pres": pres, "acc": acc, "pos": pos}
    json.dump(out, open(PRIORS, "w"), indent=2)
    print(f"built priors from {sum(n_band)} real rows; n/band {n_band} -> {PRIORS}", flush=True)
    return out


def load_priors():
    if os.path.exists(PRIORS):
        return json.load(open(PRIORS))
    return build_priors()


def band_of(score):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= score <= hi:
            return i
    return len(BANDS) - 1


# --------------------------------------------------------------- generation
def sample_spec(rng, target, priors):
    """Sample present elements + per-element acc/pos from the real per-band priors, rejection-
    sampled to land near `target`. Keeps the realistic structure (which elements survive at a
    given score) instead of a uniform random subset."""
    b = band_of(target)
    pres = np.array(priors["pres"][b]); acc = np.array(priors["acc"][b]); pos = np.array(priors["pos"][b])
    best = None
    for _ in range(400):
        present = np.where(rng.random(20) < pres)[0]
        if len(present) == 0:
            continue
        a = (rng.random(len(present)) < acc[present]).astype(int)
        p = (rng.random(len(present)) < pos[present]).astype(int)
        score = int(len(present) + a.sum() + p.sum())
        cand = (present.tolist(), a, p, score)
        if abs(score - target) <= 2:
            return cand
        if best is None or abs(score - target) < abs(best[3] - target):
            best = cand
    return best


def generate(masks, priors, rng, target, style="full"):
    """style='full' = v2 (per-element jitter + type-aware degradation + gaps);
    'gentle' = E1 (mild sinusoidal tremor + shift only). Both use prior-driven presence."""
    present, acc, pos, score = sample_spec(rng, target, priors)
    canvas = np.zeros((H, W), bool)
    vec = np.zeros(60, np.float32)
    for k, e in enumerate(present):
        m = masks[e].astype(np.uint8) * 255
        if style == "full":
            m = jitter_affine(m, rng)                              # natural variation (label-safe)
        if not acc[k]:
            m = degrade_accuracy(m, e, rng, style)
        elif style == "full" and rng.random() < 0.3:
            m = add_gaps(m, rng, int(rng.integers(1, 3)))          # subtle, keeps accuracy=1
        if not pos[k]:
            m = shift(m, rng)
        canvas |= (m > 127)
        vec[e * 3 + 0] = 1
        vec[e * 3 + 1] = int(acc[k])
        vec[e * 3 + 2] = int(pos[k])
    img = renorm(np.where(canvas, 0, 255).astype(np.uint8))
    return img, vec, int(vec.sum())


# --------------------------------------------------------------- CNN read (eval)
def load_cnn():
    import glob
    from ai_training.component_calibration import load_model
    paths = sorted(glob.glob("/app/data/models/model_Components_*.pth"))
    mp = paths[-1] if paths else BASELINE + ".pth"      # latest (best) component model
    meta = json.load(open(mp.replace(".pth", "_metadata.json")))
    print(f"eval model: {os.path.basename(mp)}", flush=True)
    return load_model(mp, meta), meta


def cnn_read(model, meta, img):
    import torch
    from ai_training.preprocessing import preprocess_bytes_for_prediction
    arr = preprocess_bytes_for_prediction(to_png_bytes(img), metadata=meta, debug=False)
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float())).numpy()[0]
    sc = meta.get("score_calibration") or {}
    cal = float(np.clip(np.dot(probs, sc["weights"]) + sc["bias"], 0, 60)) if sc.get("weights") else None
    thr = np.array(meta.get("thresholds", [0.5] * 60), np.float32)
    return probs, cal, thr


def features_json(vec, score):
    v = vec.astype(int)
    return json.dumps({"Total_Score": score, "components": {
        "presence": v[0::3].tolist(), "accuracy": v[1::3].tolist(), "position": v[2::3].tolist()}})


# --------------------------------------------------------------- CLI actions
def do_preview(n, scores, style="full"):
    os.makedirs(OUT, exist_ok=True)
    _, _, masks = build_element_ink()
    sizes = [int(m.sum()) for m in masks]
    print(f"element ink sizes (px): min {min(sizes)} med {int(np.median(sizes))} max {max(sizes)}  | strokes={style}", flush=True)
    rng = np.random.default_rng(11)
    priors = load_priors()
    targets = scores if scores else [int(t) for t in np.linspace(4, 34, n)]
    model, meta = load_cnn()
    tiles = []
    print(f"\n{'#':>2} {'target':>6} {'TRUE':>5} {'CNN':>5} {'|err|':>6} {'pres✓':>6}")
    for i, t in enumerate(targets):
        img, vec, score = generate(masks, priors, rng, t, style)
        probs, cal, thr = cnn_read(model, meta, img)
        pred = (probs >= thr).astype(int)
        presM = int(np.sum((vec[0::3] == 1) & (pred[0::3] == 1)))
        cv2.imwrite(f"{OUT}/ex_{i:02d}_t{t}_true{score}_cnn{cal:.0f}.png", img)
        tiles.append(cv2.putText(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), f"true {score} / cnn {cal:.0f}",
                                 (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA))
        print(f"{i:>2} {t:>6} {score:>5} {cal:>5.0f} {abs(cal-score):>6.1f} {presM:>3}/{int(vec[0::3].sum())}", flush=True)
    rows = [np.hstack(tiles[j:j+3]) for j in range(0, len(tiles), 3) if len(tiles[j:j+3]) == 3]
    if rows:
        wmax = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
        cv2.imwrite(f"{OUT}/contact_sheet.png", np.vstack(rows))
    print(f"\noutputs in {OUT} (ex_*.png + contact_sheet.png)", flush=True)


def do_insert(n, smin, smax, seed, style="full"):
    from database import SessionLocal, TrainingDataImage
    from datetime import datetime
    _, _, masks = build_element_ink()
    priors = load_priors()
    rng = np.random.default_rng(seed)
    db = SessionLocal()
    added = 0
    for i in range(n):
        target = int(rng.integers(smin, smax + 1))
        img, vec, score = generate(masks, priors, rng, target, style)
        png = to_png_bytes(img)
        db.add(TrainingDataImage(
            uid=f"SYNTH-{seed}-{i}", patient_id=f"SYNTH_{seed}_{i}", task_type="COPY",
            source_format="SYNTHETIC", original_filename=f"synth_{seed}_{i}.png",
            original_file_data=png, processed_image_data=png,
            image_hash=hashlib.sha256(png + str(i).encode()).hexdigest(),
            extraction_metadata=json.dumps({"synthetic": True, "target": target, "seed": seed}),
            features_data=features_json(vec, score),
            session_id=f"synth_{datetime.utcnow():%Y%m%d}"))
        added += 1
    db.commit()
    total = db.query(TrainingDataImage).filter(TrainingDataImage.source_format == "SYNTHETIC").count()
    db.close()
    print(f"inserted {added} synthetic rows (scores {smin}-{smax}); total SYNTHETIC rows now {total}", flush=True)


def do_purge():
    from database import SessionLocal, TrainingDataImage
    from sqlalchemy import or_
    db = SessionLocal()
    # never delete human-validated rows (write-protect)
    nrm = db.query(TrainingDataImage).filter(
        TrainingDataImage.source_format == "SYNTHETIC",
        or_(TrainingDataImage.validated == False, TrainingDataImage.validated.is_(None))
    ).delete(synchronize_session=False)
    db.commit(); db.close()
    print(f"purged {nrm} SYNTHETIC rows (validated rows preserved)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, metavar="N")
    ap.add_argument("--scores", type=str, help="comma list of target scores for preview")
    ap.add_argument("--insert", type=int, metavar="N")
    ap.add_argument("--score-min", type=int, default=0)
    ap.add_argument("--score-max", type=int, default=35)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--purge", action="store_true")
    ap.add_argument("--build-priors", action="store_true", help="(re)compute element_priors.json from real labels")
    ap.add_argument("--strokes", choices=["gentle", "full"], default="full",
                    help="gentle = E1-style mild tremor only; full = v2 type-aware degradations")
    args = ap.parse_args()
    if args.build_priors:
        build_priors()
    elif args.purge:
        do_purge()
    elif args.insert:
        do_insert(args.insert, args.score_min, args.score_max, args.seed, args.strokes)
    else:
        scores = [int(x) for x in args.scores.split(",")] if args.scores else None
        do_preview(args.preview or 9, scores, args.strokes)


if __name__ == "__main__":
    main()
