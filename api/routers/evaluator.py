"""
Evaluator — inter-rater & model-vs-human reliability study.

Build a frozen set of held-out images across score bands; N human raters score them
blind (via the component scoring UI); a chosen model scores them too (results screen).
Ratings live in the DB. Analysis (Phase 3) compares deviations from ground truth.
"""
import json
import random
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session

from database import (get_db, TrainingDataImage, EvaluationStudy, EvaluationItem,
                      EvaluationRating)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/evaluator", tags=["evaluator"])

N_BANDS = 7
BAND_LABELS = {0: "0-9", 1: "10-19", 2: "20-29", 3: "30-39", 4: "40-49", 5: "50-59", 6: "60"}


def _band(score: int) -> int:
    return min(int(score) // 10, 6)


def rater_labels(n: int):
    return [chr(65 + i) for i in range(max(1, n))]   # A, B, C, ...


def _components_present(comp) -> bool:
    return bool(comp and comp.get("presence") and comp.get("accuracy") and comp.get("position"))


def _current_model_val_ids():
    """(model_filename, set(val_image_ids)) for the current components model, else newest."""
    import glob, os
    from routers.ai_training_models import get_current_model
    fn = get_current_model("components")
    if not fn:
        cands = sorted(glob.glob("/app/data/models/model_Components_*.pth"))
        fn = os.path.basename(cands[-1]) if cands else None
    if not fn:
        return None, set()
    meta_path = f"/app/data/models/{fn.replace('.pth', '')}_metadata.json"
    try:
        meta = json.load(open(meta_path))
        return fn, set(meta.get("val_image_ids") or [])
    except Exception:
        return fn, set()


def _progress(db: Session, study: EvaluationStudy):
    """Per-rater rated-count for a study."""
    total = db.query(EvaluationItem).filter(EvaluationItem.study_id == study.id).count()
    out = {}
    for r in rater_labels(study.n_raters):
        out[r] = db.query(EvaluationRating).filter(
            EvaluationRating.study_id == study.id, EvaluationRating.rater == r).count()
    return {"total_items": total, "per_rater": out}


def _study_dict(db: Session, s: EvaluationStudy):
    return {
        "id": s.id, "name": s.name, "model_filename": s.model_filename,
        "n_raters": s.n_raters, "raters": rater_labels(s.n_raters),
        "per_band_config": json.loads(s.per_band_config) if s.per_band_config else {},
        "status": s.status, "created_at": s.created_at.isoformat() if s.created_at else None,
        "progress": _progress(db, s),
    }


@router.post("/studies")
def build_study(payload: dict = Body(default={}), db: Session = Depends(get_db)):
    """Build a study: sample held-out TELEFRED-with-components per score band.
    Body: {name?, n_raters=2, per_band={band_index: count} (default 20 each), seed=42}."""
    name = (payload or {}).get("name") or f"study_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    n_raters = int((payload or {}).get("n_raters") or 2)
    seed = int((payload or {}).get("seed") or 42)
    per_band_req = (payload or {}).get("per_band") or {}
    # default 20 per band
    want = {b: int(per_band_req.get(str(b), per_band_req.get(b, 20))) for b in range(N_BANDS)}

    model_fn, val_ids = _current_model_val_ids()
    if not val_ids:
        raise HTTPException(status_code=400, detail="No held-out (val) image ids found for the current model")

    # held-out TELEFRED images that carry ground-truth components, binned by score
    by_band = {b: [] for b in range(N_BANDS)}
    rows = db.query(TrainingDataImage).filter(
        TrainingDataImage.source_format == "TELEFRED",
        TrainingDataImage.id.in_(list(val_ids)),
        TrainingDataImage.features_data.isnot(None)).all()
    for img in rows:
        try:
            f = json.loads(img.features_data)
        except Exception:
            continue
        ts = f.get("Total_Score")
        comp = f.get("components")
        if ts is None or not _components_present(comp):
            continue
        by_band[_band(ts)].append((img.id, int(ts), comp))

    rnd = random.Random(seed)
    study = EvaluationStudy(name=name, model_filename=model_fn, n_raters=n_raters,
                            per_band_config=json.dumps(want), status="open")
    db.add(study); db.flush()   # get study.id

    actual = {}
    selected = []
    for b in range(N_BANDS):
        pool = by_band[b][:]
        rnd.shuffle(pool)
        pick = pool[:max(0, want.get(b, 0))]
        actual[BAND_LABELS[b]] = len(pick)
        for image_id, ts, comp in pick:
            selected.append((image_id, b, ts, comp))
    # present in RANDOM order (not grouped by score) so raters aren't biased by ascending scores
    rnd.shuffle(selected)
    for order, (image_id, b, ts, comp) in enumerate(selected):
        db.add(EvaluationItem(study_id=study.id, image_id=image_id, band=b, order_idx=order,
                              gt_total=ts, gt_components=json.dumps(comp)))
    db.commit()
    logger.info(f"Built study {study.id} '{name}': {len(selected)} items, per band {actual}, model {model_fn}")
    return {"success": True, "study": _study_dict(db, study), "selected_per_band": actual, "total": len(selected)}


@router.get("/studies")
def list_studies(db: Session = Depends(get_db)):
    studies = db.query(EvaluationStudy).order_by(EvaluationStudy.created_at.desc()).all()
    return {"studies": [_study_dict(db, s) for s in studies]}


@router.get("/studies/{study_id}")
def get_study(study_id: int, db: Session = Depends(get_db)):
    s = db.query(EvaluationStudy).filter(EvaluationStudy.id == study_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Study not found")
    # per-band counts
    items = db.query(EvaluationItem).filter(EvaluationItem.study_id == study_id).all()
    per_band = {}
    for it in items:
        per_band[BAND_LABELS[it.band]] = per_band.get(BAND_LABELS[it.band], 0) + 1
    d = _study_dict(db, s)
    d["per_band"] = per_band
    return d


@router.delete("/studies/{study_id}")
def delete_study(study_id: int, db: Session = Depends(get_db)):
    from database import EvaluationModelRun
    db.query(EvaluationRating).filter(EvaluationRating.study_id == study_id).delete()
    db.query(EvaluationItem).filter(EvaluationItem.study_id == study_id).delete()
    db.query(EvaluationModelRun).filter(EvaluationModelRun.study_id == study_id).delete()
    db.query(EvaluationStudy).filter(EvaluationStudy.id == study_id).delete()
    db.commit()
    return {"success": True}


@router.get("/studies/{study_id}/queue")
def study_queue(study_id: int, rater: str, db: Session = Depends(get_db)):
    """Ordered items for a rater + that rater's own prior rating (resume/review).
    No ground-truth or model values are leaked to the rater (blind)."""
    s = db.query(EvaluationStudy).filter(EvaluationStudy.id == study_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Study not found")
    if rater not in rater_labels(s.n_raters):
        raise HTTPException(status_code=400, detail=f"Unknown rater '{rater}'")
    items = db.query(EvaluationItem).filter(
        EvaluationItem.study_id == study_id).order_by(EvaluationItem.order_idx).all()
    ratings = {r.image_id: r for r in db.query(EvaluationRating).filter(
        EvaluationRating.study_id == study_id, EvaluationRating.rater == rater).all()}
    out = []
    for it in items:
        r = ratings.get(it.image_id)
        out.append({
            "image_id": it.image_id, "band": BAND_LABELS[it.band], "order_idx": it.order_idx,
            "rated": r is not None,
            "total_score": (r.total_score if r else None),
            "components": (json.loads(r.components) if (r and r.components) else None),
        })
    return {"study_id": study_id, "rater": rater, "items": out,
            "rated": sum(1 for x in out if x["rated"]), "total": len(out)}


@router.post("/studies/{study_id}/rating")
def save_rating(study_id: int, payload: dict = Body(...), db: Session = Depends(get_db)):
    """Upsert a rater's blind score for one study image."""
    s = db.query(EvaluationStudy).filter(EvaluationStudy.id == study_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Study not found")
    rater = payload.get("rater")
    image_id = payload.get("image_id")
    if rater not in rater_labels(s.n_raters) or image_id is None:
        raise HTTPException(status_code=400, detail="rater and image_id required")
    comp = payload.get("components")
    total = payload.get("total")
    if total is None and _components_present(comp):
        total = sum(comp["presence"]) + sum(comp["accuracy"]) + sum(comp["position"])
    row = db.query(EvaluationRating).filter(
        EvaluationRating.study_id == study_id, EvaluationRating.image_id == image_id,
        EvaluationRating.rater == rater).first()
    if not row:
        row = EvaluationRating(study_id=study_id, image_id=image_id, rater=rater)
        db.add(row)
    row.components = json.dumps(comp) if comp else None
    row.total_score = int(total) if total is not None else None
    row.updated_at = datetime.utcnow()
    db.commit()
    return {"success": True, "study_id": study_id, "image_id": image_id, "rater": rater,
            "total_score": row.total_score}


# ============================================================================
# Analysis (Phase 3): model-vs-rater + inter-rater reliability statistics
# ============================================================================

def _vec(comp):
    """components dict -> 60-vector [pres,acc,pos]*20, or None."""
    if not _components_present(comp):
        return None
    out = []
    for e in range(20):
        out += [int(comp["presence"][e]), int(comp["accuracy"][e]), int(comp["position"][e])]
    return out


def _ensure_model_run(db: Session, study_id: int, model_filename: str):
    """Return {image_id: {'total':int,'components':vec60}} for a model over the study's
    images, computing + caching it (EvaluationModelRun) on first request."""
    from database import EvaluationModelRun
    run = db.query(EvaluationModelRun).filter(
        EvaluationModelRun.study_id == study_id,
        EvaluationModelRun.model_filename == model_filename).first()
    if run and run.scores:
        return json.loads(run.scores)

    import os, numpy as np, torch
    from ai_training.component_calibration import load_model
    from ai_training.preprocessing import preprocess_bytes_for_prediction
    mp = f"/app/data/models/{model_filename}"
    meta_path = mp.replace(".pth", "_metadata.json")
    if not os.path.exists(mp) or not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Model or metadata not found")
    meta = json.load(open(meta_path))
    model = load_model(mp, meta)
    thr = np.array(meta.get("thresholds", [0.5] * 60), np.float32)
    sc = meta.get("score_calibration") or {}
    w = np.array(sc.get("weights", [0] * 60), np.float32); b = float(sc.get("bias", 0))

    items = db.query(EvaluationItem).filter(EvaluationItem.study_id == study_id).all()
    ids = [it.image_id for it in items]
    imgs = {im.id: im.processed_image_data for im in
            db.query(TrainingDataImage).filter(TrainingDataImage.id.in_(ids)).all()}
    scores = {}
    for iid in ids:
        data = imgs.get(iid)
        if not data:
            continue
        arr = preprocess_bytes_for_prediction(data, metadata=meta, debug=False)
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float())).numpy()[0]
        hard = (probs >= thr).astype(int)
        total = float(np.clip(probs @ w + b, 0, 60)) if sc.get("weights") else float(hard.sum())
        scores[str(iid)] = {"total": round(total, 1), "components": hard.tolist()}
    db.add(EvaluationModelRun(study_id=study_id, model_filename=model_filename, scores=json.dumps(scores)))
    db.commit()
    return scores


def _mae(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.abs(a - b))) if len(a) else None


def _rmse(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else None


def _bias(pred, ref):
    import numpy as np
    a, b = np.asarray(pred, float), np.asarray(ref, float)
    return float(np.mean(a - b)) if len(a) else None


def _pearson(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _bland_altman(pred, ref):
    import numpy as np
    a, b = np.asarray(pred, float), np.asarray(ref, float)
    if len(a) < 2:
        return None
    diff = a - b
    md, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    return {"mean_diff": round(md, 3), "sd": round(sd, 3),
            "loa_low": round(md - 1.96 * sd, 3), "loa_high": round(md + 1.96 * sd, 3), "n": len(a)}


def _icc21(matrix):
    """ICC(2,1) absolute agreement, two-way random, single measures. matrix: n×k."""
    import numpy as np
    m = np.asarray(matrix, float)
    n, k = m.shape
    if n < 2 or k < 2:
        return None
    grand = m.mean()
    msr = k * ((m.mean(1) - grand) ** 2).sum() / (n - 1)               # rows (targets)
    msc = n * ((m.mean(0) - grand) ** 2).sum() / (k - 1)               # columns (raters)
    sse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    return float((msr - mse) / denom) if denom != 0 else None


def _cohen_kappa_binary(a, b):
    import numpy as np
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    if n == 0:
        return None
    po = float(np.mean(a == b))
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1 else float((po - pe) / (1 - pe))


@router.get("/studies/{study_id}/analysis")
def analysis(study_id: int, model: str = None, db: Session = Depends(get_db)):
    import numpy as np
    s = db.query(EvaluationStudy).filter(EvaluationStudy.id == study_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Study not found")
    if not model:
        from routers.ai_training_models import get_current_model
        model = get_current_model("components") or s.model_filename
    if not model:
        raise HTTPException(status_code=400, detail="No model available")

    items = db.query(EvaluationItem).filter(EvaluationItem.study_id == study_id).order_by(EvaluationItem.order_idx).all()
    gt_total = {it.image_id: it.gt_total for it in items}
    gt_vec = {it.image_id: _vec(json.loads(it.gt_components)) for it in items if it.gt_components}
    model_scores = _ensure_model_run(db, study_id, model)

    raters = rater_labels(s.n_raters)
    rt = {r: {} for r in raters}   # rater -> {image_id: (total, vec60)}
    for row in db.query(EvaluationRating).filter(EvaluationRating.study_id == study_id).all():
        if row.rater in rt and row.components:
            rt[row.rater][row.image_id] = (row.total_score, _vec(json.loads(row.components)))

    # sources to compare vs ground truth: each rater + the model
    def src_totals(getter, ids):
        ref, pred = [], []
        for iid in ids:
            g, p = gt_total.get(iid), getter(iid)
            if g is not None and p is not None:
                ref.append(g); pred.append(p)
        return pred, ref

    sources = {}
    # raters
    for r in raters:
        ids = list(rt[r].keys())
        pred, ref = src_totals(lambda i: rt[r].get(i, (None,))[0], ids)
        sources[f"Rater {r}"] = {"kind": "rater", "n": len(ref),
                                 "total": _total_stats(pred, ref)}
    # model (all GT images)
    mpred, mref = src_totals(lambda i: (model_scores.get(str(i)) or {}).get("total"), list(gt_total.keys()))
    sources["Model"] = {"kind": "model", "model": model, "n": len(mref), "total": _total_stats(mpred, mref)}

    # per-sub-label agreement & kappa vs GT (presence/accuracy/position interleaved → 60 cols)
    def vec_getter(r):
        return lambda i: rt[r].get(i, (None, None))[1]
    per_source_sublabel = {}
    for r in raters:
        per_source_sublabel[f"Rater {r}"] = _sublabel_vs_gt(vec_getter(r), gt_vec, list(rt[r].keys()))
    per_source_sublabel["Model"] = _sublabel_vs_gt(
        lambda i: (model_scores.get(str(i)) or {}).get("components"), gt_vec, list(gt_vec.keys()))

    # inter-rater (pairwise) on total score + sub-labels
    inter = {}
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            ra, rb = raters[i], raters[j]
            common = [iid for iid in rt[ra] if iid in rt[rb]]
            ta = [rt[ra][iid][0] for iid in common]; tb = [rt[rb][iid][0] for iid in common]
            kappas = []
            for iid in common:
                va, vb = rt[ra][iid][1], rt[rb][iid][1]
                if va and vb:
                    kappas.append(np.mean([1 if x == y else 0 for x, y in zip(va, vb)]))
            inter[f"{ra}↔{rb}"] = {
                "n": len(common), "mae": _mae(ta, tb), "icc": _icc21(list(zip(ta, tb))) if len(common) >= 2 else None,
                "bland_altman": _bland_altman(ta, tb), "mean_agreement": float(np.mean(kappas)) if kappas else None}

    # headline: model MAE vs GT compared to raters' and inter-rater
    rater_maes = [sources[f"Rater {r}"]["total"]["mae"] for r in raters if sources[f"Rater {r}"]["total"]["mae"] is not None]
    inter_maes = [v["mae"] for v in inter.values() if v["mae"] is not None]
    model_mae = sources["Model"]["total"]["mae"]
    headline = None
    if model_mae is not None and rater_maes:
        worst_rater = max(rater_maes); mean_rater = sum(rater_maes) / len(rater_maes)
        within = model_mae <= worst_rater + 1e-9
        headline = {
            "model_mae": round(model_mae, 3), "mean_rater_mae": round(mean_rater, 3),
            "worst_rater_mae": round(worst_rater, 3),
            "interrater_mae": round(sum(inter_maes) / len(inter_maes), 3) if inter_maes else None,
            "model_within_human_range": bool(within),
            "verdict": (f"Model Total_Score MAE ({model_mae:.2f}) is "
                        + ("within" if within else "ABOVE")
                        + f" the human rater range (mean {mean_rater:.2f}, worst {worst_rater:.2f})"
                        + (f"; inter-rater MAE {sum(inter_maes)/len(inter_maes):.2f}." if inter_maes else ".")),
        }

    # most-critical sub-labels: lowest mean agreement-vs-GT across raters
    critical = _critical_sublabels(per_source_sublabel, raters)

    # per-image totals (for Bland-Altman / scatter plots on the dashboard)
    points = []
    for it in items:
        iid = it.image_id
        points.append({"image_id": iid, "band": BAND_LABELS[it.band], "gt": gt_total.get(iid),
                       "model": (model_scores.get(str(iid)) or {}).get("total"),
                       "raters": {r: rt[r].get(iid, (None,))[0] for r in raters}})

    return {"study_id": study_id, "model": model, "raters": raters,
            "n_items": len(items), "sources": sources, "inter_rater": inter,
            "per_source_sublabel": per_source_sublabel, "headline": headline,
            "most_critical": critical, "points": points}


def _total_stats(pred, ref):
    return {"mae": _mae(pred, ref), "rmse": _rmse(pred, ref), "bias": _bias(pred, ref),
            "pearson": _pearson(pred, ref), "bland_altman": _bland_altman(pred, ref),
            "icc": _icc21(list(zip(pred, ref))) if len(ref) >= 2 else None}


def _sublabel_vs_gt(getter, gt_vec, ids):
    """Per-column (60) agreement % and Cohen κ of a source vs GT; plus aspect/element rollups."""
    import numpy as np
    A, G = [], []
    for iid in ids:
        v, g = getter(iid), gt_vec.get(iid)
        if v and g:
            A.append(v); G.append(g)
    if not A:
        return {"n": 0, "per_sublabel": None, "macro_agreement": None, "macro_kappa": None, "per_aspect": None}
    A, G = np.array(A, int), np.array(G, int)
    agree, kappa = [], []
    for c in range(60):
        agree.append(float(np.mean(A[:, c] == G[:, c])))
        kappa.append(_cohen_kappa_binary(A[:, c], G[:, c]))
    aspects = {"presence": list(range(0, 60, 3)), "accuracy": list(range(1, 60, 3)), "position": list(range(2, 60, 3))}
    per_aspect = {k: round(float(np.mean([agree[c] for c in cols])), 4) for k, cols in aspects.items()}
    kv = [k for k in kappa if k is not None]
    return {"n": len(A), "per_sublabel_agreement": [round(x, 4) for x in agree],
            "per_sublabel_kappa": [None if k is None else round(k, 4) for k in kappa],
            "macro_agreement": round(float(np.mean(agree)), 4),
            "macro_kappa": round(float(np.mean(kv)), 4) if kv else None,
            "per_aspect": per_aspect}


def _critical_sublabels(per_source_sublabel, raters):
    """Rank the 60 sub-labels by mean rater agreement-vs-GT (lowest = most critical)."""
    import numpy as np
    rater_keys = [f"Rater {r}" for r in raters]
    mats = [per_source_sublabel[k]["per_sublabel_agreement"] for k in rater_keys
            if per_source_sublabel[k].get("per_sublabel_agreement")]
    if not mats:
        return []
    mean_agree = np.mean(np.array(mats, float), axis=0)
    aspects = ["P", "A", "Pos"]
    out = []
    for c in range(60):
        e = c // 3
        out.append({"sublabel": f"E{e+1:02d}-{aspects[c%3]}", "element": e + 1,
                    "aspect": aspects[c % 3], "rater_agreement": round(float(mean_agree[c]), 4)})
    out.sort(key=lambda x: x["rater_agreement"])
    return out[:15]


@router.get("/studies/{study_id}/export.csv")
def export_csv(study_id: int, model: str = None, db: Session = Depends(get_db)):
    from fastapi.responses import PlainTextResponse
    s = db.query(EvaluationStudy).filter(EvaluationStudy.id == study_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Study not found")
    if not model:
        from routers.ai_training_models import get_current_model
        model = get_current_model("components") or s.model_filename
    items = db.query(EvaluationItem).filter(EvaluationItem.study_id == study_id).order_by(EvaluationItem.order_idx).all()
    model_scores = _ensure_model_run(db, study_id, model) if model else {}
    raters = rater_labels(s.n_raters)
    rt = {r: {} for r in raters}
    for row in db.query(EvaluationRating).filter(EvaluationRating.study_id == study_id).all():
        if row.rater in rt:
            rt[row.rater][row.image_id] = row.total_score
    cols = ["image_id", "band", "gt_total"] + [f"rater_{r}_total" for r in raters] + ["model_total"]
    lines = [",".join(cols)]
    for it in items:
        ms = (model_scores.get(str(it.image_id)) or {}).get("total", "")
        vals = [it.image_id, BAND_LABELS[it.band], it.gt_total] + [rt[r].get(it.image_id, "") for r in raters] + [ms]
        lines.append(",".join("" if v is None else str(v) for v in vals))
    return PlainTextResponse("\n".join(lines), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="evaluator_study_{study_id}.csv"'})
