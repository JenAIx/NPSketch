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
    order = 0
    for b in range(N_BANDS):
        pool = by_band[b][:]
        rnd.shuffle(pool)
        pick = pool[:max(0, want.get(b, 0))]
        actual[BAND_LABELS[b]] = len(pick)
        for image_id, ts, comp in pick:
            db.add(EvaluationItem(study_id=study.id, image_id=image_id, band=b, order_idx=order,
                                  gt_total=ts, gt_components=json.dumps(comp)))
            order += 1
    db.commit()
    logger.info(f"Built study {study.id} '{name}': {order} items, per band {actual}, model {model_fn}")
    return {"success": True, "study": _study_dict(db, study), "selected_per_band": actual, "total": order}


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
