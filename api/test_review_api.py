#!/usr/bin/env python3
"""API tests for the review/labeling tool: the component endpoints must read and save the
human (real) labels AND keep the model prediction in parallel (never clobbered).

Run inside the container:
  docker exec -e PYTHONPATH=/app npsketch-api python3 /app/test_review_api.py
Uses throwaway rows (source_format='TEST_REVIEW'), cleaned up at the end.
"""
import sys, json, io, urllib.request, urllib.error
sys.path.insert(0, "/app")
import numpy as np
from PIL import Image
from database import SessionLocal, TrainingDataImage

BASE = "http://localhost:8000/api"
TAG = "TEST_REVIEW"


def http(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read().decode())


def png_bytes():
    b = io.BytesIO(); Image.new("RGB", (568, 274), "white").save(b, "PNG"); return b.getvalue()


def vec(p, a, pos):
    return {"presence": p, "accuracy": a, "position": pos}


def mk_row(db, model_pred=None, features=None):
    row = TrainingDataImage(
        uid=None, patient_id="TESTPAT", task_type="COPY", source_format=TAG,
        original_filename="t.png", original_file_data=png_bytes(), processed_image_data=png_bytes(),
        image_hash=None, features_data=(json.dumps(features) if features else None),
        model_prediction=(json.dumps(model_pred) if model_pred else None), session_id="test")
    db.add(row); db.commit(); return row.id


PASS = []; FAIL = []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ok  " if cond else "FAIL  ") + name, flush=True)


def main():
    db = SessionLocal()
    # tidy any leftovers from a previous run
    db.query(TrainingDataImage).filter(TrainingDataImage.source_format == TAG).delete(); db.commit()

    model_pred = {"model": "test", "Total_Score": 42.0, "score_hard": 40,
                  "components": vec([1]*20, [1]*18+[0,0], [1]*15+[0]*5)}
    rid = mk_row(db, model_pred=model_pred, features=None)            # unscored + has model prediction
    rid_missing = mk_row(db, model_pred=model_pred, features=None)    # second unscored, for queue test

    # 1. GET features returns the model prediction in parallel, real empty
    st, d = http("GET", f"/training-data-image/{rid}/features")
    check("GET returns model_prediction", d.get("model_prediction", {}).get("Total_Score") == 42.0)
    check("GET model components intact", d["model_prediction"]["components"]["presence"] == [1]*20)
    check("GET real features empty pre-save", not d.get("has_features"))

    # 2. only_missing queue includes the unscored row, with model score
    st, d = http("GET", f"/training-data/review-queue?source_format={TAG}&only_missing=true")
    item = next((x for x in d["items"] if x["id"] == rid), None)
    check("review-queue(only_missing) lists unscored row", item is not None)
    check("review-queue gives model_score for unscored", item and item["model_score"] == 42.0)

    # 3. SAVE real components (human label) via the existing endpoint
    real = {"Total_Score": 30, "components": vec([1]*15+[0]*5, [1]*10+[0]*10, [1]*12+[0]*8)}
    st, d = http("POST", f"/training-data-image/{rid}/features", real)
    check("POST save succeeds", st == 200 and d.get("success"))

    # 4. GET after save: real saved correctly AND model_prediction UNCHANGED (parallel)
    st, d = http("GET", f"/training-data-image/{rid}/features")
    check("real Total_Score saved", d["features"].get("Total_Score") == 30)
    check("real components round-trip", d["features"]["components"]["presence"] == [1]*15+[0]*5)
    check("model_prediction preserved after human save", d.get("model_prediction", {}).get("Total_Score") == 42.0)
    check("model components preserved after save", d["model_prediction"]["components"]["accuracy"] == [1]*18+[0,0])

    # 5. DB-level: both columns populated in parallel
    db.expire_all()
    row = db.query(TrainingDataImage).filter(TrainingDataImage.id == rid).first()
    check("DB features_data set", row.features_data and json.loads(row.features_data)["Total_Score"] == 30)
    check("DB model_prediction set", row.model_prediction and json.loads(row.model_prediction)["Total_Score"] == 42.0)

    # 6. scored review-queue: saved row appears with correct diff |30-42|=12. The human save
    #    write-protects (validated=True), which the queue excludes by default → include_validated.
    st, d = http("GET", f"/training-data/review-queue?source_format={TAG}&min_diff=5&include_validated=true")
    item = next((x for x in d["items"] if x["id"] == rid), None)
    check("review-queue(min_diff) lists discrepant row", item is not None)
    check("review-queue diff = |real-model|", item and abs(item["diff"] - 12.0) < 1e-6)
    check("saved row no longer in only_missing", all(x["id"] != rid for x in
          http("GET", f"/training-data/review-queue?source_format={TAG}&only_missing=true")[1]["items"]))

    # 7. write-protect: human save marks the row validated
    check("human save sets validated=True", bool(row.validated))
    st, d = http("GET", f"/training-data-image/{rid}/features")
    check("GET exposes validated", d.get("validated") is True)

    # 8. _validated:false saves without locking (programmatic)
    http("POST", f"/training-data-image/{rid_missing}/features",
         {"Total_Score": 5, "components": vec([1]*5+[0]*15, [0]*20, [0]*20), "_validated": False})
    db.expire_all()
    rm = db.query(TrainingDataImage).filter(TrainingDataImage.id == rid_missing).first()
    check("_validated:false does not lock", not rm.validated)

    # 9. synthetic purge preserves validated rows (the write-protect contract)
    from sqlalchemy import or_
    sid = "TESTPURGE"
    db.add(TrainingDataImage(source_format="SYNTHETIC", session_id=sid, patient_id="SYNTH_t",
                             task_type="COPY", validated=False, processed_image_data=png_bytes()))
    db.add(TrainingDataImage(source_format="SYNTHETIC", session_id=sid, patient_id="SYNTH_v",
                             task_type="COPY", validated=True, processed_image_data=png_bytes()))
    db.commit()
    npurged = db.query(TrainingDataImage).filter(
        TrainingDataImage.session_id == sid, TrainingDataImage.source_format == "SYNTHETIC",
        or_(TrainingDataImage.validated == False, TrainingDataImage.validated.is_(None))
    ).delete(synchronize_session=False); db.commit()
    remain = db.query(TrainingDataImage).filter(TrainingDataImage.session_id == sid).all()
    check("purge deletes non-validated synthetic", npurged == 1)
    check("purge PRESERVES validated synthetic", len(remain) == 1 and bool(remain[0].validated))
    db.query(TrainingDataImage).filter(TrainingDataImage.session_id == sid).delete(synchronize_session=False); db.commit()

    # teardown
    db.query(TrainingDataImage).filter(TrainingDataImage.source_format == TAG).delete(); db.commit(); db.close()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed" + (f"  -> {FAIL}" if FAIL else "  ✓ all green"), flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
