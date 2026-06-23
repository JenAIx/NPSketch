#!/usr/bin/env python3
"""Tests for recent data-view / training-data backend contracts:
  - /training-data-images filters: only_missing, scored_only, validated, score range, recent_hours
  - per-row computed fields: has_components, validated (synthetic stays unvalidated)
  - /save-drawn-image: stores raw original + normalized; normalized_file passthrough; dup detection
  - current-model marker: set-current / current / is_current  (marker is backed up + restored)
  - predict-single return_normalized

Run inside the container:
  docker exec -e PYTHONPATH=/app npsketch-api python3 /app/test_data_view_api.py
Uses throwaway rows (source_format starts with 'TEST_DV'/'SYNTHETIC' tag), cleaned up at the end.
"""
import sys, json, io, os, urllib.request, urllib.error
from datetime import datetime, timedelta
sys.path.insert(0, "/app")
import numpy as np
from PIL import Image
from database import SessionLocal, TrainingDataImage

BASE = "http://localhost:8000/api"
TAG = "TEST_DV"
PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ✓ " if cond else "  ✗ ") + name)


def http(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}


def png_bytes(seed=0):
    rng = np.random.RandomState(seed)
    arr = np.full((274, 568, 3), 255, np.uint8)
    arr[rng.randint(0, 274, 40), rng.randint(0, 568, 40)] = 0   # a few black pixels (unique hash)
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


def comps(p):
    return {"presence": [p] * 20, "accuracy": [p] * 20, "position": [p] * 20}


def mk_row(db, source, features=None, validated=False, validated_at=None, seed=0):
    row = TrainingDataImage(
        patient_id=TAG, task_type="COPY", source_format=source,
        original_filename="t.png", original_file_data=png_bytes(seed), processed_image_data=png_bytes(seed),
        image_hash=f"{TAG}_{seed}", features_data=(json.dumps(features) if features else None),
        validated=validated, validated_at=validated_at, session_id="test")
    db.add(row); db.commit(); return row.id


def cleanup(db):
    # save-drawn-image rows get a name-derived patient_id, so match the TAG source_format too
    db.query(TrainingDataImage).filter(
        (TrainingDataImage.patient_id == TAG) | (TrainingDataImage.source_format == TAG)
    ).delete(synchronize_session=False)
    db.commit()


# ----------------------------------------------------------------------------
def filter_tests():
    print("FILTERS — /training-data-images")
    db = SessionLocal(); cleanup(db)
    ids = {}
    ids["full"]    = mk_row(db, TAG, {"Total_Score": 55, "components": comps(1)}, seed=1)      # has comps
    ids["scored"]  = mk_row(db, TAG, {"Total_Score": 25}, seed=2)                              # score only
    ids["missing"] = mk_row(db, TAG, None, seed=3)                                             # no features
    ids["synth"]   = mk_row(db, "SYNTHETIC", {"Total_Score": 5, "components": comps(0)}, seed=4)  # synthetic
    ids["recent"]  = mk_row(db, TAG, {"Total_Score": 40, "components": comps(1)},
                            validated=True, validated_at=datetime.utcnow(), seed=5)
    db.close()

    def rows(q):
        _, d = http("GET", f"/training-data-images?limit=20000&{q}")
        return {r["id"]: r for r in d.get("images", [])}

    allr = rows("")
    check("has_components: full/synth/recent yes, scored no",
          allr.get(ids["full"], {}).get("has_components") and allr.get(ids["synth"], {}).get("has_components")
          and allr.get(ids["recent"], {}).get("has_components") and not allr.get(ids["scored"], {}).get("has_components"))
    check("validated field: real-with-features=true, synthetic=false, missing=false",
          allr[ids["full"]]["validated"] and allr[ids["scored"]]["validated"] and allr[ids["recent"]]["validated"]
          and not allr[ids["synth"]]["validated"] and not allr[ids["missing"]]["validated"])

    m = rows("only_missing=true")
    check("only_missing: missing in, others out", ids["missing"] in m and ids["full"] not in m and ids["scored"] not in m)

    s = rows("scored_only=true")
    check("scored_only: scored in; full/missing/synth out",
          ids["scored"] in s and ids["full"] not in s and ids["missing"] not in s and ids["synth"] not in s)

    vt = rows("validated=true")
    check("validated=true: full/scored/recent in; synth/missing out",
          all(ids[k] in vt for k in ("full", "scored", "recent")) and ids["synth"] not in vt and ids["missing"] not in vt)
    vf = rows("validated=false")
    check("validated=false: synth/missing in; full out", ids["synth"] in vf and ids["missing"] in vf and ids["full"] not in vf)

    hi = rows("score_min=50&score_max=60")
    check("score 50-60: full(55) in; scored(25)/synth(5) out", ids["full"] in hi and ids["scored"] not in hi and ids["synth"] not in hi)
    lo = rows("score_min=0&score_max=9")
    check("score 0-9: synth(5) in; full(55) out", ids["synth"] in lo and ids["full"] not in lo)

    rc = rows("recent_hours=24")
    check("recent_hours=24: recent in; full(no validated_at) out", ids["recent"] in rc and ids["full"] not in rc)

    db = SessionLocal(); cleanup(db); db.close()


# ----------------------------------------------------------------------------
def save_drawn_tests():
    print("SAVE-DRAWN — original + normalized + dup")
    import time
    base = int(time.time()) % 90000 + 1000   # unique image bytes per run (avoid cross-run hash clash)
    name = f"{TAG}_save_{int(time.time())}"
    raw = png_bytes(seed=base)
    # multipart by hand
    def post_multipart(fields, files):
        boundary = "----testboundary12345"
        body = b""
        for k, v in fields.items():
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
        for k, (fn, data) in files.items():
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\n"
                     f"Content-Type: image/png\r\n\r\n").encode() + data + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(BASE + "/save-drawn-image", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    st, d = post_multipart(
        {"name": name, "total_score": "30", "components": json.dumps(comps(1)),
         "source_format": TAG, "task_type": "UPLOAD"},
        {"file": ("raw.png", raw)})
    ok = st == 200 and d.get("id")
    check("save-drawn returns id", bool(ok))
    if ok:
        iid = d["id"]
        with urllib.request.urlopen(BASE + f"/training-data-image/{iid}/original", timeout=30) as r:
            orig = r.read()
        with urllib.request.urlopen(BASE + f"/training-data-image/{iid}/processed", timeout=30) as r:
            proc = Image.open(io.BytesIO(r.read()))
        check("original stored == raw (verbatim geometry)", Image.open(io.BytesIO(orig)).size == Image.open(io.BytesIO(raw)).size)
        check("processed normalized to 568x274", proc.size == (568, 274))
        _, feats = http("GET", f"/training-data-image/{iid}/features")
        check("features stored: Total_Score + components + validated",
              feats["features"]["Total_Score"] == 30 and feats["features"].get("components") and feats["validated"])
        # duplicate
        st2, d2 = post_multipart({"name": name + "_dup", "source_format": TAG, "task_type": "UPLOAD"},
                                 {"file": ("raw.png", raw)})
        check("duplicate raw rejected (400)", st2 == 400)
        # normalized_file passthrough: distinct normalized image stored verbatim
        norm = png_bytes(seed=base + 1)
        st3, d3 = post_multipart({"name": name + "_pt", "total_score": "10", "components": json.dumps(comps(0)),
                                  "source_format": TAG, "task_type": "UPLOAD"},
                                 {"file": ("raw2.png", png_bytes(seed=base + 2)), "normalized_file": ("n.png", norm)})
        if st3 == 200:
            with urllib.request.urlopen(BASE + f"/training-data-image/{d3['id']}/processed", timeout=30) as r:
                got = r.read()
            check("normalized_file stored verbatim (passthrough)", got == norm)
        else:
            check("normalized_file passthrough save", False)

    db = SessionLocal(); cleanup(db); db.close()


# ----------------------------------------------------------------------------
def current_model_tests():
    print("CURRENT MODEL — marker round-trip")
    marker = "/app/data/models/current_models.json"
    backup = open(marker).read() if os.path.exists(marker) else None
    try:
        _, ml = http("GET", "/ai-training/models")
        comp_models = [m for m in ml.get("models", []) if (m.get("feature") or "").lower() == "components"]
        if not comp_models:
            print("  ! SKIP: no components model"); return
        target = comp_models[-1]["filename"]   # pick one
        st, d = http("POST", "/ai-training/models/set-current", {"filename": target})
        check("set-current ok + mode components", st == 200 and d.get("mode") == "components")
        _, cur = http("GET", "/ai-training/models/current")
        check("current reflects target", (cur["current"].get("components") or {}).get("filename") == target)
        _, ml2 = http("GET", "/ai-training/models")
        check("list is_current flag set on target",
              any(m["filename"] == target and m["is_current"] for m in ml2["models"]))
        st, _ = http("POST", "/ai-training/models/set-current", {"filename": "does_not_exist.pth"})
        check("set-current unknown file 404", st == 404)
    finally:
        if backup is not None:
            open(marker, "w").write(backup)   # restore the real current marker


# ----------------------------------------------------------------------------
def predict_normalized_tests():
    print("PREDICT — return_normalized")
    db = SessionLocal()
    img = (db.query(TrainingDataImage)
           .filter(TrainingDataImage.source_format == "TELEFRED",
                   TrainingDataImage.processed_image_data.isnot(None)).first())
    db.close()
    import glob
    models = sorted(glob.glob("/app/data/models/model_Components_*.pth"))
    if not img or not models:
        print("  ! SKIP: no TELEFRED image or components model"); return
    model_fn = os.path.basename(models[-1])
    # fetch its processed bytes
    with urllib.request.urlopen(BASE + f"/training-data-image/{img.id}/processed", timeout=30) as r:
        proc = r.read()
    boundary = "----pn12345"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"model_filename\"\r\n\r\n{model_fn}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"return_normalized\"\r\n\r\ntrue\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"i.png\"\r\n"
             f"Content-Type: image/png\r\n\r\n").encode() + proc + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + "/ai-training/models/predict-single", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode())
    ni = d.get("normalized_image")
    check("predict returns normalized_image data URL", bool(ni) and ni.startswith("data:image/png;base64,"))
    if ni:
        import base64
        png = base64.b64decode(ni.split(",", 1)[1])
        check("normalized_image decodes to 568x274 PNG", Image.open(io.BytesIO(png)).size == (568, 274))


if __name__ == "__main__":
    try:
        filter_tests()
        save_drawn_tests()
        current_model_tests()
        predict_normalized_tests()
    finally:
        db = SessionLocal(); cleanup(db); db.close()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL); sys.exit(1)
    print("ALL GREEN ✅")
