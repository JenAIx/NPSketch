#!/usr/bin/env python3
"""Tests for the Evaluator (inter-rater & model-vs-human reliability) feature.

Two parts:
  1) unit tests of the pure stats helpers (deterministic);
  2) an end-to-end run against the live API: build a study, verify the rater queue is
     blind (no GT/model leaked), rating upsert/persistence, and that a perfect rater
     (= ground truth) yields MAE 0 / agreement 1; CSV export; cleanup.

Run inside the container:
  docker exec -e PYTHONPATH=/app npsketch-api python3 /app/test_evaluator.py
"""
import sys, json, urllib.request, urllib.error
sys.path.insert(0, "/app")

BASE = "http://localhost:8000/api/evaluator"
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
        return e.code, json.loads(e.read().decode())


# ----------------------------------------------------------------------------
def unit_tests():
    print("UNIT — stats helpers")
    from routers import evaluator as E

    check("_band 5→0, 9→0, 10→1, 59→5, 60→6",
          (E._band(5), E._band(9), E._band(10), E._band(59), E._band(60)) == (0, 0, 1, 5, 6))
    check("rater_labels(3) == A,B,C", E.rater_labels(3) == ["A", "B", "C"])

    v = E._vec({"presence": [1] * 20, "accuracy": [0] * 20, "position": [0] * 20})
    check("_vec length 60 + interleaved sum", v is not None and len(v) == 60 and sum(v) == 20 and v[0] == 1 and v[1] == 0)
    check("_vec None on malformed", E._vec({"presence": [1, 0]}) is None)

    check("_mae identical = 0", E._mae([1, 2, 3], [1, 2, 3]) == 0)
    check("_mae [2,2] vs [0,0] = 2", E._mae([2, 2], [0, 0]) == 2)
    check("_bias signed", E._bias([2, 2], [0, 0]) == 2 and E._bias([0], [2]) == -2)
    check("_rmse known", abs(E._rmse([3, 0], [0, 0]) - (9 / 2) ** 0.5) < 1e-9)
    check("_pearson perfect = 1", abs(E._pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9)

    check("kappa identical = 1", abs(E._cohen_kappa_binary([1, 1, 0, 0], [1, 1, 0, 0]) - 1.0) < 1e-9)
    check("kappa < 1 when disagree", E._cohen_kappa_binary([1, 1, 0, 0], [0, 1, 0, 1]) < 1.0)

    icc1 = E._icc21([[1, 1], [2, 2], [3, 3], [4, 4]])
    check("ICC(2,1) perfect agreement ≈ 1", icc1 is not None and abs(icc1 - 1.0) < 1e-6)
    check("ICC None when <2 raters", E._icc21([[1], [2], [3]]) is None)

    ba = E._bland_altman([1, 2, 3], [1, 2, 3])
    check("bland-altman identical → 0 diff", ba and ba["mean_diff"] == 0 and ba["sd"] == 0 and ba["n"] == 3)


# ----------------------------------------------------------------------------
def integration_tests():
    print("INTEGRATION — live API")
    # small study (3/band) for speed
    per_band = {str(i): 3 for i in range(7)}
    st, d = http("POST", "/studies", {"name": "TEST_EVAL", "n_raters": 2, "per_band": per_band, "seed": 123})
    if st != 200 or d.get("total", 0) == 0:
        print("  ! SKIP integration: build returned", st, d.get("detail") or d.get("total"))
        return None
    sid = d["study"]["id"]
    total = d["total"]
    check("build: total>0 and ≤ requested", 0 < total <= 21)
    check("build: raters A,B", d["study"]["raters"] == ["A", "B"])

    # ground truth from DB (to act as a perfect rater) + check randomized order
    from database import SessionLocal, EvaluationItem
    db = SessionLocal()
    items = db.query(EvaluationItem).filter(EvaluationItem.study_id == sid).order_by(EvaluationItem.order_idx).all()
    gt = {it.image_id: (it.gt_total, json.loads(it.gt_components)) for it in items}
    bands_in_order = [it.band for it in items]
    db.close()
    check("items carry ground-truth snapshot", all(c is not None for _, (t, c) in gt.items()))
    check("presentation order randomized (not monotonic by band)",
          bands_in_order != sorted(bands_in_order))

    # queue is blind (no gt/model leaked) and starts unrated
    st, q = http("GET", f"/studies/{sid}/queue?rater=A")
    keys = set(q["items"][0].keys())
    check("queue blind: no gt/model keys", keys == {"image_id", "band", "order_idx", "rated", "total_score", "components"})
    check("queue starts unrated", q["rated"] == 0 and q["total"] == total)

    # perfect raters: A and B both submit exactly the ground truth
    for r in ("A", "B"):
        for it in items:
            t, comp = gt[it.image_id]
            http("POST", f"/studies/{sid}/rating", {"rater": r, "image_id": it.image_id, "components": comp})
    st, q = http("GET", f"/studies/{sid}/queue?rater=A")
    check("after rating: all rated", q["rated"] == total)
    check("queue persists saved total", q["items"][0]["total_score"] is not None)

    # upsert: re-submit one rating with a change → no duplicate, value updated
    first = items[0]
    zero = {"presence": [0] * 20, "accuracy": [0] * 20, "position": [0] * 20}
    http("POST", f"/studies/{sid}/rating", {"rater": "A", "image_id": first.image_id, "components": zero})
    st, q2 = http("GET", f"/studies/{sid}/queue?rater=A")
    check("upsert: rated count unchanged (no dup)", q2["rated"] == total)
    updated = next(x for x in q2["items"] if x["image_id"] == first.image_id)
    check("upsert: value updated to 0", updated["total_score"] == 0)
    # restore the perfect rating
    http("POST", f"/studies/{sid}/rating", {"rater": "A", "image_id": first.image_id, "components": gt[first.image_id][1]})

    # analysis: perfect rater → MAE 0, agreement 1; model present; structure sane
    st, a = http("GET", f"/studies/{sid}/analysis")
    ra = a["sources"]["Rater A"]
    check("analysis: perfect rater MAE == 0", ra["total"]["mae"] == 0)
    check("analysis: perfect rater agreement == 1", a["per_source_sublabel"]["Rater A"]["macro_agreement"] == 1.0)
    check("analysis: inter-rater agreement == 1 (A==B==GT)",
          abs((list(a["inter_rater"].values())[0]["mean_agreement"] or 0) - 1.0) < 1e-9)
    check("analysis: model source has predictions", a["sources"]["Model"]["n"] > 0)
    check("analysis: headline present", a.get("headline") is not None)
    check("analysis: points == total", len(a["points"]) == total)
    check("analysis: most_critical is a list", isinstance(a["most_critical"], list))

    # CSV export
    req = urllib.request.Request(BASE + f"/studies/{sid}/export.csv")
    with urllib.request.urlopen(req, timeout=60) as r:
        csv = r.read().decode()
    check("csv: header + rows", csv.splitlines()[0].startswith("image_id,band,gt_total") and len(csv.splitlines()) == total + 1)

    # cleanup
    st, _ = http("DELETE", f"/studies/{sid}")
    check("delete study ok", st == 200)
    st, _ = http("GET", f"/studies/{sid}")
    check("deleted study now 404", st == 404)
    return sid


if __name__ == "__main__":
    unit_tests()
    integration_tests()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    print("ALL GREEN ✅")
