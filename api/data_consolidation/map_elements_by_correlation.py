#!/usr/bin/env python3
"""Recover element identity DIRECTLY from the labels (robust).

For each manual-extracted region r, compute over all TELEFRED images the ink fraction
inside r, then correlate it with each label column presence[e]. The region whose
ink-presence best tracks label e IS model element e (Hungarian on the correlation
matrix). This is direct (not the diffuse difference-of-means heatmap) so the matching
is high-confidence. Re-saves element_definitions.json in model ELEM01..20 order.
"""
import io, json, sys
import numpy as np
import cv2
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/app"); sys.path.insert(0, "/app/data_consolidation")
from database import get_db, TrainingDataImage

W, H = 568, 274
DEFS = "/app/data/element_definitions.json"


def region_mask(strokes):
    m = np.zeros((H, W), np.uint8)
    for s in strokes:
        w = int(max(2, s.get("width", 8)))
        for p in s["points"]:
            cv2.circle(m, (int(p[0]), int(p[1])), max(1, w // 2), 255, -1)
    return m > 0


def main():
    defs = json.load(open(DEFS))
    regions = [region_mask(el["strokes"]) for el in defs["elements"]]   # record order r
    rsize = np.array([max(1, r.sum()) for r in regions])

    db = next(get_db())
    rows = (db.query(TrainingDataImage)
            .filter(TrainingDataImage.source_format == "TELEFRED",
                    TrainingDataImage.features_data.isnot(None)).all())
    inkfrac, pres = [], []
    n = 0
    for rrow in rows:
        try:
            comp = json.loads(rrow.features_data).get("components")
        except (ValueError, TypeError):
            comp = None
        if not comp or not comp.get("presence") or not rrow.processed_image_data:
            continue
        g = np.array(Image.open(io.BytesIO(rrow.processed_image_data)).convert("L").resize((W, H)))
        ink = g < 128
        inkfrac.append([float((ink & regions[r]).sum()) / rsize[r] for r in range(20)])
        pres.append([int(x) for x in comp["presence"]])
        n += 1
        if n % 1000 == 0:
            print(f"  {n} images...", flush=True)
    db.close()
    inkfrac = np.array(inkfrac); pres = np.array(pres, float)   # [N,20], [N,20]
    print(f"used {n} images", flush=True)

    # corr[r][e] = Pearson(inkfrac[:,r], presence[:,e])
    corr = np.zeros((20, 20))
    for r in range(20):
        a = inkfrac[:, r]; a = (a - a.mean())
        for e in range(20):
            b = pres[:, e]; b = (b - b.mean())
            denom = np.sqrt((a * a).sum() * (b * b).sum())
            corr[r, e] = (a * b).sum() / denom if denom > 0 else 0.0
    r_idx, e_idx = linear_sum_assignment(-corr)
    rec_to_model = {int(r): int(e) for r, e in zip(r_idx, e_idx)}

    diag = [corr[r, rec_to_model[r]] for r in range(20)]
    identity = sum(1 for r in rec_to_model if rec_to_model[r] == r)
    print(f"\nrecord-order == model-order for {identity}/20")
    print(f"assignment correlation: min {min(diag):.2f} med {np.median(diag):.2f} max {max(diag):.2f}")
    print("mapping (record# -> model ELEM#, correlation):")
    for r in range(20):
        e = rec_to_model[r]
        flag = "  <-- weak" if corr[r, e] < 0.3 else ""
        print(f"  record {r+1:2d} -> ELEM{e+1:02d}   r={corr[r,e]:.2f}{flag}")

    model_to_rec = {e: r for r, e in rec_to_model.items()}
    defs["elements"] = [{"element": e + 1, "strokes": defs["elements"][model_to_rec[e]]["strokes"]}
                        for e in range(20)]
    defs["order"] = "model_label_order (validated by label-correlation over %d images)" % n
    defs["mapping_confidence"] = {f"ELEM{e+1:02d}": round(float(corr[model_to_rec[e], e]), 3) for e in range(20)}
    json.dump(defs, open(DEFS, "w"), indent=2)
    print(f"\nre-saved {DEFS} in model ELEM01..20 order")


if __name__ == "__main__":
    main()
