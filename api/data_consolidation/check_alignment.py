#!/usr/bin/env python3
"""
Audit the alignment/framing of every processed_image_data in the DB.

For each normalized 568×274 image we compute the ink bounding box and report, per
source_format: the margins (left/right/top/bottom px), how tightly the figure fills
the frame, and the centering offsets. Flags outliers:
  - clipped:     any margin < MIN_MARGIN (figure touches the edge → possibly cut off)
  - loose:       min margin > MAX_MARGIN (figure too small / not tightly cropped)
  - off-center:  large left/right or top/bottom asymmetry
  - blank:       (almost) no ink

Run:
  docker exec -e PYTHONPATH=/app npsketch-api python3 /app/data_consolidation/check_alignment.py
  docker exec -e PYTHONPATH=/app npsketch-api python3 /app/data_consolidation/check_alignment.py --list-outliers
"""
import io, sys, argparse
from collections import defaultdict
import numpy as np
from PIL import Image
from database import get_db, TrainingDataImage

W, H = 568, 274
INK = 200            # gray < INK = ink
MIN_MARGIN = 2       # px; below → clipped at edge
MAX_MARGIN = 30      # px; min-margin above → loosely cropped
OFF_X = 45           # |left-right| above → off-center horizontally
OFF_Y = 30           # |top-bottom| above → off-center vertically


def bbox_of(img_bytes):
    g = np.array(Image.open(io.BytesIO(img_bytes)).convert("L").resize((W, H)))
    ys, xs = np.where(g < INK)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), len(xs)


def pct(a):
    a = np.array(a)
    return f"min={a.min():.0f} p10={np.percentile(a,10):.0f} med={np.median(a):.0f} p90={np.percentile(a,90):.0f} max={a.max():.0f}"


def audit(rows):
    """Compute per-source alignment stats + flagged id lists for the given DB rows."""
    by_src = defaultdict(lambda: {"L": [], "R": [], "T": [], "B": [], "fill": [],
                                  "clipped": [], "loose": [], "offc": [], "blank": 0, "n": 0})
    for r in rows:
        if not r.processed_image_data:
            continue
        s = by_src[r.source_format]; s["n"] += 1
        bb = bbox_of(r.processed_image_data)
        if bb is None:
            s["blank"] += 1; continue
        x0, y0, x1, y1, ink = bb
        L, R, T, B = x0, W - 1 - x1, y0, H - 1 - y1
        fill = ((x1 - x0 + 1) * (y1 - y0 + 1)) / (W * H)
        s["L"].append(L); s["R"].append(R); s["T"].append(T); s["B"].append(B); s["fill"].append(fill)
        if min(L, R, T, B) < MIN_MARGIN:
            s["clipped"].append(r.id)
        if min(L, R, T, B) > MAX_MARGIN:
            s["loose"].append(r.id)
        if abs(L - R) > OFF_X or abs(T - B) > OFF_Y:
            s["offc"].append(r.id)
    return by_src


def print_report(by_src, list_outliers=False):
    print(f"{'source':12} {'n':>5} {'blank':>5} {'clip':>5} {'loose':>5} {'offc':>5}   margins / fill")
    for src in sorted(by_src):
        s = by_src[src]
        if not s["L"]:
            print(f"{src:12} {s['n']:>5} {s['blank']:>5}  (all blank)"); continue
        print(f"{src:12} {s['n']:>5} {s['blank']:>5} {len(s['clipped']):>5} {len(s['loose']):>5} {len(s['offc']):>5}")
        print(f"             L[{pct(s['L'])}]")
        print(f"             R[{pct(s['R'])}]")
        print(f"             T[{pct(s['T'])}]")
        print(f"             B[{pct(s['B'])}]")
        print(f"             fill[min={min(s['fill']):.2f} med={np.median(s['fill']):.2f} max={max(s['fill']):.2f}]")
        if list_outliers:
            print(f"             clipped ids: {s['clipped'][:15]}")
            print(f"             loose ids:   {s['loose'][:15]}")
            print(f"             off-center:  {s['offc'][:15]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-outliers", action="store_true")
    args = ap.parse_args()
    db = next(get_db())
    rows = db.query(TrainingDataImage).all()
    db.close()
    print_report(audit(rows), list_outliers=args.list_outliers)


if __name__ == "__main__":
    main()
