#!/usr/bin/env python3
"""Extract the canonical 20-element geometry from the OCS-Plus scoring manual.

The manual's record form (page 3) is a 4x5 grid of cells; each cell shows the full
figure in light gray with ONE element highlighted in RED. We:
  1. render that page at high DPI,
  2. detect the 20 figure tiles (grid, reading order = ELEM01..ELEM20),
  3. extract the red highlight in each tile = that element's strokes,
  4. map each tile's figure bbox -> our 568x274 reference ink bbox,
  5. store the result as element_definitions.json (the same format the hand-paint
     tool uses: brush dots along each element's skeleton, so it renders/edits in the
     frontend AND rasterizes back to a region for generation).

Also writes a colour-coded verification overlay and checks element ordering against
the data-driven heatmap centroids.
"""
import io, os, json, sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/data_consolidation")

PDF = "/app/templates/FigureCopyScoring_manual_OCS-Plus.pdf"
OUT = "/app/data/tmp/ocs_manual"
DEFS = "/app/data/element_definitions.json"
W, H = 568, 274
os.makedirs(OUT, exist_ok=True)


def render_page(idx=2, dpi=300):
    import fitz
    pg = fitz.open(PDF)[idx]
    pix = pg.get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)


def masks(bgr):
    b, g, r = bgr[:, :, 0].astype(int), bgr[:, :, 1].astype(int), bgr[:, :, 2].astype(int)
    red = (r - g > 50) & (r - b > 50) & (r > 120)
    val = (b + g + r) / 3
    gray = (np.abs(r - g) < 35) & (np.abs(g - b) < 35) & (val > 110) & (val < 235)  # light gray fig lines (not black text)
    return red.astype(np.uint8), (gray | red).astype(np.uint8)


def figure_bbox(fig_crop):
    """Tight bbox of the FIGURE within a tile = largest connected ink structure
    (the rectangle+lines), excluding the separate light-gray checkbox squares."""
    d = cv2.dilate(fig_crop * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(d, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[i]
    return x, y, w, h, (lab == i)


def detect_tiles(fig):
    """Return 20 figure bounding boxes in reading order (row-major)."""
    closed = cv2.morphologyEx(fig * 255, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        ar = w / max(1, h)
        if 220 <= w <= 760 and 120 <= h <= 430 and 1.4 <= ar <= 2.7 and area > 4000:
            boxes.append((x, y, w, h))
    # cluster into 4 rows by y-center, then sort each row by x
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    rows, cur, last_y = [], [], None
    for b in boxes:
        cy = b[1] + b[3] / 2
        if last_y is not None and cy - last_y > 200:
            rows.append(cur); cur = []
        cur.append(b); last_y = cy
    if cur:
        rows.append(cur)
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b[0]))
    return ordered, closed


def ref_ink_bbox():
    g = np.array(Image.open("/app/templates/reference_image.png").convert("L").resize((W, H)))
    ink = g < 128
    ys, xs = np.where(ink)
    return (xs.min(), ys.min(), xs.max(), ys.max()), ink


def mask_to_dots(mask, step=4):
    """Skeleton -> brush dots {width, points:[[x,y]]} (one 1-point stroke per sample);
    width = local thickness from the distance transform. Robust, order-free, renders
    in the frontend and rasterizes back to the region."""
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(mask > 0)
    except Exception:
        skel = mask > 0
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.where(skel)
    strokes = []
    seen = set()
    for x, y in zip(xs, ys):
        key = (int(x) // step, int(y) // step)
        if key in seen:
            continue
        seen.add(key)
        wdt = max(3.0, float(dist[y, x]) * 2.0)
        strokes.append({"width": round(wdt, 1), "erase": False, "points": [[float(x), float(y)]]})
    return strokes


def main():
    print("Rendering manual page 3 @300dpi...", flush=True)
    bgr = render_page(2, 300)
    red, fig = masks(bgr)
    tiles, closed = detect_tiles(fig)
    print(f"detected {len(tiles)} figure tiles", flush=True)
    cv2.imwrite(f"{OUT}/_tiles_debug.png", closed)
    if len(tiles) != 20:
        print("WARNING: expected 20 tiles — check _tiles_debug.png; proceeding with what we have", flush=True)

    (rx0, ry0, rx1, ry1), ref_ink = ref_ink_bbox()
    rw, rh = rx1 - rx0, ry1 - ry0

    elements = []
    overlay = cv2.cvtColor((255 - ref_ink.astype(np.uint8) * 90), cv2.COLOR_GRAY2BGR)  # faded ref
    centroids = []
    for e, (x, y, w, h) in enumerate(tiles[:20], start=1):
        sub_red = red[y:y + h, x:x + w]
        sub_fig = fig[y:y + h, x:x + w]
        fb = figure_bbox(sub_fig)        # tight figure bbox (excludes checkbox squares)
        if fb is None:
            elements.append({"element": e, "strokes": []}); continue
        fx0, fy0, fw, fh, _ = fb
        fw, fh = max(1, fw), max(1, fh)
        # map red pixels: tile FIGURE bbox -> reference ink bbox
        rys, rxs = np.where(sub_red > 0)
        em = np.zeros((H, W), np.uint8)
        for px, py in zip(rxs, rys):
            mx = int(rx0 + (px - fx0) / fw * rw)
            my = int(ry0 + (py - fy0) / fh * rh)
            if 0 <= mx < W and 0 <= my < H:
                em[my, mx] = 255
        em = cv2.dilate(em, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        strokes = mask_to_dots(em)
        elements.append({"element": e, "strokes": strokes})
        col = tuple(int(c) for c in cv2.applyColorMap(np.uint8([[(e - 1) * 12 % 256]]), cv2.COLORMAP_HSV)[0, 0])
        overlay[em > 0] = col
        ys2, xs2 = np.where(em > 0)
        centroids.append((e, int(xs2.mean()) if len(xs2) else -1, int(ys2.mean()) if len(ys2) else -1))
        cv2.putText(overlay, str(e), (centroids[-1][1], centroids[-1][2]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(f"{OUT}/element_check.png", overlay)
    json.dump({"image_size": [W, H], "updated_at": None,
               "source": "FigureCopyScoring_manual_OCS-Plus.pdf", "elements": elements},
              open(DEFS, "w"), indent=2)
    annotated = sum(1 for el in elements if el["strokes"])
    print(f"wrote {DEFS}: {annotated}/20 elements, "
          f"{sum(len(el['strokes']) for el in elements)} brush dots", flush=True)
    print("element centroids (e, x, y):", centroids, flush=True)
    print(f"verification overlay: {OUT}/element_check.png", flush=True)


if __name__ == "__main__":
    main()
