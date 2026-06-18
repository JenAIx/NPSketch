#!/usr/bin/env python3
"""Recover the TRUE element identity for the manual-extracted regions.

The manual record form is numbered in reading order, which need not equal the model's
ELEM01..20 label order. The data-driven heatmap for model-element e (mean ink when
presence[e]=1 minus =0) is derived straight from the labels, so it localizes where
model-element e actually lives. We match each extracted region r to the heatmap e it
best overlaps (Hungarian assignment) -> permutation record_order -> model label order.
Re-saves element_definitions.json in model-label order and reports the mapping.
"""
import json, sys
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/app"); sys.path.insert(0, "/app/data_consolidation")
from ai_training.component_heatmaps import data_driven, refine

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
    regions = [region_mask(el["strokes"]) for el in defs["elements"]]  # record order r
    print("Computing data-driven heatmaps (model ELEM order)...", flush=True)
    heats, *_ = data_driven()                                          # model order e
    heats = [refine(h, gamma=2.2) for h in heats]

    # cost[r][e] = -overlap of region r with heatmap e (mean heat inside region)
    cost = np.zeros((20, 20))
    for r in range(20):
        a = regions[r]
        denom = max(1, a.sum())
        for e in range(20):
            cost[r, e] = -(heats[e][a].sum() / denom)
    r_idx, e_idx = linear_sum_assignment(cost)
    # mapping: record region r_idx[k] IS model element e_idx[k]
    rec_to_model = {int(r): int(e) for r, e in zip(r_idx, e_idx)}

    identity = sum(1 for r in rec_to_model if rec_to_model[r] == r)
    print(f"\nrecord-order == model-order for {identity}/20 elements")
    print("mapping (record# -> model ELEM#, +score):")
    for r in range(20):
        e = rec_to_model[r]
        print(f"  record {r+1:2d} -> ELEM{e+1:02d}   (overlap {-cost[r,e]:.3f})")

    # re-save in MODEL label order: model element (e+1) gets the strokes of the
    # record region assigned to it.
    model_to_rec = {e: r for r, e in rec_to_model.items()}
    out_elements = []
    for e in range(20):
        r = model_to_rec[e]
        out_elements.append({"element": e + 1, "strokes": defs["elements"][r]["strokes"]})
    defs["elements"] = out_elements
    defs["order"] = "model_label_order (validated via data-driven heatmaps)"
    json.dump(defs, open(DEFS, "w"), indent=2)
    print(f"\nre-saved {DEFS} in model ELEM01..20 order")


if __name__ == "__main__":
    main()
