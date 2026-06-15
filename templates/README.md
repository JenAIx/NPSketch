# NPSketch — Unified Training Data Base (`/templates`)

This directory is the **single, unified source of truth** for all NPSketch training
data. All five original deliveries were consolidated into one schema on 2026-06-12
(branch `feature/component-score-model`).

```
templates/
├── img/            all raw original images, renamed to <uid>.<ext>   (gitignored — data)
├── labels.csv      one row per image, unified schema (tracked)
├── README.md       this file
├── reference_image.png
└── old/            the original source folders, untouched (gitignored — archive/provenance)
```

Rebuild everything (`img/` + `labels.csv`) from `old/` with:

```bash
python3 api/data_consolidation/consolidate_templates.py --templates /path/to/templates
```

(host-side, pure stdlib — see `api/data_consolidation/README.md`).

---

## The scoring model (why this unifies cleanly)

Every source scores the **same standardized OCS-Plus Figure-Copy figure**:
**20 elements × 3 binary aspects (Presence / Accuracy / Position) = 60 sub-points,
summing to a TotalScore of 0–60.** Documented in
`old/bsp_ocsplus_202511/readme.txt` (`Sum(ELEM01xxx : ELEM20xxx), Min=0, Max=3*20=60`).

## `labels.csv` schema

One row per image. Columns:

| column | meaning |
|---|---|
| `uid` | globally unique row key = image filename without extension |
| `patient_id` | `<SRC>-<origID>` — groups COPY+RECALL of the same patient (use for patient-level splits) |
| `source` | `TELEFRED` \| `ALGORITHM` \| `OCS_MACHINE` \| `OXFORD` |
| `cond` | `COPY` \| `RECALL` |
| `filename` | file in `img/` |
| `orig_id`, `orig_filename` | provenance (original ID + filename) |
| `total_score` | official TotalScore 0–60 (for OCS_MACHINE: derived = sum of sub-labels) |
| `total_score_sum` | sum of the 60 sub-labels (QA cross-check; empty if no sub-labels) |
| `label_status` | `scored` or `zero` (TotalScore 0 / blank) |
| `ELEM01PRES … ELEM20POS` | 60 binary sub-labels (0/1); **empty** where the source has none (OXFORD) |
| `gender`, `age`, `test_date` | optional metadata (TeleFred only) |

**Aspect order is normalized to `PRES, ACC, POS`** (Algorithm/OCS are native;
TeleFred `Position→POS`, `Accuracy→ACC` are remapped — pure column reassignment,
each is 0/1).

## Source mapping

| source | sub-labels | condition mapping | id → patient_id |
|---|---|---|---|
| TELEFRED (202606 ∪ 20260119) | yes (Presence/Position/Accuracy → ELEMnn) | fc0→COPY, fc1→RECALL | numeric → `TF-<id>` |
| ALGORITHM 20260112 | yes (ELEMnnPRES/ACC/POS native) | CPY→COPY, MEM→RECALL | `PC…/KPC…` → `ALG-<id>` |
| OCS_MACHINE (bsp Machine_rater) | yes (native) | separate COPY/RECALL CSVs | `PC…` → `OCSM-<id>` |
| OXFORD 202512 | **no** (TotalScore only) | COPY/RECALL native | `C…` → `OXF-<id>` |

TeleFred is a **union of both deliveries with 202606 precedence** (deduped by
filename; overlapping images are byte-identical). The OCS-Plus Human ratings
(xlsx, 0–3 per element only), the `.mat` files and `readme.txt` remain in
`old/bsp_ocsplus_202511/` as reference and are **not** in the unified set.

## Consolidation results (2026-06-12)

| source | rows | COPY / RECALL |
|---|---|---|
| TELEFRED | 6392 | 3212 / 3180 |
| ALGORITHM | 777 | 389 / 388 |
| OCS_MACHINE | 12 | 6 / 6 |
| OXFORD | 908 | 455 / 453 |
| **TOTAL** | **8089** | |

- Distinct patients: **4065** (4010 with both COPY+RECALL, 55 single-condition).
- **Reconciliation:** 52 OXFORD label rows skipped (no image on disk); 2 exact
  duplicate OXFORD rows dropped (kept first). Integrity verified: every row has an
  image, every image has exactly one row, all `uid` unique, spot-checked images are
  byte-identical to the originals.
- **Label QA** (`total_score` vs sub-label sum): ALGORITHM 777/777 and OCS_MACHINE
  12/12 exact; TELEFRED 5991/6392 exact (rest mostly ±1 manual rounding at high
  scores), **23 gross mismatches (|Δ|>5)** — likely TotalScore data-entry errors
  (e.g. `TF-2024_12_08-4378-COPY`: official 0 but sub-labels sum to 59).

## ⚠️ Open assumptions (must resolve before pooling sub-labels across sources)

1. **Element correspondence.** This consolidation assumes `ELEM01…20` (and
   TeleFred `…1…20`) are the **same 20 physical elements in the same order across
   all sources**. The OCS-Plus standard (readme) makes this very likely, but it is
   **not pixel-verified**. It is irrelevant for the consolidation itself (each row
   keeps its own labels) but **critical** for the component-score model, which pools
   sub-labels by element index. → Confirm against the reference figure / with the
   data provider before training the component model.

2. **TeleFred score-0 / blank scans.** `label_status=zero` covers **987 TELEFRED
   rows**, which include ~380 blank (zero-ink) scans that the previous DB import
   filtered out. Whether these are *true* zeros (valuable low-end data — exactly the
   range the holistic model failed on) or *“not administered / scan failed”* (noise)
   is unresolved. → Clarify with the data provider; filter via `label_status` at
   import/training time as appropriate.

3. **23 gross label mismatches** (above): for the component model the 60 sub-labels
   are the training target and `total_score` is derived — trust the sub-labels and
   flag/exclude these rows; for holistic-score work, review them.

## Next step (not done here)

Re-import `labels.csv` + `img/` into the DB (`training_data_images`) for training.
The importer reads `img/<filename>` (raw original) and the unified labels; the
existing preprocessing pipeline normalizes (red-pixel extraction for TeleFred/OCS,
bbox-crop for the rest → 568×274, 2 px lines). Provenance lives in `old/`.
