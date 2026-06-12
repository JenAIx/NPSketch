# Data Consolidation

Builds the unified training-data base in `/templates` from the original per-source
deliveries. See `templates/README.md` for the resulting schema and results.

## `consolidate_templates.py`

Reads every source from `templates/old/`, maps it onto one schema (20 elements × 3
binary aspects PRES/ACC/POS = 60 sub-labels, plus TotalScore), copies the **raw
original** images to `templates/img/<uid>.<ext>`, and writes `templates/labels.csv`.

- Pure Python **stdlib** (csv, os, shutil) — no image processing, no DB, no deps.
- Runs on the **host** (`/templates` is host-writable; the container mount is
  read-only).
- **Idempotent**: rebuilds `img/` and `labels.csv` from `old/` on every run.
- Requires the source folders to already be under `templates/old/`.

```bash
# full run
python3 api/data_consolidation/consolidate_templates.py --templates /home/ste/NPSketch/templates

# build labels.csv + report only, skip copying images
python3 api/data_consolidation/consolidate_templates.py --templates /home/ste/NPSketch/templates --dry-run
```

The run prints a reconciliation report: per-source row counts, label QA
(`total_score` vs sub-label sum, with gross-mismatch list), condition distribution,
skipped label-without-image rows, dropped duplicate rows, and patient count.

## Source handling

| source folder (under `old/`) | delimiter | sub-labels | notes |
|---|---|---|---|
| `training_data_telefred_202606` + `…_20260119` | `;` | Presence/Position/Accuracy 1-20 | union, 202606 precedence (dedup by filename) |
| `algorithm_training_data_20260112` | `;` | ELEMnnPRES/ACC/POS | id keeps native K-prefix |
| `bsp_ocsplus_202511/Machine_rater` | `,` | ELEMnnPRES/ACC/POS | TotalScore = sum (no column); Human/xlsx + matfiles ignored |
| `training_data_oxford_manual_rater_202512` | `,` | none (TotalScore only) | ~52 label rows have no image (skipped) |

To onboard a **new** source, add a `load_<source>()` generator yielding
`(unified_row, src_image_path)` and register it in `main()`.
