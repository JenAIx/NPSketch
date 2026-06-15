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

## `import_unified.py` — THE single DB import path

Imports `templates/labels.csv` + `img/` into `training_data_images`. This is now
the **only** batch path into the DB (the old per-source CLI populators and the bulk
web endpoints `/api/extract-training-data[-oxford]` were retired — the latter return
HTTP 410). Interactive single-image upload (`/api/save-drawn-image`, prediction) is
unaffected.

```bash
# dry-run: detection + dedup + curation report, no DB writes
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/data_consolidation/import_unified.py --dry-run

# real import (after a fresh DB reset)
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/data_consolidation/import_unified.py
```

**Image style is auto-detected per image** (not hard-coded by source) — priority
cascade in `detect_style()`: red ink → red extraction; else dark + black-source name
→ bbox-crop; else heuristic (red-template w/o ink = blank). The source name is only a
consistency anchor; detected-vs-expected mismatches are logged.

**Curation rules:** blanks (no ink) skipped · SHA256-dedup on the original bytes ·
hash groups with conflicting nonzero scores dropped as corrupt · negative scores
dropped · `|total − sublabel_sum| > 5` kept but flagged (`extraction_metadata.label_warning`).

**Writes per row:** `uid`, `patient_id`, `task_type` (=cond), `source_format`,
`original_file_data` (raw), `processed_image_data` (568×274, 2 px), `image_hash`
(SHA256), `extraction_metadata` (style, counts, warnings), and `features_data`:
```json
{ "Total_Score": 45,
  "components": { "presence": [..20..], "accuracy": [..20..], "position": [..20..] } }
```
`components` is `null` for sources without sub-labels (OXFORD).

### Full DB reset + reload

```bash
docker exec npsketch-api cp /app/data/npsketch.db /app/data/npsketch.db.bak_<date>
docker exec -e PYTHONPATH=/app npsketch-api python3 -c \
  "from database import Base, engine, init_database; Base.metadata.drop_all(bind=engine); init_database()"
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/data_consolidation/import_unified.py
```
