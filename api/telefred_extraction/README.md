# TeleFred Dataset Extraction

Scripts for importing **TeleFred** deliveries into the NPSketch training database
(`training_data_images`, `source_format="TELEFRED"`).

A TeleFred dataset is a directory containing:

```
training_data_telefred_<tag>/
├── fc0/                 # PNG drawings, task type FC0  (red ink on a printed template)
├── fc1/                 # PNG drawings, task type FC1
├── fc0.csv             # ;-separated metadata incl. ID, FileName, TotalScore
└── fc1.csv
```

Drawings are stored as **red ink** on a printed reference. Import extracts the red
pixels (reusing `api/ocs_extraction/ocs_extractor.py`), renders them onto a 568×274
canvas, and stores both the original PNG and the processed image.

---

## Contents

- **`telefred_scan.py`** — pre-import sanity check. Reports per folder: PNG count,
  files with/without red pixels, unreadable files, resolution mix, red-pixel stats.
- **`telefred_import.py`** — imports into the database. Filters to images with ≥1 red
  pixel, dedups by SHA256 of the original PNG, writes `features_data={"Total_Score": N}`
  when `TotalScore ≠ 0`. Supports **multiple `--base` directories** for union imports.

Database fields written per image: `patient_id` (CSV `ID`), `task_type` (`FC0`/`FC1`),
`source_format="TELEFRED"`, `image_hash` (SHA256 of original), `session_id`
(`telefred_<today>`), `extraction_metadata` (JSON: source dataset, resolution, red count,
bbox, threshold), `features_data` (only if scored).

---

## Workflow

Run inside the API container (`PYTHONPATH=/app` so `database` / `ocs_extraction` resolve):

```bash
# 1. (optional) scan a delivery before importing
docker exec -e PYTHONPATH=/app npsketch-api python3 \
  /app/telefred_extraction/telefred_scan.py \
  --input-base /app/templates/training_data_telefred_202606

# 2. import a single delivery (no per-folder limit)
docker exec -e PYTHONPATH=/app npsketch-api python3 \
  /app/telefred_extraction/telefred_import.py \
  --base /app/templates/training_data_telefred_202606 \
  --limit 0
```

### Union of several deliveries (label precedence)

When the same drawing appears in more than one delivery (byte-identical PNG), the **first**
`--base` that contains it wins — its CSV label is the one that lands in the database; later
duplicates are skipped. List the newest / most-authoritative dataset first:

```bash
docker exec -e PYTHONPATH=/app npsketch-api python3 \
  /app/telefred_extraction/telefred_import.py \
  --base /app/templates/training_data_telefred_202606 \
  --base /app/templates/training_data_telefred_20260119 \
  --limit 0
```

This produces the union of both deliveries: overlapping images keep the **202606** label,
images only present in the older delivery are still added.

> **Rebuild note:** import only *adds* rows (and skips duplicates); it never deletes. To
> rebuild the TeleFred set from scratch, first remove the existing rows, then re-import:
> `DELETE FROM training_data_images WHERE source_format='TELEFRED';`
> Back up `data/npsketch.db` before doing so.

---

## Options (`telefred_import.py`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--base` | (required, repeatable) | Dataset dir with `fc0/`, `fc1/`, `fc0.csv`, `fc1.csv`. Earlier = higher precedence. |
| `--limit` | `100` | Max images per folder per base; `0` or negative = no limit. |
| `--min-score` | `0` | Skip rows with `TotalScore` below this. |
| `--config` | auto | Path to `ocs_extractor.conf` (red threshold / padding). Defaults to `r≥200, g≤100, b≤100`, padding 5px. |
| `--output` | `/app/data/tmp/telefred_processed` | Temp dir for rendered images (not stored in DB). |
