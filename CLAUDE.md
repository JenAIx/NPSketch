# NPSketch — Agent Guide (CLAUDE.md)

Operational reference for working in this repo: Docker layout, file-access patterns, and the
gotchas that aren't obvious from the code. Read this first, then `README.md` for the full
functional documentation.

> 📐 **For the system architecture — an ASCII data-flow graph and a component-by-component map of
> training / synthetic / image-recognition / DB — see [`BLUEPRINT.md`](BLUEPRINT.md).** This file
> keeps the layout & cheat-sheet an agent needs at its fingertips; BLUEPRINT is the big picture.

---

## 1. What this project is

**NPSketch** is a computer-vision + ML application for analysing hand-drawn neuropsychological
figures (e.g. OCS-Plus / Oxford copy & recall tasks).

**AI-only (2026-06).** The app trains ResNet-18 CNNs to score drawings directly from the image,
in three modes: **regression** (`Total_Score`), **classification** (custom score classes), and
**components** (the 60 OCS-Plus sub-labels = 20 elements × Presence/Accuracy/Position; Total_Score =
their sum). The classical algorithm pipeline (Hough line detection, Hungarian template matching,
reference templates, line-based evaluation) was **removed** — including its routers, services,
`image_processing/`, and DB tables. Only `line_normalizer.py` (shared preprocessing) survives from it.

**Stack:** FastAPI (Python 3.10+) · SQLite (`npsketch.db`, single table `training_data_images`) ·
PyTorch (ResNet-18) · OpenCV / PIL · static HTML/JS frontend served by nginx.

---

## 2. Repository layout (verified)

> Big-picture data flow & subsystem diagram: [`BLUEPRINT.md`](BLUEPRINT.md) §1. Below is the
> file-level tree for precise navigation.

```
NPSketch/
├── api/                              # FastAPI backend → mounted to /app in the container
│   ├── main.py                       # App entry point, mounts all routers
│   ├── database.py                   # SQLAlchemy models + DB setup (get_db / SessionLocal)
│   ├── models.py                     # FastAPI request/response Pydantic models  (≠ config/models.py)
│   ├── line_normalizer.py            # Shared 2px line-thickness normalization
│   │
│   ├── routers/                      # API endpoint modules (see §6)
│   │   ├── admin.py                  # reset-database, cleanup-tmp
│   │   ├── upload.py                 # /normalize-image, /check-duplicate
│   │   ├── training_data.py          # Training-data management, /save-drawn-image (components)
│   │   ├── ai_training_base.py       # Dataset info, features, distributions, start-training
│   │   ├── ai_training_classification.py  # Class generation / custom-class endpoints
│   │   ├── ai_training_models.py     # Model list / metadata / test / predict-single / current-model marker
│   │   └── evaluator.py              # Evaluator: rater-reliability studies (build/queue/rating/analysis)
│   │
│   ├── data_consolidation/           # consolidate_templates.py + import_unified.py (the single import)
│   │   ├── coregistration.py         # shared gated RIGID_BODY pystackreg align (feature/coreg; not adopted)
│   │   ├── extract_elements_from_manual.py  # 20-element geometry from the scoring-manual PDF
│   │   └── map_elements_by_correlation.py   # validate element identity → model ELEM01-20 order
│   ├── ai_training/                  # ML pipeline
│   │   ├── model.py                  # ResNet-18 CNN
│   │   ├── trainer.py                # Training orchestration (regression / classification / components)
│   │   ├── data_loader.py            # DB → augmented training set
│   │   ├── dataset.py                # PyTorch Dataset / DataLoader
│   │   ├── data_augmentation.py      # Diversity-controlled augmentation
│   │   ├── split_strategy.py         # Patient-level stratified split (stratified_group_split; SYNTH_ → train)
│   │   ├── normalization.py / preprocessing.py
│   │   ├── gen_synth_image.py        # element-grounded synthetic low-score generator (current; exact labels)
│   │   ├── synthetic_score_based.py  # OLD score-targeted generator — BROKEN (ReferenceImage table gone)
│   │   ├── component_calibration.py / component_heatmaps.py  # post-hoc calibration + element maps
│   │   ├── warmup_scheduler.py / visualization.py
│   │   └── *.md                      # CONTENT_PROTECTION, MODEL_METADATA, TRAINING_PIPELINE_ANALYSIS
│   │
│   ├── mat_extraction/ · ocs_extraction/ · oxford_extraction/ · telefred_extraction/  # legacy/scan helpers
│   ├── config/                       # training_config.yaml + config_loader.py + models.py (Pydantic)
│   ├── line_normalizer.py            # shared 2px line normalization (kept from the old pipeline)
│   ├── migrations/                   # add_image_hash.{py,sql} (legacy)
│   ├── utils/logger.py               # Structured logging
│   ├── Dockerfile · requirements.txt
│   └── (root-level dev/debug scripts: start_*_training.py, grid_search_quick.py,
│        debug_cont_lines_detailed.py, analyze_latest_training.py, test_*.py — see §9)
│
├── webapp/                           # Frontend (static HTML/JS/CSS, served by nginx)
│   ├── index.html · evaluate.html (merged upload+draw: predict / label & save) · run_test.html
│   ├── upload.html · draw_testimage.html   # redirect stubs → evaluate.html?input=upload|draw
│   ├── ai_training*.html (menu, overview, train, data_view, data_upload, evaluator)
│   ├── admin.html · docs.html
│   └── css/ (common.css, ai_training_common.css) · js/ (component_result.js, component_editor.js)
│
├── data/                             # Persistent volume (RW): npsketch.db, models/, visualizations/, logs/, tmp/
├── templates/                        # Input data volume (RO): bsp_ocsplus_202511/, training_data_oxford_*/
├── docs/                             # (currently empty)
├── nginx/nginx.conf · docker-compose.yml
├── README.md · CHANGELOG.md · CLAUDE.md (this file)
```

---

## 3. Docker setup

Two services in `docker-compose.yml`:

| Service | Container        | Port (host:container) | Role |
|---------|------------------|-----------------------|------|
| nginx   | `npsketch-nginx` | `80:80`               | Serves the frontend + reverse-proxies `/api` |
| api     | `npsketch-api`   | `8000:8000`           | FastAPI backend (OpenCV, PyTorch) |

The app is reached via **nginx on port 80** → `http://localhost`. The API is also directly
reachable on `http://localhost:8000` (e.g. `/api/docs`).

### Volume mounts

| Host        | Container                       | Access | Purpose |
|-------------|---------------------------------|--------|---------|
| `./api`     | `/app`                          | **RW** | Backend code (uvicorn `--reload` hot-reload) |
| `./data`    | `/app/data`                     | **RW** | DB, models, visualizations, logs, tmp |
| `./templates`| `/app/templates`               | **RO** | Source input data |
| `./webapp`  | `/usr/share/nginx/html`         | **RO** | Frontend (nginx) |
| `./nginx/nginx.conf` | `/etc/nginx/nginx.conf` | **RO** | Reverse-proxy config |

> `/app/templates` is **read-only**. To produce files that belong there, write to `/app/data/tmp/`
> inside the container, then `docker cp` them out to the host.

### Common commands

```bash
docker compose up -d                 # start
docker compose up --build -d         # rebuild after Dockerfile / requirements change
docker compose logs -f api           # follow API logs
docker compose restart               # restart
docker exec npsketch-api python3 /app/<script>.py [args]   # run code in the container
docker exec -it npsketch-api /bin/bash                     # shell
```

Run Python in the container (note the container name is **`npsketch-api`**, not `api`):

```bash
docker exec npsketch-api python3 -c "
from database import get_db, TrainingDataImage
db = next(get_db())
print('rows:', db.query(TrainingDataImage).count())
db.close()
"
```

Some ML scripts need `PYTHONPATH=/app`:

```bash
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/ai_training/gen_synth_image.py \
  --preview 9 --scores 5,10,15,20,25,30   # generate + CNN-read, no DB write
```

---

## 4. File-access rules (inside the container)

- Always use **container paths** (`/app/...`) in Python code, not host paths.
- DB: `/app/data/npsketch.db`. Always go through `get_db()` / `SessionLocal()` from `database.py`;
  `db.commit()` after writes, `db.close()` when done.
- Templates: read from `/app/templates/...`; **never write there** — write to `/app/data/tmp/` then
  `docker cp npsketch-api:/app/data/tmp/<f> ./templates/...`.
- Models → `/app/data/models/`. Visualizations → `/app/data/visualizations/`. Logs →
  `/app/data/logs/training.log` (10 MB rotation, 5 backups).

---

## 5. Key database tables

Defined in `api/database.py`. The main ML table:

**`training_data_images`** (current contents: 7693 rows, all from the unified import — see §7)
```python
id              int   PK
uid             str   # unique key from templates/labels.csv, e.g. "TF-2020_08_27-257-COPY"
patient_id      str   # source-prefixed, groups COPY+RECALL: "TF-257", "ALG-PC0001", "OXF-C0078"
task_type       str   # COPY | RECALL | MANUAL  (REFERENCE for templates; MANUAL = standalone LOWSCORER figures)
source_format   str   # TELEFRED | OXFORD | ALGORITHM | OCS_MACHINE | LOWSCORER  (+ SYNTHETIC; legacy: MAT/OCS/DRAWN/UPLOAD)
original_file_data    bytes   # BLOB, original
processed_image_data  bytes   # BLOB, normalized 568×274 PNG, 2px black lines
image_hash      str   # SHA256 of the ORIGINAL file (duplicate detection)
features_data   str   # JSON: Total_Score + optional 60 component sub-labels (see below); NULL = unlabelled
                      #   (e.g. label_status='zero' placeholders — kept as images, excluded from training)
uploaded_at     datetime
# (+ session_id, extraction_metadata depending on source)
# (quality_check_status/date columns were removed in 2.1.0)
```

Evaluator tables (rater-reliability studies, auto-created by `init_database()`):
`evaluation_studies` (config: name, model_filename, n_raters, per_band_config),
`evaluation_items` (study image set + frozen GT snapshot: gt_total, gt_components),
`evaluation_ratings` (each rater's blind score per image — the rater work lives here),
`evaluation_model_runs` (cached model prediction per (study, model) for analysis).

Other tables: `reference_images` (templates + manually-defined `lines_data` JSON),
`uploaded_images` (drawings for algorithm evaluation), `evaluation_results` (comparison output).
Confirm exact columns in `database.py` before relying on them — this list is a summary.

`features_data` shape (component sub-labels are the OCS-Plus 20 elements × Presence/Accuracy/Position;
`components` is null for sources without them, e.g. OXFORD):
```json
{ "Total_Score": 45,
  "components": { "presence": [..20..], "accuracy": [..20..], "position": [..20..] } }
```
(Older rows could also carry a `Custom_Class` block for classification class definitions.)

---

## 6. API endpoints (by router)

Mounted in `api/main.py`. Full interactive list: `http://localhost/api/docs`.

- **upload.py** — `/api/normalize-image`, `/api/check-duplicate` (algorithm `/api/upload` +
  `/api/register-image` were removed)
- **training_data.py** — `/api/save-drawn-image` (accepts `components`, `source_format` UPLOAD/DRAWN),
  `/api/training-data-images` (list returns `total_score` + `has_components`; `only_missing` filter),
  `/api/training-data-image/{id}` (GET/DELETE), `/{id}/features`, `/{id}/original`.
  (Bulk `/api/extract-training-data[-oxford]` return HTTP 410; ground-truth, algorithm-eval,
  quality-check, and bulk feature-CSV endpoints were removed.)
- **ai_training_base.py** — `/api/ai-training/dataset-info`, `/available-features`,
  `/feature-distribution/{feature}`, `/start-training`, `/training-status`
- **ai_training_classification.py** — `/api/ai-training/custom-class-distribution/{feature}`,
  `/generate-classes`, `/recalculate-class-counts`
- **ai_training_models.py** — `/api/ai-training/models` (GET/DELETE; GET returns `is_current`),
  `/models/{filename}/metadata`, `/models/test`, `/models/predict-single` (`return_normalized`),
  `/models/run-on-test-images` ("Run Tests"), `/models/current` (GET), `/models/set-current` (POST)
- **evaluator.py** — `/api/evaluator/studies` (GET/POST), `/studies/{id}` (GET/DELETE),
  `/studies/{id}/queue?rater=`, `/studies/{id}/rating` (POST), `/studies/{id}/analysis?model=`,
  `/studies/{id}/export.csv`
- **admin.py** — `/api/admin/reset-database`, `/api/admin/cleanup-tmp`

> Endpoint paths are summarised from the routers; treat `/api/docs` as the source of truth.

---

## 7. Pipeline cheat-sheet

> Prose walkthrough of the same pipeline (sources → consolidate → import → train → infer):
> [`BLUEPRINT.md`](BLUEPRINT.md) §3–§7. Below are the exact numbers/flags to keep at hand.

**Normalized image format (all extractors + augmentation output):** 568×274 px, RGB PNG, black lines
on white, line thickness **2.00 px** (Zhang-Suen thinning + dilation), ~5–7px margin.

**Import method (unified, 2026-06):** All training data lives in the consolidated base
`templates/labels.csv` + `templates/img/` (built by `api/data_consolidation/consolidate_templates.py`).
The **single** DB import path is `api/data_consolidation/import_unified.py` (auto-detects red vs black
image style, dedups by SHA256, skips blanks, writes the 60 sub-labels into `features_data.components`).
`source_format` ∈ {TELEFRED, OXFORD, ALGORITHM, OCS_MACHINE, **LOWSCORER**} (+ **SYNTHETIC** for
generated low-score images, see §7); `task_type` ∈ {COPY, RECALL, **MANUAL**}. The incremental
`import_unified.py --only-source <fmt>` imports just one source without re-inserting the rest
(non-destructive; how LOWSCORER was added).
- The old per-source CLI populators (`telefred_import.py`, `oxford_db_populator.py`,
  `algorithm_db_populator.py`) were **deleted**; the bulk web endpoints
  `/api/extract-training-data[-oxford]` were **retired (HTTP 410)**.
- The preprocessing libraries (`ocs_extraction`, `oxford_extraction/oxford_normalizer.py`,
  `mat_extraction`, `line_normalizer.py`) remain — `import_unified.py` reuses them.
- Interactive single-image upload (`/api/save-drawn-image`, prediction) is unchanged.

**Augmentation (current):** pre-shrink 5% (→ ~14px margins) → diversity-controlled transforms
→ SSIM filter (vs original <0.95, between augs <0.93, progressive retry) → re-binarize (175)
→ 2px line normalize. ~6 augs/image (7× total). Mix: 50% rotation+translation, 33% warp,
17% warp+combined. Rotation ±5°, translation ±3px, IDW local warp (9 control points, 15–20px).

**Three training modes** (selected by `target_feature` in `run_training_job`, all share the ResNet-18
backbone + patient-level split + 284×137 input downscale + augmentation + epoch logging + best
checkpoint):
- **regression** (e.g. `Total_Score`): linear output + MSE, min-max target normalization,
  WeightedRandomSampler over score bins. Metrics R²/RMSE/MAE/MAPE + per-score-decade.
- **classification** (`Custom_Class_<N>`): softmax + CrossEntropy, inverse-frequency class weights.
  Metrics accuracy/F1/precision/recall/confusion-matrix.
- **components** (`Components`, 2026-06): 60 sub-labels (20 elements × Presence/Accuracy/Position),
  `BCEWithLogitsLoss` + per-label `pos_weight`, Total_Score = sum. Trains on **TELEFRED + SYNTHETIC +
  any human-`validated` row** (`data_loader` `include_validated` → `source.in_(('TELEFRED','SYNTHETIC'))
  OR validated==True`, components required; `pos_weight` over the same set). Metrics
  per-sub-label/aspect/component F1 + derived-score R²/RMSE/MAE + per-decade. Launch:
  `start_telefred_component_training.py`. `predict-single` returns 60 probabilities + derived score.

**Shared defaults:** ResNet-18 (ImageNet backbone), Adam, batch 8, ReduceLROnPlateau (×0.5,
patience 5, min 1e-6), differential LR (backbone ×0.1), dropout 0.5, weight-decay 1e-4, early stopping.
(Regression uses a **linear** head now — the old sigmoid saturated on the skewed score distribution.)

**Synthetic low-score images (current):** the component CNN was blind below ~18 (real low-score
drawings are scarce). Fixed by **element-grounded generation** — the 20-element geometry is
extracted from the official scoring manual into `data/element_definitions.json` (see
`docs/SCORING_CRITERIA.md`; hand-editable via the annotation tool in `component_map.html`), then
`gen_synth_image.py` composes low-score figures as `element region ∩ reference ink = real strokes`
with **exact** 60-component labels (type-aware position/accuracy degradation). `--insert` writes rows
as `source_format='SYNTHETIC'`, `patient_id='SYNTH_*'` (forced into train via `split_strategy`);
component training picks them up (`data_loader` source filter `('TELEFRED','SYNTHETIC')` **+ any
`validated` row** when `include_validated`). `build_priors` samples the per-band conditionals from
**TELEFRED + LOWSCORER** (real low-score data sharpens the otherwise-sparse low bands). `--purge`
removes synthetic rows but **preserves `validated` ones** — validate a synthetic row in the data-view
("✓ Validieren (so sperren)") to keep a good one across a purge. Earlier: +500 synthetic cut
derived-score MAE −25 % (0–29). The deployed model is whatever `data/models/current_models.json`
pins (set via "⭐ Set as current"); a retrain on real + LOWSCORER + freshly-rebuilt synthetic was
launched 2026-06-24. (The old `synthetic_score_based.py` is broken — loads the removed
`ReferenceImage` table.)

**LOWSCORER (2026-06):** 662 manually-rated *real* low-score copy figures (Total_Score 1–15), the
real-data complement to synthetic. Imported `validated=True` via a `load_lowscorer` loader in
`consolidate_templates.py` (each ELEM 0–3 score → PRES/ACC/POS, `2`→`[1,1,0]`) + `import_unified.py
--only-source LOWSCORER`. See `templates/old/low_scores/` (source) — gitignored.

**Element geometry is hand-verified — do NOT "fix" it.** `element_definitions.json` `order` is stamped
`hand_verified_against_manual`. The recurring "5 elements swapped" report from
`map_elements_by_correlation.py` / `validate_element_order.py` is a **statistical artifact**: the
disputed elements {4,9,14,15,19,20} are present in 65–99 % of drawings, so their near-constant
`presence` column defeats the ink-vs-presence correlation and the present-minus-absent heatmap. Do
NOT blindly re-run those Hungarian re-save scripts — they would corrupt the verified geometry.

> **Branches:** `feature/coreg` holds the (not-adopted) coregistration experiment + the element/
> synthetic work; `feature/synthetic-gen` continues the synthetic generator (v2 realism). DB backups:
> `data/npsketch.precoreg.db`, `data/npsketch.presynth.db`.

---

## 8. Configuration & logging

- **Central config:** `api/config/training_config.yaml` — single source of truth, no hardcoded
  training values. Access via `from config import get_config`. Note the live values:
  `augmentation.rotation_range = [-5, 5]`, `translation_range = [-3, 3]`,
  `training.defaults.batch_size = 8`.
- **Env overrides:** `NPSKETCH_<SECTION>_<KEY>` (e.g. `NPSKETCH_TRAINING_DEFAULTS_BATCH_SIZE=16`,
  `NPSKETCH_LOGGING_LEVEL=DEBUG`). These only override keys that already exist in
  `training_config.yaml`.
- **Secrets (gitignored root `.env`):** `CLOUDFLARE_TUNNEL_TOKEN` (used by `cloudflared` via Compose)
  and `NPSKETCH_ACCESS_TOKEN` (the app access-token gate — read directly with `os.getenv` in
  `api/auth.py`, **not** via the YAML override loader; passed to the `api` service in
  `docker-compose.yml`). Rotating `NPSKETCH_ACCESS_TOKEN` invalidates all existing sessions. Note:
  editing bind-mounted single files (`nginx.conf`) needs `docker compose up -d --force-recreate nginx`
  to take effect — a plain reload sees the stale inode.
- **Type safety:** Pydantic models in `api/config/models.py` (`TrainingConfig`, `AugmentationConfig`,
  `SyntheticImageConfig`, `ModelMetadata`). Do not confuse with `api/models.py` (FastAPI schemas).
- **Logging:** `from utils.logger import get_logger`. File: `/app/data/logs/training.log`.

---

## 9. Gotchas & known doc/code mismatches

Things that could bite — verify against code, don't trust prose blindly:

- **Router naming:** older docs/READMEs refer to a single `ai_training.py`. It is actually split
  into `ai_training_base.py`, `ai_training_classification.py`, `ai_training_models.py`.
- **`algorithm_extraction/`** (`algorithm_db_populator.py`) exists in the tree but is undocumented in
  README. Inspect it before assuming behaviour; its purpose is unclear (see §10).
  (`image_quality_check/` was removed in 2.1.0 — the quality check had no effect on training.)
- **Dead doc links:** earlier text referenced `DATA_IMPORT_FLOW.md` and
  `api/ai_training/LOCAL_WARPING.md`. Neither exists. The real ai_training docs are
  `CONTENT_PROTECTION.md`, `MODEL_METADATA.md`, `TRAINING_PIPELINE_ANALYSIS.md`.
- **Two `models.py`:** `api/models.py` (FastAPI I/O schemas) vs `api/config/models.py` (config
  validation). Import the right one.
- **Synthetic images:** `synthetic_score_based.py` is the current (and only) generator; the earlier
  `synthetic_bad_images.py` / `generate_synthetic_images.py` were deleted in the AI-only pivot.
- **Upload vs Draw prediction differ:** upload and draw are now one page (`evaluate.html`;
  `upload.html`/`draw_testimage.html` redirect to it). The **Draw** source sends the raw 568×274 canvas
  straight to `/api/ai-training/models/predict-single`; the **Upload** source first runs
  `/api/normalize-image` (auto-crop + rescale + center), which can shift margins/line-scale and change
  the prediction. Both feed predict/label via `getActiveImageBlob()`.
- **Root-level dev scripts** (`start_fast_training.py`, `start_full_training.py`,
  `grid_search_quick.py`, `debug_cont_lines_detailed.py`, `analyze_latest_training.py`,
  `test_*.py`) are ad-hoc helpers, not part of the served API. Treat as scratch tooling.

---

## 10. Open questions (flagged for the human)

- What is `algorithm_extraction/` for, and is it a supported import path alongside MAT/OCS/Oxford?
- Are the root-level `start_*` / `grid_search` / `debug_*` scripts still used, or can they be archived?
- Is `docs/` meant to hold anything? It is currently empty.
- (Partly addressed) Component head now also trains on validated OXFORD/DRAWN. Still open: label
  the NULL-feature ("Only Missing") TELEFRED images so they re-enter training; and run a real
  **Evaluator** study with 2 human raters for the publication numbers.

---

## 11. Quick debugging

```bash
docker ps | grep npsketch-api               # is it running?
docker logs -f npsketch-api                 # logs
docker exec npsketch-api ls -la /app/data   # check writable data dir
docker exec npsketch-api python3 -c "from database import get_db; next(get_db()); print('DB ok')"
```

---

**Container:** `npsketch-api` · **Workdir:** `/app` · **App URL:** http://localhost ·
**API docs:** http://localhost/api/docs · **Version:** 2.3.0
