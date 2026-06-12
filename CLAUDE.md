# NPSketch — Agent Guide (CLAUDE.md)

Operational reference for working in this repo: architecture, Docker layout, file-access
patterns, and the gotchas that aren't obvious from the code. Read this first, then `README.md`
for the full functional documentation.

---

## 1. What this project is

**NPSketch** is a computer-vision + ML application for analysing hand-drawn neuropsychological
figures (e.g. OCS-Plus / Oxford copy & recall tasks).

Two largely independent capabilities live side by side:

1. **Algorithm pipeline** — detect lines (Hough + iterative pixel subtraction), match a drawing
   against a reference template (Hungarian algorithm), and score correct / missing / extra lines.
2. **AI pipeline** — train ResNet-18 CNNs (regression *or* classification) to predict clinical
   features (`Total_Score`, `MMSE`, custom score classes) directly from a drawing image.

**Stack:** FastAPI (Python 3.10+) · SQLite (`npsketch.db`) · PyTorch (ResNet-18) · OpenCV / PIL ·
static HTML/JS frontend served by nginx.

---

## 2. Repository layout (verified)

```
NPSketch/
├── api/                              # FastAPI backend → mounted to /app in the container
│   ├── main.py                       # App entry point, mounts all routers
│   ├── database.py                   # SQLAlchemy models + DB setup (get_db / SessionLocal)
│   ├── models.py                     # FastAPI request/response Pydantic models  (≠ config/models.py)
│   ├── line_normalizer.py            # Shared 2px line-thickness normalization
│   │
│   ├── routers/                      # API endpoint modules (see §6)
│   │   ├── admin.py                  # Admin & migrations
│   │   ├── upload.py                 # Upload + algorithm evaluation
│   │   ├── evaluations.py            # Evaluation CRUD
│   │   ├── references.py             # Reference templates
│   │   ├── test_images.py            # Test image management
│   │   ├── training_data.py          # Training-data management (MAT/OCS extract, quality check)
│   │   ├── ai_training_base.py       # Dataset info, features, distributions, start-training
│   │   ├── ai_training_classification.py  # Class generation / custom-class endpoints
│   │   └── ai_training_models.py     # Model list / metadata / test / predict-single / delete
│   │
│   ├── image_processing/             # Algorithm CV library
│   │   ├── line_detector.py          # Hough Transform line detection
│   │   ├── comparator.py             # Hungarian-algorithm line matching
│   │   ├── image_registration.py     # Optional alignment to reference
│   │   └── utils.py
│   │
│   ├── ai_training/                  # ML pipeline
│   │   ├── model.py                  # ResNet-18 CNN
│   │   ├── trainer.py                # Training orchestration
│   │   ├── data_loader.py            # DB → augmented training set
│   │   ├── dataset.py                # PyTorch Dataset / DataLoader
│   │   ├── data_augmentation.py      # Diversity-controlled augmentation
│   │   ├── split_strategy.py         # Stratified train/val split
│   │   ├── normalization.py / preprocessing.py
│   │   ├── classification_generator.py
│   │   ├── synthetic_score_based.py  # Score-targeted synthetic images (v1.2.0, current)
│   │   ├── synthetic_bad_images.py / generate_synthetic_images.py  # Earlier synthetic approaches
│   │   ├── warmup_scheduler.py / visualization.py
│   │   └── *.md                      # CONTENT_PROTECTION, MODEL_METADATA, TRAINING_PIPELINE_ANALYSIS
│   │
│   ├── mat_extraction/               # MATLAB .mat extractor (+ .conf)
│   ├── ocs_extraction/               # OCS red-pixel extractor (+ .conf)
│   ├── oxford_extraction/            # Oxford PNG+CSV importer (normalizer, db_populator, validators)
│   ├── algorithm_extraction/         # algorithm_db_populator.py  (undocumented elsewhere — see §9)
│   ├── image_quality_check/          # contour_quality.py — training-image quality check
│   ├── config/                       # training_config.yaml + config_loader.py + models.py (Pydantic)
│   ├── services/                     # evaluation_service.py, reference_service.py
│   ├── migrations/                   # add_image_hash.{py,sql}
│   ├── utils/logger.py               # Structured logging
│   ├── Dockerfile · requirements.txt
│   └── (root-level dev/debug scripts: start_*_training.py, grid_search_quick.py,
│        debug_cont_lines_detailed.py, analyze_latest_training.py, test_*.py — see §9)
│
├── webapp/                           # Frontend (static HTML/JS/CSS, served by nginx) — 13 pages
│   ├── index.html · upload.html · reference.html · draw_testimage.html · run_test.html
│   ├── ai_training*.html (menu, overview, train, data_view, data_upload)
│   ├── training_evaluations.html · admin.html · docs.html
│   └── css/ (common.css, ai_training_common.css) · js/
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
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/ai_training/synthetic_score_based.py \
  --scores 0,10,20,30,40 --samples-per-score 10
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

**`training_data_images`**
```python
id              int   PK
patient_id      str   # "PC56", "Park_16", "C0078"
task_type       str   # COPY | RECALL | REFERENCE
source_format   str   # MAT | OCS | OXFORD | DRAWN | UPLOAD
original_file_data    bytes   # BLOB, original
processed_image_data  bytes   # BLOB, normalized 568×274 PNG, 2px black lines
image_hash      str   # SHA256 of the ORIGINAL file (duplicate detection)
features_data   str   # JSON, e.g. {"Total_Score": 45, "Custom_Class": {...}}
quality_check_status  str   # valid | invalid | NULL
quality_check_date    datetime
uploaded_at     datetime
# (+ ground_truth_*, session_id, extraction_metadata depending on source)
```

Other tables: `reference_images` (templates + manually-defined `lines_data` JSON),
`uploaded_images` (drawings for algorithm evaluation), `evaluation_results` (comparison output).
Confirm exact columns in `database.py` before relying on them — this list is a summary.

`features_data` example with a custom classification class:
```json
{ "Total_Score": 45,
  "Custom_Class": { "3": { "label": 1, "name_custom": "Fair",
                           "name_generic": "Class_1 [44-51]", "boundaries": [0, 44, 52, 60] } } }
```

---

## 6. API endpoints (by router)

Mounted in `api/main.py`. Full interactive list: `http://localhost/api/docs`.

- **upload.py** — `/api/upload`, `/api/normalize-image`, `/api/register-image`, `/api/check-duplicate`
- **evaluations.py** — `/api/evaluations/recent`, `/api/evaluations/{id}` (GET/DELETE), `/api/evaluations/{id}/evaluate`
- **references.py** — `/api/references`, `/api/references/{id}/image`, `/api/visualizations/{file}`
- **training_data.py** — `/api/training-data/upload`, `/api/extract-training-data`,
  `/api/training-data-evaluations`, `/api/training-data-image/{id}/evaluate|ground-truth`,
  `/api/training-data-image-quality-check/start|status`, `/api/training-data-image/{id}/quality-status`
- **ai_training_base.py** — `/api/ai-training/dataset-info`, `/available-features`,
  `/feature-distribution/{feature}`, `/start-training`, `/training-status`
- **ai_training_classification.py** — `/api/ai-training/custom-class-distribution/{feature}`,
  `/generate-classes`, `/recalculate-class-counts`
- **ai_training_models.py** — `/api/ai-training/models` (GET/DELETE),
  `/models/{filename}/metadata`, `/models/test`, `/models/predict-single`
- **admin.py** — admin / migration endpoints
- **test_images.py** — test-image management

> Endpoint paths are summarised from the routers; treat `/api/docs` as the source of truth.

---

## 7. Pipeline cheat-sheet

**Normalized image format (all extractors + augmentation output):** 568×274 px, RGB PNG, black lines
on white, line thickness **2.00 px** (Zhang-Suen thinning + dilation), ~5–7px margin.

**Import method (unified, 2026-06):** All training data lives in the consolidated base
`templates/labels.csv` + `templates/img/` (built by `api/data_consolidation/consolidate_templates.py`).
The **single** DB import path is `api/data_consolidation/import_unified.py` (auto-detects red vs black
image style, dedups by SHA256, skips blanks, writes the 60 sub-labels into `features_data.components`).
`source_format` ∈ {TELEFRED, OXFORD, ALGORITHM, OCS_MACHINE}; `task_type` ∈ {COPY, RECALL}.
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

**Training defaults (config):** ResNet-18 (ImageNet backbone), Adam, batch 8, sigmoid output for
regression, ReduceLROnPlateau (×0.5, patience 5, min 1e-6), differential LR (backbone ×0.1),
dropout 0.5, weight-decay 1e-4, early stopping (patience 15). Classification adds inverse-frequency
class weights + stratified split. Metrics: regression R²/RMSE/MAE/MAPE; classification
accuracy/F1/precision/recall/confusion-matrix.

**Synthetic score-based images (v1.2.0):** generate images targeting scores 0–40 to fix low-score
imbalance; added before the split and augmented like real data. See `synthetic_score_based.py`.

---

## 8. Configuration & logging

- **Central config:** `api/config/training_config.yaml` — single source of truth, no hardcoded
  training values. Access via `from config import get_config`. Note the live values:
  `augmentation.rotation_range = [-5, 5]`, `translation_range = [-3, 3]`,
  `training.defaults.batch_size = 8`.
- **Env overrides:** `NPSKETCH_<SECTION>_<KEY>` (e.g. `NPSKETCH_TRAINING_DEFAULTS_BATCH_SIZE=16`,
  `NPSKETCH_LOGGING_LEVEL=DEBUG`).
- **Type safety:** Pydantic models in `api/config/models.py` (`TrainingConfig`, `AugmentationConfig`,
  `SyntheticImageConfig`, `ModelMetadata`). Do not confuse with `api/models.py` (FastAPI schemas).
- **Logging:** `from utils.logger import get_logger`. File: `/app/data/logs/training.log`.

---

## 9. Gotchas & known doc/code mismatches

Things that could bite — verify against code, don't trust prose blindly:

- **Router naming:** older docs/READMEs refer to a single `ai_training.py`. It is actually split
  into `ai_training_base.py`, `ai_training_classification.py`, `ai_training_models.py`.
- **`algorithm_extraction/`** (`algorithm_db_populator.py`) and **`image_quality_check/`** exist in
  the tree but are undocumented in README. Inspect them before assuming behaviour; the purpose of
  `algorithm_extraction` is unclear (see §10).
- **Dead doc links:** earlier text referenced `DATA_IMPORT_FLOW.md` and
  `api/ai_training/LOCAL_WARPING.md`. Neither exists. The real ai_training docs are
  `CONTENT_PROTECTION.md`, `MODEL_METADATA.md`, `TRAINING_PIPELINE_ANALYSIS.md`.
- **Two `models.py`:** `api/models.py` (FastAPI I/O schemas) vs `api/config/models.py` (config
  validation). Import the right one.
- **Three synthetic-image generators** coexist (`synthetic_score_based.py` is current;
  `synthetic_bad_images.py` / `generate_synthetic_images.py` are earlier). Prefer the score-based one.
- **Upload vs Draw prediction differ:** `draw_testimage.html` sends the raw 568×274 canvas straight
  to `/api/ai-training/models/predict-single`; `upload.html` first runs `/api/normalize-image`
  (auto-crop + rescale + center), which can shift margins/line-scale and change the prediction.
- **Version drift:** `README.md` header still says "v1.0" while the footer/`CHANGELOG` say 1.2.0.
- **Root-level dev scripts** (`start_fast_training.py`, `start_full_training.py`,
  `grid_search_quick.py`, `debug_cont_lines_detailed.py`, `analyze_latest_training.py`,
  `test_*.py`) are ad-hoc helpers, not part of the served API. Treat as scratch tooling.

---

## 10. Open questions (flagged for the human)

- What is `algorithm_extraction/` for, and is it a supported import path alongside MAT/OCS/Oxford?
- Are the root-level `start_*` / `grid_search` / `debug_*` scripts still used, or can they be archived?
- Is `docs/` meant to hold anything? It is currently empty.
- Should the earlier synthetic-image generators be removed now that score-based is the standard?

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
**API docs:** http://localhost/api/docs · **Version:** 1.2.0
