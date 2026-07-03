# NPSketch — Architecture Blueprint

> System-level map of every component and how they interact. For day-to-day agent
> conventions see `CLAUDE.md`; for functional docs see `README.md`. This file is the
> "big picture" — read it first to understand *where things live and how data flows*.

**What it is:** a computer-vision + ML app that scores hand-drawn neuropsychological figures
(OCS-Plus copy/recall tasks) directly from the image with ResNet-18 CNNs — no classical
line-detection pipeline (that was removed). Three training modes: **regression** (Total_Score),
**classification** (custom score classes), **components** (60 OCS-Plus sub-labels = 20 elements
× Presence/Accuracy/Position; Total_Score = their sum). Deployed model today:
`model_Components_20260625_055133.pth`. Version **2.3.0**.

---

## 1. Top-level architecture

```
                                     ┌───────────────────────────────────────────────┐
                                     │                  BROWSER (user)                │
                                     │  evaluate · ai_training_* · component_map ·    │
                                     │  admin · docs   (static HTML/JS)               │
                                     └───────────────────┬───────────────────────────┘
                                                         │ HTTP
   Cloudflare Quick Tunnel  ─────────────────────────────┤  (public, ephemeral URL)
   (npsketch-cloudflared → nginx:80)                     │
                                                         ▼
                                     ┌───────────────────────────────────────────────┐
                                     │  nginx  (npsketch-nginx, :80)                  │
                                     │  serves /webapp  ·  proxies /api → api:8000    │
                                     └───────────────────┬───────────────────────────┘
                                                         │ /api/*
                                                         ▼
   ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   │  FastAPI backend  (npsketch-api, :8000, /app)                                             │
   │                                                                                           │
   │   ROUTERS: admin · upload · training_data · ai_training_{base,classification,models} ·    │
   │            evaluator                                                                      │
   └───────┬───────────────────────┬───────────────────────────┬──────────────────────────────┘
           │                       │                           │
           ▼                       ▼                           ▼
   ┌───────────────┐      ┌──────────────────┐        ┌──────────────────────────┐
   │ IMAGE         │      │  MODEL TRAINING   │        │  INFERENCE / PREDICT     │
   │ RECOGNITION   │      │  (ai_training/)   │        │  predict-single ·        │
   │ (preprocess)  │      │                   │        │  run-on-test-images ·    │
   │ line_normalize│      │  ResNet-18        │        │  gradcam                 │
   │ 568×274, 2px  │      │  ├ regression MSE │        │  → 60 probs + score      │
   │ →CNN 284×137  │      │  ├ classific. CE  │        └───────────┬──────────────┘
   └───────┬───────┘      │  └ components BCE │                    │
           │              │     (60 labels)   │                    │ reads
           │              └───────┬───────────┘                    │
           │                      │ reads / writes                 │
           ▼                      ▼                                ▼
   ┌───────────────────────────────────────────────────────────────────────────────┐
   │  SQLite  data/npsketch.db                                                       │
   │  training_data_images  (images + 60-label features)                            │
   │  evaluation_{studies,items,ratings,model_runs}  (rater-reliability)            │
   └───────────────────────────────▲───────────────────────────────────────────────┘
                                    │ import_unified.py  (dedup, style-detect, preprocess)
                                    │
   ┌────────────────────────────────┴──────────────────────────────────────────────┐
   │  CONSOLIDATED BASE   templates/labels.csv  +  templates/img/                    │
   │        ▲ consolidate_templates.py  (host; rebuilds from all sources)            │
   └────────┼───────────────────────────────────────────────────────────────────────┘
            │
   ┌────────┴───────────────────────────────────────┐     ┌──────────────────────────────┐
   │  SOURCES  templates/old/*                       │     │  SYNTHETIC  (element-grounded)│
   │  TELEFRED · ALGORITHM · OCS_MACHINE · OXFORD ·  │     │  gen_synth_image.py           │
   │  LOWSCORER (real low-score 1–16)                │◄────┤  element_definitions.json ∩   │
   └─────────────────────────────────────────────────┘prior│  reference ink → exact labels │
                                                            └──────────────────────────────┘
                                    │                                   ▲
                                    └── build_priors() samples ─────────┘
                                        per-band from TELEFRED + LOWSCORER

   Supporting artefacts in data/:  models/*.pth + *_metadata.json · current_models.json ·
   element_definitions.json (geometry) · element_priors.json · visualizations/ · logs/
```

**One-line data story:** `templates/old/* → consolidate_templates.py → labels.csv+img/ →
import_unified.py → npsketch.db → ai_training (augment→train) → models/*.pth → inference in the
browser`. Synthetic low-score figures are injected into the DB to fill the weak low band; the
Evaluator subsystem runs blind human-rater studies against a chosen model.

---

## 2. Deployment (`docker-compose.yml`, `nginx/nginx.conf`)

Three services on the `npsketch-network` bridge (talk by container name):

| Service | Container | Image | Port | Role |
|---------|-----------|-------|------|------|
| nginx | `npsketch-nginx` | `nginx:latest` | `80:80` | Serves `/webapp`; reverse-proxies `/api/`→`api:8000`; `client_max_body_size 500M` |
| api | `npsketch-api` | Dockerfile | `8000:8000` | FastAPI + PyTorch + OpenCV + SQLite; uvicorn `--reload` |
| cloudflared | `npsketch-cloudflared` | `cloudflare/cloudflared:latest` | — | Quick Tunnel `--url http://nginx:80`, metrics `:2000` (`/ready`, `/quicktunnel`) → **ephemeral public URL** |

**Volume mounts:**

| Host | Container | Access | Purpose |
|------|-----------|--------|---------|
| `./api` | `/app` | RW | Backend code (hot-reload) |
| `./data` | `/app/data` | RW | DB, models, visualizations, logs, tmp, element JSONs |
| `./templates` | `/app/templates` | **RO** | Consolidated base + source archives (`old/`) |
| `./webapp` | `/usr/share/nginx/html` | RO | Frontend |
| `./nginx/nginx.conf` | `/etc/nginx/nginx.conf` | RO | Proxy config |

> `/app/templates` is read-only inside the container — write generated files to `/app/data/tmp/`
> then `docker cp` out. **App URL:** http://localhost · **API docs:** http://localhost/api/docs

---

## 3. Data pipeline (sources → DB)

### 3a. Consolidation — `api/data_consolidation/consolidate_templates.py` (host, stdlib, no DB)
Rebuilds the single consolidated base `templates/labels.csv` + `templates/img/` from all source
folders in `templates/old/`. One loader per source, each yielding `(unified_row, image_path)`:

| Loader | Source folder | `source_format` | `task_type` | Labels |
|--------|---------------|-----------------|-------------|--------|
| `load_telefred` | `training_data_telefred_{202606,20260119}` | TELEFRED | COPY/RECALL | full 60 |
| `load_algorithm` | `algorithm_training_data_20260112` | ALGORITHM | COPY/RECALL | full 60 |
| `load_ocs_machine` | `bsp_ocsplus_202511/Machine_rater` | OCS_MACHINE | COPY/RECALL | full 60 |
| `load_oxford` | `training_data_oxford_manual_rater_202512` | OXFORD | COPY/RECALL | **Total only** |
| `load_lowscorer` | `low_scores/Bildbewertung.csv` | LOWSCORER | MANUAL | 60 (ELEM 0-3 → PRES/ACC/POS) |

Output CSV schema (49 cols): `uid, patient_id, source, cond, filename, orig_id, orig_filename,
total_score, total_score_sum, label_status, ELEM01PRES…ELEM20POS (60), gender, age, test_date`.
The `_split_elem_score` rule: `0/''→[0,0,0]`, `1→[1,0,0]`, `2→[1,1,0]`, `3→[1,1,1]` (the **`2`**
split presence+accuracy is a known approximation — the "review-later" set).

### 3b. Import — `api/data_consolidation/import_unified.py` (container)
The **single** DB import path. `labels.csv` + `img/` → `training_data_images`.
- **Style auto-detect** (`detect_style`): red ink (TELEFRED scans) → `process_red` (ocs_extraction);
  black lines (`BLACK_SOURCES = OXFORD/ALGORITHM/OCS_MACHINE/LOWSCORER`) → `process_black`
  (oxford_extraction). Both output 568×274 PNG, black 2px lines.
- **Dedup:** SHA256 on originals; hash-groups with conflicting nonzero scores dropped; blanks skipped.
- **Write-protect:** rows with `validated=True` are skipped on reimport (preserve human corrections).
- **`build_features`** → `features_data` JSON `{Total_Score, components:{presence,accuracy,position}}`
  (`components=null` for OXFORD / unscored placeholders).
- **Flags:** `--only-source <fmt>` (incremental, non-destructive add), `--dry-run`, `--coregister`
  (gated pystackreg align, experimental, not adopted).

### 3c. Element geometry helpers
- `extract_elements_from_manual.py` — extracts the 20-element geometry from the OCS-Plus scoring
  manual PDF → `data/element_definitions.json` (hand-verified; **do not auto-"fix"**).
- `map_elements_by_correlation.py` — Hungarian validation of element order (the recurring
  "5 swapped elements" report is a **statistical artifact** of high-presence elements — do not re-run blindly).

### 3d. Preprocessing libs (shared "image recognition" layer)
`api/line_normalizer.py` (Zhang-Suen skeletonize → dilate to 2px) · `ocs_extraction/` (red-pixel
extraction) · `oxford_extraction/oxford_normalizer.py` (auto-crop + AR-fit + normalize).
`mat_extraction/`, `telefred_extraction/` are legacy/inactive. **Canonical normalized format:**
568×274 RGB, black lines on white, 2.00px thickness, ~5–7px margin.

---

## 4. Database (`api/database.py`)

`sqlite:////app/data/npsketch.db`, accessed via `get_db()` / `SessionLocal`. `init_database()` on
startup runs `create_all` + additive migrations.

**`training_data_images`** (main ML table):
- Identity: `id`, `uid` (unique), `patient_id` (groups COPY+RECALL), `task_type`
  (COPY/RECALL/MANUAL/REFERENCE), `source_format`.
- Image: `original_file_data` (BLOB), `processed_image_data` (BLOB 568×274), `image_hash` (SHA256),
  `original_filename`.
- Labels: `features_data` (JSON; NULL = unlabelled), `model_prediction` (JSON, frozen),
  `validated` (bool), `validated_at`.
- Meta: `session_id`, `uploaded_at`, `extraction_metadata`.

**Evaluator tables:** `evaluation_studies` (config), `evaluation_items` (image set + frozen GT
snapshot), `evaluation_ratings` (blind per-rater scores), `evaluation_model_runs` (cached model
prediction per study+model).

---

## 5. ML pipeline (`api/ai_training/`)

**Entry:** `run_training_job(config)` in `api/routers/ai_training_base.py` — detects mode from
`target_feature`, prepares data (augmentation), runs `CNNTrainer.train()`, saves checkpoint + metadata.

> Deep dives: `api/ai_training/CONTENT_PROTECTION.md` (augmentation content-bounds safety) ·
> `api/ai_training/MODEL_METADATA.md` (checkpoint metadata schema) ·
> `api/ai_training/TRAINING_PIPELINE_ANALYSIS.md` (early-stopping/leakage audit + recommendations).

- **`model.py` — `DrawingClassifier`:** ResNet-18 (ImageNet backbone, grayscale-adapted). Configurable
  `num_outputs` = 1 (regression) / N (classification) / **60** (components). Linear head (the old
  sigmoid saturated on skewed scores). `freeze_backbone`/`unfreeze_all`.
- **`trainer.py` — `CNNTrainer`:** the three modes share backbone + patient-split + 284×137 downscale
  + augmentation + best-checkpoint + epoch logging:

  | Mode | Selector | Loss | Imbalance | Metrics |
  |------|----------|------|-----------|---------|
  | regression | numeric feature | MSE + min-max target norm | WeightedRandomSampler (bins) | R²/RMSE/MAE/MAPE + per-decade |
  | classification | `Custom_Class_*` | CrossEntropy | inverse-frequency class weights | acc/F1/precision/recall/confusion |
  | components | `Components` | BCEWithLogitsLoss + per-label pos_weight | pos_weight | per-label/aspect F1 + derived-score R²/RMSE/MAE |

  Adam + differential LR (backbone ×0.1) + ReduceLROnPlateau (×0.5, patience 5, min 1e-6) + dropout
  0.5 + weight-decay 1e-4 + early stopping. `save_model`/`load_model` write `*.pth` + `*_metadata.json`.
- **Data loading — `dataset.py` / `data_loader.py` / `split_strategy.py`:** DB rows → augmented set.
  Components filter: `source_format.in_(('TELEFRED','SYNTHETIC')) OR validated==True` (so LOWSCORER +
  any human-validated OXFORD/DRAWN row joins in), `include_validated`, optional `exclude_sources`
  (non-destructive ablation). `stratified_group_split` splits at patient level (no COPY/RECALL leak);
  `SYNTH_*` / synthetic forced into train.
- **Augmentation — `data_augmentation.py`:** pre-shrink 5% → diversity-controlled transforms
  (50% rot+trans, 33% warp, 17% warp+combined; rot ±5°, trans ±3px, IDW warp 9 pts 15–20px) →
  SSIM filter (vs original <0.95, between augs <0.93, progressive retry) → re-binarize (175) →
  2px normalize. ~6 augs/image (7× total).
- **Preprocessing/normalization — `preprocessing.py` / `normalization.py`:** pre-shrink → binarize
  (175) → 2px line-normalize → grayscale [0,1] → downscale to 284×137. `TargetNormalizer` (min-max)
  for regression targets only.

---

## 6. Synthetic low-score data (`api/ai_training/gen_synth_image.py`)

The component CNN was blind below ~18 (real low-score drawings are scarce). Fixed by
**element-grounded generation**: `element ink = hand-defined region ∩ reference figure ink` → real
strokes with **exact** 60-component labels (type-aware position/accuracy degradation: affine jitter,
sinusoidal tremor, partial cuts, detail distortion, positional shift).
- `build_priors()` samples per-band P(present/accuracy/position) from **TELEFRED + LOWSCORER**
  (real low-score data sharpens the sparse low bands) → `element_priors.json`.
- CLI: `--preview N` (generate + CNN-read, no DB), `--insert N --score-min/-max` (DB rows
  `source_format='SYNTHETIC'`, `patient_id='SYNTH_*'`, forced to train), `--purge` (remove synthetic
  but **preserve validated** ones).
- `synthetic_score_based.py` is the OLD generator — **broken** (loads the removed `ReferenceImage` table).

**LOWSCORER** is the real-data complement: manually-rated real low-score copy figures (Total 1–16;
662 originals + a 50-image 16-point batch pending manual review), imported `validated=True` via
`load_lowscorer` + `import_unified.py --only-source LOWSCORER`.

> Related docs: `docs/SCORING_CRITERIA.md` (OCS-Plus rubric: 20 elements × P/A/P = 60) ·
> `MASTER_PLAN_SYN_TRAIN.md` (synthetic-training experiment log, per-band MAE results) ·
> `templates/README.md` (consolidated-base `labels.csv` schema & source mapping).

---

## 7. Components subsystem (60 sub-labels)

- **60 labels** = 20 elements × {Presence, Accuracy, Position}; `Total_Score` = their sum.
- `component_calibration.py` — post-hoc per-label F1-optimal thresholds + non-negative-least-squares
  score readout, written into model metadata (zero training cost).
- `component_heatmaps.py` — per-element localization (mean-ink-difference + Grad-CAM) →
  `data/visualizations/element_map/`.
- `data/element_definitions.json` — the hand-verified 20-element geometry (editable in
  `component_map.html`); consumed by the synthetic generator and the paint tool.
- Inference (`/api/ai-training/models/predict-single`) returns 60 probabilities + derived score;
  the editor UI shows a live confidence overlay and lets a human validate/correct (write-protect).

---

## 8. API surface (routers, mounted in `api/main.py`; full list at `/api/docs`)

| Router | Prefix | Key endpoints |
|--------|--------|---------------|
| `admin.py` | `/api/admin` | reset-database, cleanup-tmp, label-breakdown, cloudflare-status |
| `upload.py` | `/api` | normalize-image, check-duplicate |
| `training_data.py` | `/api` | save-drawn-image, training-data-images (+`only_missing`), .../original·processed·features, review-queue, training-data-stats |
| `ai_training_base.py` | `/api/ai-training` | dataset-info, available-features, training-readiness, start-training, stop-training, training-status |
| `ai_training_classification.py` | `/api/ai-training` | feature-distribution, custom-class-distribution, generate-classes, recalculate-classes |
| `ai_training_models.py` | `/api/ai-training` | models (GET/DELETE/rename), models/current (+set-current), metadata, test, predict-single, run-on-test-images, gradcam, component-map/rebuild, element-definitions (GET/POST) |
| `evaluator.py` | `/api/evaluator` | studies (GET/POST/DELETE), studies/{id}/queue, /rating, /analysis, /export.csv |

Plus `GET /api/health` and static `/api/visualizations`.

---

## 9. Frontend (`webapp/`, served by nginx; talks to `/api/*`)

| Page | Purpose |
|------|---------|
| `index.html` | Landing / navigation |
| `evaluate.html` | Unified upload-or-draw → predict → label & save (upload runs normalize-image first; draw sends raw canvas) |
| `upload.html`, `draw_testimage.html` | Redirect stubs → `evaluate.html?input=upload\|draw` |
| `ai_training.html` | Models & Training menu |
| `ai_training_overview.html` | Training status, model list, current-model marker, test run |
| `ai_training_train.html` | Launch/monitor a training job |
| `ai_training_data_view.html` | DB image browser: filter by source/label, inline predict, validate, label 60 components |
| `ai_training_data_upload.html` | Bulk CSV+image upload |
| `ai_training_evaluator.html` | Rater-reliability studies (create, queue, blind-rate, analyse, export) |
| `component_map.html` | Hand-edit element geometry → `element_definitions.json` |
| `run_test.html` · `reference.html` | Quick inference test · reference figure |
| `admin.html` · `docs.html` | Reset/cleanup/logs/tunnel · README/CHANGELOG links |

JS: `js/component_result.js` (render 60-component prediction), `js/component_editor.js` (element
region editing), `js/training_status.js` (poll training-status). CSS: `css/common.css`,
`css/ai_training_common.css`.

---

## 10. Config, logging, data dir

- **Config:** `api/config/training_config.yaml` (single source of truth: augmentation, training
  defaults per mode, synthetic, model, split). Access `from config import get_config;
  config.get("training.defaults.batch_size")`. Env overrides `NPSKETCH_<SECTION>_<KEY>`. Pydantic
  validation in `api/config/models.py` (**not** `api/models.py`, which is FastAPI I/O schemas).
- **Logging:** `from utils.logger import get_logger` → `/app/data/logs/training.log` (10 MB rotation,
  5 backups).
- **`data/`:** `npsketch.db` (+ `npsketch.pre*.db` backups) · `models/*.pth` + `*_metadata.json` ·
  `models/current_models.json` (pins the deployed model) · `element_definitions.json` ·
  `element_priors.json` · `visualizations/` · `logs/` · `tmp/`.

---

## 11. Known gotchas (see `CLAUDE.md` §9 for the full list)

- Router `ai_training.py` is split into `_base` / `_classification` / `_models`.
- Two `models.py` (FastAPI I/O vs config validation) — import the right one.
- Upload vs Draw predictions differ (upload normalizes first; draw sends the raw canvas).
- Element geometry is hand-verified — the "swapped elements" correlation report is an artifact; do
  not re-run the Hungarian re-save scripts.
- `synthetic_score_based.py` is broken (removed `ReferenceImage` table); use `gen_synth_image.py`.
- Root-level `start_*` / `grid_search_*` / `debug_*` / `test_*` scripts are ad-hoc scratch tooling,
  not part of the served API.
