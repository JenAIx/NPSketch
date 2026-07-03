# NPSketch v2.3 — Documentation

**AI scoring of hand-drawn neuropsychological figures (OCS-Plus copy & recall).**

NPSketch trains and applies **ResNet-18 CNN** models that score a drawing directly from the image —
no hand-crafted line detection. The earlier classical pipeline (Hough lines, template matching,
reference editor) was removed in v2.0.

---

## What it does

Three training/prediction modes share one ResNet-18 backbone:

- **Regression** — predict the continuous `Total_Score` (0–60).
- **Classification** — predict a custom score class (e.g. *Poor / Fair / Good*).
- **Components** — predict the **60 OCS-Plus sub-labels** (20 elements × Presence / Accuracy /
  Position); `Total_Score` = their sum. Dense supervision, incl. the sparse low-score range
  (real LOWSCORER + element-grounded synthetic).

---

## Architecture

**Stack:** FastAPI (Python 3.10+) · SQLite (`npsketch.db`, single table `training_data_images`) ·
PyTorch (ResNet-18) · OpenCV / PIL · static HTML/JS frontend served by nginx.

Two Docker services (`docker-compose.yml`):

| Service | Container | Port | Role |
|---------|-----------|------|------|
| nginx | `npsketch-nginx` | 80 | serves the frontend + reverse-proxies `/api` |
| api | `npsketch-api` | 8000 | FastAPI backend (OpenCV, PyTorch) |

Volumes: `./api → /app` (RW, hot-reload) · `./data → /app/data` (RW: DB, models, visualizations,
logs) · `./templates → /app/templates` (RO: source data) · `./webapp → nginx html` (RO).

---

## Pages

- **`/evaluate.html`** — the main workbench for a single figure.
  - *Top tabs* choose the image **source**: 📤 **Upload** (drag/drop → auto-normalize to 568×274 →
    scale/rotate/move correction) or 🎨 **Draw** (pen/eraser canvas).
  - *Bottom tabs* choose the **action**: 🤖 **Predict** (pick a model → score / class / component
    breakdown) or 🏷️ **Train / Label** (set the 20 components, `Total_Score` = sum, save to the
    training data; includes a saved-drawings browser to reload/edit/delete).
  - `?input=upload|draw` preselects the source tab. The last-used model is remembered.
- **`/ai_training.html`** — AI training menu (overview, train, component map).
- **`/ai_training_overview.html`** — dataset stats, available features, saved models, and the
  **Component Map** link.
- **`/ai_training_train.html`** — configure and start a training run (Total_Score, custom classes,
  or Components) directly from the UI; live progress.
- **`/ai_training_data_view.html`** — browse/filter training images; per-item Total_Score + 🧩
  component labels; filters: *Only Missing*, *Σ-only*, *Validated*, *Total_Score range*, *Last 24h*
  (+ a matched/total chip). Click a row to view/edit components (image switch Processed/Original +
  crop, validated 🔒 lock). Stats cards incl. a **Best Model** card that opens model details and
  can set the ⭐ **current model**. (Bulk import is CLI-only via `import_unified.py`.)
- **`/ai_training_evaluator.html`** — **Evaluator**: build a held-out study, blind-rate it with N
  raters (component editor, no model hint), then an analysis dashboard (deviation vs ground truth,
  inter-rater ICC/κ, Bland-Altman, most-critical components) + CSV export.
- **`/component_map.html`** — explainability: where each of the 20 elements sits on the reference
  figure (data-driven heatmaps **and** the component model's Grad-CAM attention). See below.
- **`/run_test.html`** — batch-evaluate a model over the drawn test images (predicted vs expected).
- **`/admin.html`** — reset database, clean up temp files.
- **`/docs.html`** — this document.

> Note: `upload.html` and `draw_testimage.html` now redirect to `evaluate.html`. The old
> `reference.html` (algorithm-era reference editor) was removed — it no longer exists.

---

## Data pipeline

All training data lives in one consolidated base: `templates/labels.csv` + `templates/img/`, imported
through the **single** path `api/data_consolidation/import_unified.py` (auto-detects red vs black ink
style, dedups by SHA256, skips blank scans, writes the 60 sub-labels into `features_data.components`).
Sources: `TELEFRED`, `OXFORD`, `ALGORITHM`, `OCS_MACHINE`.

**Unscored placeholders:** rows marked `label_status='zero'` in the source are *unscored placeholders*
(not genuine zero-score drawings). They are imported as images but with **no features**
(`features_data = NULL`), so they are excluded from training yet appear under *Only Missing* for later
labelling.

### Preprocessing & normalization

Every image (import, upload, augmentation output) is brought to one canonical format so the CNN sees
consistent input: **auto-crop → 568 × 274 px → binarize (≈175) → 2.00 px line thickness** (Zhang-Suen
skeletonize + controlled dilation, `api/line_normalizer.py`). At training time it is downscaled once
more to the **284 × 137** grayscale model input, `[0, 1]`.

> The **Upload** source runs `/api/normalize-image` (auto-crop + rescale + center) first, which can
> shift margins/line-scale slightly versus the raw **Draw** canvas — worth knowing when the same
> figure predicts differently drawn vs uploaded.

---

## Model & training

One **ResNet-18** backbone (ImageNet-pretrained, adapted to 1 grayscale channel) with a swappable head
serves three modes, selected by `target_feature`; all share the patient-level split, augmentation, the
284×137 input and best-checkpoint early stopping.

- **Regression** (`Total_Score`) — linear output + MSE, score-bin `WeightedRandomSampler`. Metrics
  R²/RMSE/MAE/MAPE.
- **Classification** (`Custom_Class_<N>`) — softmax + CrossEntropy, inverse-frequency class weights.
  Metrics accuracy/macro-F1/precision/recall.
- **Components** (60 sub-labels = 20 elements × Presence/Accuracy/Position) — `BCEWithLogitsLoss` with
  per-label `pos_weight`; `Total_Score` = sum of the predicted sub-labels. Trains on **TELEFRED +
  SYNTHETIC + any `validated` row** (incl. the **LOWSCORER** set and validated OXFORD/DRAWN). Metrics
  per sub-label/aspect/component F1 + derived-score R²/RMSE/MAE.

**Data augmentation** is diversity-controlled (SSIM-filtered, not purely random) and runs per split so
no augmented copy crosses train↔val: pre-shrink 5 % → rotation ±5° / translation ±3 px / IDW local
warp → ~6 augs per image (~7× total). **Shared defaults:** Adam · batch 8 · ReduceLROnPlateau ·
differential LR (backbone ×0.1) · dropout 0.5 · weight-decay 1e-4. The split groups by `patient_id`
(COPY+RECALL never straddle it) with a hard zero-overlap assertion — that's what keeps the metrics
honest. Exact numbers live in `api/config/training_config.yaml` (no hard-coded values).

### Low-score data (real + synthetic)

The component model was weak in the sparse low-score range, addressed two ways:
- **LOWSCORER** — manually-rated **real** low-score figures (`Total_Score` 1–16), imported
  `validated=True` (`source_format='LOWSCORER'`, `task_type='MANUAL'`).
- **Synthetic (element-grounded)** — `api/ai_training/gen_synth_image.py` composes figures from the
  hand-verified 20-element geometry (`element_definitions.json`): `region ∩ reference ink = real
  strokes`, with **exact** 60-component labels and per-band element priors sampled from real
  TELEFRED + LOWSCORER. `--insert` / `--purge` manage the `SYNTHETIC` rows (validated ones survive a
  purge).

### Reference metrics (clean component model)

Validation (patient-level held-out) of the current TELEFRED component model: macro-F1 ≈ 0.96,
sub-label accuracy ≈ 0.94; derived `Total_Score` MAE ≈ 2.7, R² ≈ 0.76.

---

## Configuration & logging

- **Central config**: `api/config/training_config.yaml` is the single source of truth (augmentation
  ranges, batch size, LR, early-stopping, model input, synthetic-image settings). Access via
  `from config import get_config`.
- **Env overrides**: `NPSKETCH_<SECTION>_<KEY>` (e.g. `NPSKETCH_TRAINING_DEFAULTS_BATCH_SIZE=16`,
  `NPSKETCH_LOGGING_LEVEL=DEBUG`).
- **Type safety**: Pydantic models in `api/config/models.py`.
- **Logs**: `/app/data/logs/training.log` (10 MB rotation, 5 backups).

## Database

Single table **`training_data_images`** (SQLite, `data/npsketch.db`):

| Column | Notes |
|--------|-------|
| `uid`, `patient_id` | source-prefixed; `patient_id` groups COPY+RECALL (e.g. `TF-257`) |
| `task_type` | COPY · RECALL · MANUAL |
| `source_format` | TELEFRED · OXFORD · ALGORITHM · OCS_MACHINE · LOWSCORER · SYNTHETIC (+ legacy DRAWN/UPLOAD) |
| `original_file_data`, `processed_image_data` | original BLOB + normalized 568×274 PNG |
| `image_hash` | SHA256 of the original (dedup) |
| `features_data` | JSON `{Total_Score, components:{presence[20],accuracy[20],position[20]}}`; **NULL = unlabelled** |
| `validated`, `validated_at` | human write-protect; component training also includes any validated row |
| `model_prediction` | best/current model's prediction, stored in parallel (never overwrites `features_data`) |
| `uploaded_at`, `session_id`, `extraction_metadata` | provenance |

Plus the **Evaluator** tables: `evaluation_studies`, `evaluation_items` (image set + GT snapshot),
`evaluation_ratings` (each rater's blind score), `evaluation_model_runs` (cached model run per study).

---

## Component Map (explainability)

`/component_map.html` localizes the 20 elements on the reference figure two complementary ways:

- **Data-driven** — `mean(ink | presence=1) − mean(ink | presence=0)`, computed **within each
  Total_Score band** and averaged. Stratifying removes the confound that an absent element usually
  just means a sparser drawing, so each element localizes to its own region (e.g. E17 → circle,
  E14 → star, E20 → right rectangle).
- **Grad-CAM** — attention of the component CNN's presence logit on the reference figure, i.e. where
  the trained model looks. A quick check that it learned the right regions.

Regenerate the maps with:

```bash
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/ai_training/component_heatmaps.py
```

Output (overlays + labeled composites + `manifest.json`) lands in
`data/visualizations/element_map/`, served at `/api/visualizations/element_map/`.

---

## Key API endpoints

Full interactive list: `http://localhost/api/docs`.

- `GET /api/health` — status + app version.
- `POST /api/normalize-image`, `POST /api/check-duplicate` — upload preprocessing.
- `POST /api/save-drawn-image` — save a drawn/uploaded image as training data (accepts `total_score`,
  `components`, `source_format` = DRAWN/UPLOAD).
- `GET /api/training-data-images` — list (returns `total_score`, `has_components`, `validated`;
  filters `only_missing` / `scored_only` / `validated` / `score_min`+`score_max` / `recent_hours`).
- `GET /api/training-data-image/{id}` · `/{id}/features` · `/{id}/original` · `/{id}/processed` ·
  `/{id}/crop-and-reprocess` · `DELETE`.
- `GET /api/ai-training/dataset-info` · `/available-features` · `/feature-distribution/{feature}`.
- `POST /api/ai-training/start-training` · `GET /training-status`.
- `GET /api/ai-training/models` (incl. `is_current`) · `/models/{file}/metadata` ·
  `POST /models/predict-single` (`return_normalized`) · `GET /models/current` · `POST /models/set-current`.
- **Evaluator:** `POST/GET /api/evaluator/studies` · `GET/DELETE /studies/{id}` ·
  `GET /studies/{id}/queue?rater=` · `POST /studies/{id}/rating` · `GET /studies/{id}/analysis?model=` ·
  `GET /studies/{id}/export.csv`.
- `POST /api/admin/reset-database` · `/cleanup-tmp`.

---

## Quick start

```bash
docker compose up --build -d          # start (nginx :80, api :8000)
# open http://localhost
docker compose logs -f api            # follow backend logs
```

Container: `npsketch-api` · workdir `/app` · DB `/app/data/npsketch.db`.
