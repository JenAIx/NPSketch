# NPSketch v2.1 — Documentation

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
  Position); `Total_Score` = their sum. Dense supervision in the sparse low-score range. TELEFRED-only (v1).

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
- **`/ai_training_data_view.html`** — browse training images; per-item Total_Score + 🧩 component
  labels; sort by score; the *Only Missing* filter surfaces unlabelled images.
  (Bulk import is CLI-only via `import_unified.py`; the old `ai_training_data_upload.html`
  redirects here.)
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

Every image (import, upload, and augmentation output) is brought to one canonical format so the CNN
sees consistent input:

1. **Auto-crop** to the ink bounding box, then pad with ~5–7 px margin.
2. **Rescale** to **568 × 274 px** (the working resolution; AR preserved by the crop+pad).
3. **Binarize** (threshold ~175) → pure black lines on white.
4. **Line-thickness normalization** to **2.00 px**: Zhang-Suen skeletonization down to a 1-px
   skeleton, then a controlled dilation back to 2 px — so stroke width is identical regardless of
   pen/scan thickness (`api/line_normalizer.py`).

The **Upload** source additionally runs `/api/normalize-image` (auto-crop + rescale + center) before
this, which can shift margins/line-scale slightly versus the raw **Draw** canvas — keep that in mind
when comparing predictions of the same figure drawn vs uploaded.

At training time images are downscaled once more to the **284 × 137** model input (half-res, AR
preserved) and fed as a single grayscale channel normalized to `[0, 1]`.

---

## Data augmentation

Augmentation runs **per training split** (the validation set is augmented from val patients only, so
no augmented copy ever crosses the split). It is **diversity-controlled** rather than purely random —
each augmented variant must be visually different enough from the original and from the other
augmentations, otherwise it is retried.

Pipeline per image:

1. **Pre-shrink 5 %** → enlarges the margin to ~14 px so rotations/warps don't clip the figure.
2. **Diversity-controlled transforms**, drawn from a fixed mix:
   - **50 %** rotation + translation
   - **33 %** local warp only
   - **17 %** warp + combined transforms
3. **SSIM filter** — reject if too similar to the original (SSIM ≥ 0.95) or to a sibling
   augmentation (≥ 0.93); progressive parameter escalation on retry.
4. **Re-binarize** (threshold 175) and **re-normalize line thickness** to 2 px.

| Transform | Range |
|-----------|-------|
| Rotation | ±5° |
| Translation | ±3 px |
| Local warp | IDW, 9 control points, 15–20 px displacement |

Result ≈ **6 augmentations per image (~7× total)**. Ranges live in
`api/config/training_config.yaml` (`augmentation.*`) — no hard-coded values.

## Model architecture

- **Backbone**: ResNet-18, ImageNet-pretrained, first conv adapted to **1 grayscale channel**
  (RGB weights averaged).
- **Head**: `Linear(512→256) → ReLU → Dropout(0.5) → Linear(256→N)`. `N` = 1 (regression),
  #classes (classification), or **60** (components). A `Sigmoid` is appended only for the legacy
  normalized-regression head.
- **Input**: 284 × 137 grayscale, `[0, 1]`.

## Training modes

Selected by `target_feature` in the training job; all three share the backbone, the patient-level
split, augmentation, the 284×137 input and best-checkpoint early stopping.

- **Regression** (`Total_Score`): **linear** output + **MSE**, min-max target normalization,
  `WeightedRandomSampler` over score bins to counter the high-score skew. (The old sigmoid head
  saturated on the skewed distribution and was dropped.) Metrics: R² / RMSE / MAE / MAPE + per-decade.
- **Classification** (`Custom_Class_<N>`): softmax + CrossEntropy, inverse-frequency class weights.
  Metrics: accuracy / macro-F1 / precision / recall / confusion matrix.
- **Components** (`Components`, 60 sub-labels): `BCEWithLogitsLoss` with **per-label `pos_weight`**
  (each sub-label balanced independently); `Total_Score` = sum of the predicted sub-labels.
  **TELEFRED-only (v1).** Metrics: per sub-label / aspect / component F1 **and** the derived-score
  R² / RMSE / MAE + per-decade.

### Shared defaults

ResNet-18 (ImageNet) · Adam · batch 8 · `ReduceLROnPlateau` (×0.5, patience 5, min 1e-6) ·
differential LR (backbone ×0.1) · dropout 0.5 · weight-decay 1e-4 · early stopping with
best-checkpoint restore · **patient-level stratified split** (`stratified_group_split`, hard
zero-overlap assertion between train and val). The split groups by `patient_id` so a patient's COPY
and RECALL never straddle the split — this is what keeps the validation metrics honest.

### Synthetic score-based images

To counter the sparse low-score range, `api/ai_training/synthetic_score_based.py` can generate images
targeting specific scores (0–40). They are added **before** the split and augmented like real data.

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
| `task_type` | COPY · RECALL · DRAWN · UPLOAD |
| `source_format` | TELEFRED · OXFORD · ALGORITHM · OCS_MACHINE · DRAWN · UPLOAD |
| `original_file_data`, `processed_image_data` | original BLOB + normalized 568×274 PNG |
| `image_hash` | SHA256 of the original (dedup) |
| `features_data` | JSON `{Total_Score, components:{presence[20],accuracy[20],position[20]}}`; **NULL = unlabelled** |
| `uploaded_at`, `session_id`, `extraction_metadata` | provenance |

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
- `GET /api/training-data-images` — list (returns `total_score`, `has_components`; `only_missing` filter).
- `GET /api/training-data-image/{id}` · `/{id}/features` · `/{id}/original` · `DELETE`.
- `GET /api/ai-training/dataset-info` · `/available-features` · `/feature-distribution/{feature}`.
- `POST /api/ai-training/start-training` · `GET /training-status`.
- `GET /api/ai-training/models` · `/models/{file}/metadata` · `POST /models/predict-single` ·
  `POST /models/run-on-test-images`.
- `POST /api/admin/reset-database` · `/cleanup-tmp`.

---

## Quick start

```bash
docker compose up --build -d          # start (nginx :80, api :8000)
# open http://localhost
docker compose logs -f api            # follow backend logs
```

Container: `npsketch-api` · workdir `/app` · DB `/app/data/npsketch.db`.
