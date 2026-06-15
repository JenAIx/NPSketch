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
- **`/ai_training_data_upload.html`** — import source data.
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

**Normalized image format:** 568×274 px RGB PNG, black lines on white, line thickness ~2 px,
~5–7 px margin.

---

## Training (shared defaults)

ResNet-18 (ImageNet backbone) · 284×137 model input (downscaled) · Adam · batch 8 ·
ReduceLROnPlateau · differential LR (backbone ×0.1) · dropout 0.5 · early stopping ·
patient-level stratified split (zero patient overlap between train/val) · diversity-controlled
augmentation (~7× total).

- **Regression**: linear head + MSE, min-max target normalization, score-bin resampling.
- **Classification**: softmax + CrossEntropy, inverse-frequency class weights.
- **Components**: `BCEWithLogitsLoss` + per-label `pos_weight`; metrics per sub-label/aspect/component
  F1 + derived-score R²/RMSE/MAE.

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
