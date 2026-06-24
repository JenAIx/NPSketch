# NPSketch v2.3

**CNN-based scoring of hand-drawn neuropsychological figures**

NPSketch trains and applies ResNet-18 CNN models to score hand-drawn OCS-Plus figures directly from
the image — predicting the Total_Score, a score class, or the full 20-element ×
Presence/Accuracy/Position component breakdown. (The earlier classical line-detection/template-matching
pipeline was removed in v2.0; see `CLAUDE.md`.)

---

## 🎯 Features

### Core Functionality
- **Draw or upload** a figure → score it with a trained model (component breakdown for component models)
- **Three training modes**: regression (Total_Score), classification (score classes), components (60 sub-labels)
- **Unified data base**: all sources consolidated into `templates/labels.csv` + `img/`, imported via
  `api/data_consolidation/import_unified.py`
- **Run Tests**: batch-evaluate a model over your drawn test images (predicted vs. expected)
- **Duplicate Detection**: SHA256 hash-based

### AI Training Pipeline
- **CNN Model Training**: ResNet-18, three training modes (see `CLAUDE.md` §7)
- **Three Training Modes**:
  - Regression: predict continuous scores (Total_Score, MMSE)
  - Classification: predict score ranges with custom class boundaries
  - Components (2026-06): predict the 60 OCS-Plus sub-labels (20 elements × Presence/Accuracy/Position);
    Total_Score = their sum — gives dense supervision in the sparse low-score range. TELEFRED-only (v1).
- **Unified data base**: all sources consolidated into `templates/labels.csv` + `img/`, imported via
  the single `api/data_consolidation/import_unified.py` (auto-detects red/black image style, dedups,
  stores the 60 sub-labels). See `templates/README.md`.
- **Interactive Class Creation**: Visual distribution preview with customizable class names and boundaries
- **Data Augmentation**: Realistic image transformations (rotation, translation, scaling, local warping)
- **Synthetic Bad Images**: Generate low-quality images to address data imbalance (NEW)
- **Training Data Management**: Upload, label, and manage training datasets
- **Quality Check Workflow**: Background checks with invalid-image filtering
- **Performance Metrics**: 
  - Regression: R², RMSE, MAE
  - Classification: Accuracy, F1-Score, Confusion Matrix
- **Single Image Prediction**: Use trained models for real-time prediction on uploaded drawings

### Data Extraction Tools
- **MAT Extractor**: Extract images from MATLAB `.mat` files (machine recordings)
- **OCS Extractor**: Extract red-pixel drawings from human rating images
- **Oxford Normalizer**: Normalize pre-processed PNG images with CSV labels
- **Line Normalization**: Consistent 2.00px line thickness via Zhang-Suen thinning + dilation
- **Auto-Cropping**: Intelligent content-aware cropping with minimal white space

### Web Interface
- **Modern UI**: Responsive design with drag & drop upload
- **Multiple Tools**: Reference editor, test image creator, evaluation viewer
- **AI Training Interface**: Dataset upload, model training, performance monitoring
- **Real-time Feedback**: Live processing status and results
- **Global CSS Architecture**: 
  - `common.css`: Base styles for all 13 pages
  - `ai_training_common.css`: AI-specific components
  - Responsive breakpoints: 1200px, 768px, 480px
- **Consistent Navigation**: Hierarchical back-links (no footer navigation)

---

## 🏗️ Architecture

```
npsketch/
├── api/                          # FastAPI Backend
│   ├── main.py                   # Application entry point
│   ├── database.py               # SQLAlchemy models
│   ├── routers/                  # API endpoints
│   │   ├── admin.py              # Admin & migrations
│   │   ├── upload.py             # Image upload & processing
│   │   ├── evaluations.py        # Evaluation management
│   │   ├── references.py         # Reference management
│   │   ├── test_images.py        # Test image management
│   │   ├── training_data.py      # Training data management
│   │   └── ai_training.py        # AI model training
│   ├── image_processing/         # Computer vision library
│   │   ├── line_detector.py      # Hough Transform line detection
│   │   ├── comparator.py         # Hungarian algorithm matching
│   │   ├── image_registration.py # Image alignment
│   │   └── utils.py              # Image preprocessing
│   ├── ai_training/              # AI/ML pipeline
│   │   ├── model.py              # ResNet-18 CNN model
│   │   ├── trainer.py            # Training orchestration
│   │   ├── dataset.py            # PyTorch data loaders
│   │   ├── data_loader.py        # Database to dataset
│   │   ├── data_augmentation.py  # Image augmentation
│   │   └── split_strategy.py     # Stratified train/test split
│   ├── mat_extraction/           # MATLAB file extractor
│   ├── ocs_extraction/           # OCS image extractor
│   ├── oxford_extraction/        # Oxford dataset extractor
│   │   ├── oxford_normalizer.py  # Image normalization
│   │   ├── oxford_db_populator.py # Database import
│   │   └── validate_oxford_*.py  # Validation scripts
│   └── requirements.txt          # Python dependencies
├── webapp/                       # Frontend (HTML/JS/CSS)
│   ├── css/                      # Stylesheets
│   │   ├── common.css            # Global styles (all pages)
│   │   └── ai_training_common.css # AI-specific styles
│   ├── js/                       # JavaScript modules
│   │   └── ai_training_preview_target_distribution.js
│   ├── index.html                # Landing page
│   ├── evaluate.html             # Upload OR draw → predict / label & save (merged)
│   ├── upload.html · draw_testimage.html  # redirect stubs → evaluate.html
│   ├── ai_training.html          # AI training menu
│   ├── ai_training_overview.html # Dataset overview & models
│   ├── ai_training_train.html    # Model training interface
│   ├── ai_training_data_view.html # View & label data
│   └── ai_training_data_upload.html # Upload MAT/OCS
├── data/                         # Persistent data (volume)
│   ├── npsketch.db               # SQLite database
│   ├── models/                   # Trained CNN models
│   └── visualizations/           # Generated images
├── nginx/                        # Reverse proxy config
├── docker-compose.yml            # Container orchestration
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- (Optional) Python 3.10+ for local development

### Installation

```bash
# 1. Start the application
docker compose up --build -d

# 2. Access the application
# Landing Page: http://localhost
# API Docs: http://localhost/api/docs
```

### Main Pages
- **http://localhost** - Landing page with stats
- **http://localhost/evaluate.html** - Upload or draw a figure → predict with a model, or label
  (Total_Score + 20 components) and add it to the training data. `?input=upload|draw` preselects the
  source tab. (Old `upload.html` / `draw_testimage.html` redirect here.)
- **http://localhost/ai_training.html** - AI training menu
- **http://localhost/ai_training_overview.html** - Dataset overview & trained models
- **http://localhost/ai_training_train.html** - Train new models
- **http://localhost/ai_training_data_view.html** - View, filter & label training data (filters:
  Only-Missing, Σ-only, Validated, Total_Score range, Last-24h; per-image component editing with
  image switch/crop; best-model card with ⭐ current-model promotion)
- **http://localhost/ai_training_evaluator.html** - Evaluator: inter-rater & model-vs-human
  reliability study (build a held-out set, blind-rate with N raters, analyse deviations)
- **http://localhost/ai_training_data_upload.html** - Upload training data (MAT/OCS)
- **http://localhost/api/docs** - Interactive API documentation

---

## 📖 Line Detection & Comparison

### 1. Reference Definition

**Manual approach for 100% accuracy:**

1. Navigate to http://localhost/reference.html
2. Click two points on the image to define each line
3. Lines are automatically categorized (Horizontal/Vertical/Diagonal)
4. Review and save to database

### 2. Image Upload & Processing

**3-Step Workflow:**

**STEP 1: Upload & Auto-Normalization**
- Auto-crop: Removes white space
- Scale to fit: 256×256 canvas with aspect ratio preserved
- Center: Places drawing centered on canvas
- Duplicate detection: SHA256 hash checking

**STEP 2: Manual Adjustments**
- Scale control (50-300%)
- Rotation control (-180° to +180°)
- Translation controls (arrow buttons)
- Overlay mode for comparison

**STEP 3: Auto Processing (Optional, Separate Options)**
- **Line Thinning** (✓ Default, Fast ~1 sec): Reduces lines to 1px
- **Registration** (☐ Off by default, Slow ~10 sec): Aligns to reference
  - Scipy differential evolution optimization
  - IoU threshold 0.15 (only applies if good match)
  - Limited to ±30° rotation, ±15px translation, 0.85-1.25x scale
  - Skipped automatically if poor alignment (prevents aggressive cropping)

### 3. Line Detection Algorithm

**Iterative Detection with Pixel Subtraction:**

```
1. Binary Threshold (127) → Black/white separation
2. ITERATION (up to 20 times):
   a) Hough Transform detects all lines
   b) Pick LONGEST line
   c) Check for duplicates (angle ±8°, position ±25px)
   d) Draw line on mask with 8px buffer
   e) DILATE mask (5×5 ellipse kernel)
   f) SUBTRACT from image → Line removed!
3. Repeat until no more lines or 12 lines detected
4. Final filter: Remove lines < 30px
```

**Multi-Pass Strategy:**
- Pass 1 (Iter 1-10): Strict threshold (18), longer lines (35px)
- Pass 2 (Iter 11-20): Relaxed threshold (10), shorter lines (25px)

### 4. Line Comparison

**Hungarian Algorithm for Optimal Matching:**

**Similarity Calculation:**
- Position distance (40% weight): Euclidean distance between midpoints
- Angle difference (30% weight): Angular difference (0-90°)
- Length ratio (30% weight): Relative length difference

**Metrics:**
- **Correct Lines**: Matched pairs (similarity ≥ 0.5)
- **Missing Lines**: Reference lines with no match
- **Extra Lines**: Detected lines with no match
- **Reference Match Score**: correct_lines / total_reference_lines

### 5. Evaluate & Label (`evaluate.html`)

> Note: the classical line-detection/template-matching algorithm was removed in v2.0. Scoring is
> AI-only — the sections above describing line detection are historical (see `CLAUDE.md`).

`evaluate.html` is the single page for working with one figure. It has two tab rows:

- **Top — image source:** **📤 Upload** (drag/drop → auto-normalize to 568×274 → scale/rotate/move
  correction) or **🎨 Draw** (pen/eraser canvas). `?input=upload|draw` preselects the tab.
- **Bottom — action:** **🤖 Predict** or **🏷️ Train / Label** — both operate on whichever source is active.

**🤖 Predict** — pick a trained model; one renderer handles all three modes:
- **Regression** (Total_Score): score + visualization bar, denormalized raw output.
- **Classification** (custom class): predicted class + confidence + probability bars.
- **Components** (60 sub-labels): full 20-element × Presence/Accuracy/Position breakdown with the
  derived Total_Score.

**🏷️ Train / Label** — set the 20 components (Total_Score = their sum) and a name, then save to the
training data via `/api/save-drawn-image`. The `source_format` is `UPLOAD` or `DRAWN` depending on the
active source. A saved-drawings browser lets you reload a drawing (image + labels), edit and re-save,
or delete it.

---

## 🤖 AI Training System

### Overview

Train CNN models (ResNet-18) to predict clinical features (e.g., Total_Score, MMSE) from neuropsychological drawings.

**Key Features:**
- **ResNet-18**: 11M+ parameters, ImageNet pre-trained backbone
- **Data Augmentation**: Multiplies dataset size by 5-10x
- **Stratified Splits**: Balanced train/validation distribution
- **Comprehensive Metrics**: R², RMSE, MAE, MAPE
- **Model Metadata**: Training config, dataset info, performance tracking

### Quick Start

**1. Prepare Training Data:**

Navigate to http://localhost/ai_training_data_upload.html and upload:
- MAT files (machine recordings)
- OCS PNG images (human ratings)
- Manual drawings with labels

**2. Train Model:**

```bash
# Via Web Interface
# Navigate to http://localhost/ai_training_train.html
# Select target feature (e.g., Total_Score)
# Configure training parameters
# Click "Start Training"

# Or via API
curl -X POST http://localhost/api/ai-training/start-training \
  -H "Content-Type: application/json" \
  -d '{
    "target_feature": "Total_Score",
    "train_split": 0.8,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "batch_size": 8
  }'
```

**3. Monitor Training:**

- Real-time progress updates
- Epoch-by-epoch loss tracking
- Train/validation split statistics
- Final performance metrics

### Data Augmentation (Enhanced System - 2026-01-12)

**Purpose:** Increase dataset size and reduce overfitting with guaranteed diversity

**NEW Enhanced Augmentation Strategy:**

From **1 original image** → **7 total images** (1 + 6 augmented):

**Step 0: Pre-Shrink (NEW)**
- Shrink content by 5% (0.95 factor)
- Center on white canvas
- Creates ~14px margins on all sides
- Enables proper translation and rotation

**Step 1: Diversity-Controlled Generation (NEW)**
- Generate 6 augmented variations
- Each augmentation checked for similarity to original (<95% similar)
- Pairwise uniqueness check (all augmentations <93% similar to each other)
- Progressive aggressiveness on retry (automatic parameter increase)
- Comprehensive metrics tracking

**Augmentation Mix:**
- **50% Rotation + Translation** (3 augmentations): ±5° rotation, ±3px translation
- **33% Warping Only** (2 augmentations): 15-20px displacement, 9 control points
- **17% Warping + Combined** (1 augmentation): Warping + light rotation/translation

**Enhanced Techniques:**

| Technique | Range | Purpose | Notes |
|-----------|-------|---------|-------|
| **Pre-Shrink** | 5% shrink | Creates margins for transformations | NEW - Applied to all images |
| **Rotation** | ±2-5° | Simulates paper tilt / camera angle | Increased from ±3°, excludes near-zero |
| **Translation** | ±3px | Simulates position shifts | Now works properly with margins |
| **Scaling** | 95-105% | Simulates size variations | Minimal use (only with rotation) |
| **Local Warping** | 15-20px @ 9 points | Simulates paper deformation | Variable displacement |

**Diversity Control Features (NEW):**

| Feature | Threshold | Purpose |
|---------|-----------|---------|
| **SSIM to Original** | <0.95 | Ensures each augmentation is meaningfully different from source |
| **SSIM Between Augs** | <0.93 | Ensures no redundant augmentations |
| **Progressive Retry** | 10 attempts | Increases parameters if augmentation too similar |
| **Quality Tracking** | Per-aug metrics | Logs similarity scores and diversity statistics |

**Local Warping Details:**
- Uses **Inverse Distance Weighting** (IDW) interpolation (vectorized, 100× faster than TPS)
- 9 control points in 3×3 grid (at 25%, 50%, 75% horizontal & vertical)
- Each point randomly displaced 15-20 pixels (variable)
- Edge-aware reduction (protects borders)
- Smooth interpolation with border protection

**Pipeline (Every Augmented Image):**
1. Pre-shrink by 5% and center (creates margins)
2. Generate augmentation with type-based strategy
3. Check similarity to original → Reject if ≥0.95
4. Check similarity to all other augmentations → Reject if ≥0.93
5. Retry with increased parameters if rejected (up to 10 attempts)
6. Re-binarize (threshold 175)
7. Line normalization (skeleton + 2px dilation)
8. Save with comprehensive metadata

**Example Output:**
- Original dataset: 20 images
- With augmentation: 140 images (20 original + 120 augmented)
- Effective multiplier: 7×
- Diversity guarantee: 100% (all augmentations unique)
- Avg similarity to original: 0.75-0.85 (excellent diversity)

**Quality Metrics:**
- All images: Binary (only black/white, no grayscale)
- Line thickness: Consistent 2px (guaranteed by pipeline)
- No fragmentation: <10 components per image
- Guaranteed diversity: SSIM-based filtering ensures no redundant variations
- Comprehensive logging: Every augmentation includes similarity metrics

### Synthetic low-score images (current)

The component CNN was blind below ~18 because real low-score drawings are scarce. Two complementary
sources now fill the low range:

- **LOWSCORER (real, 2026-06):** 662 manually-rated copy figures with Total_Score 1–15, imported as
  `source_format='LOWSCORER'`, `task_type='MANUAL'`, `validated=True`. Each element's 0–3 score is
  split into the three binary sub-labels (presence/accuracy/position; `2`→`[1,1,0]`).
- **Synthetic (element-grounded):** `api/ai_training/gen_synth_image.py` composes low-score figures
  from the hand-verified 20-element geometry (`data/element_definitions.json`, painted in
  `component_map.html`): for each element, `region ∩ reference ink = the element's real strokes`, then
  type-aware position/accuracy degradation. Labels are **exact by construction**. Per-score-band
  element priors (`build_priors`) are sampled from real TELEFRED + LOWSCORER drawings so the synthetic
  figures are structurally realistic. `--insert N` writes `source_format='SYNTHETIC'` rows; `--purge`
  removes them (but keeps any that were manually validated in the data-view).

Both feed component training via the `data_loader` filter `source IN ('TELEFRED','SYNTHETIC') OR
validated=True` (components required). The earlier score-targeted generator
(`synthetic_score_based.py`) is **removed/broken** — ignore older docs referring to "21 reference
features" or a 0–40 score-distribution table.

### Model Architecture

**ResNet-18 CNN:**
- **Input**: 568×274 grayscale images
- **Backbone**: ResNet-18 (pre-trained on ImageNet)
- **Head**: Fully connected layers with dropout
- **Output**: 
  - Regression: 1 neuron (continuous value)
  - Classification: N neurons (one per class)

**Training Details:**
- **Optimizer**: Adam
- **Learning Rate Scheduling**: ReduceLROnPlateau (automatic LR reduction when val_loss plateaus)
  - Factor: 0.5 (halve LR on reduction)
  - Patience: 5 epochs
  - Min LR: 1e-6
  - Automatically enabled, reduces LR when validation loss stagnates
- **Differential Learning Rates**: Different LRs for backbone vs head
  - Backbone LR: `learning_rate × 0.1` (10x smaller for pre-trained ResNet layers)
  - Head LR: `learning_rate × 1.0` (normal rate for new classification/regression head)
  - Preserves ImageNet pre-trained features while allowing head to learn quickly
- **Loss Functions**: 
  - Regression: MSE (Mean Squared Error)
  - Classification: CrossEntropyLoss (with automatic class weights for imbalanced data)
- **Class Weights**: Automatically calculated using inverse frequency (only for classification)
  - Formula: `weight = total_samples / (num_classes * class_count)`
  - Applied automatically when class imbalance detected
  - Stored in model metadata for reproducibility
- **Regularization**: 
  - Dropout: Configurable (default: 0.5)
  - Weight Decay (L2): Configurable (default: 0.0001)
  - Applied to all model parameters
- **Early Stopping**: Automatic training termination when validation loss stops improving
  - Patience: 15 epochs (configurable)
  - Min Delta: 0.001
  - Automatically restores best model
  - Can be disabled (patience = 0)
- **Output Activation**:
  - Regression: Sigmoid (ensures output in [0, 1] for normalized targets)
  - Classification: None (CrossEntropyLoss handles softmax internally)
- **Input Preprocessing**: Normalization to [0, 1]
- **Target Preprocessing**: 
  - Regression: Min-max normalization [0, 1]
  - Classification: Integer class labels (no normalization)

### Performance Metrics

**Regression Metrics:**
- **R² Score**: Coefficient of determination (0-1, higher better)
- **RMSE**: Root Mean Squared Error (lower better)
- **MAE**: Mean Absolute Error (lower better)
- **MAPE**: Mean Absolute Percentage Error (lower better)

**Classification Metrics:**
- **Accuracy**: Overall classification accuracy (0-1, higher better)
- **F1-Score (Macro)**: Harmonic mean of precision and recall
- **Precision & Recall**: Per-class and macro-averaged
- **Confusion Matrix**: Visual heatmap showing true vs predicted classes

**Example Outputs:**

**Regression:**
```json
{
  "train_metrics": {
    "r2_score": 0.856,
    "rmse": 2.341,
    "mae": 1.823
  },
  "val_metrics": {
    "r2_score": 0.792,
    "rmse": 2.876,
    "mae": 2.154
  }
}
```

**Classification:**
```json
{
  "train_metrics": {
    "accuracy": 0.863,
    "f1_score_macro": 0.841,
    "precision_score_macro": 0.856,
    "confusion_matrix": [[145, 5, 2], [8, 138, 5], [3, 7, 142]]
  },
  "val_metrics": {
    "accuracy": 0.812,
    "f1_score_macro": 0.798,
    "per_class_f1": [0.83, 0.79, 0.77]
  }
}
```

### Model Testing & Validation

**Test Trained Models:**

After training, validate model performance using the built-in testing functionality.

**Access:**
1. Navigate to http://localhost/ai_training_overview.html
2. Click "▼ Show Details" on any trained model
3. Click "🧪 Test Model" button

**What Happens:**
- Loads the original train/validation split from metadata
- Filters out synthetic images (not in database)
- Evaluates model on both train and validation sets
- Displays comprehensive metrics

**Test Results Display:**

**For Regression Models:**
```
R²: 0.917 (green if > 0.7)
RMSE: 4.80
MAE: 3.36
Tested on 182 validation samples
```

**For Classification Models:**
```
Accuracy: 86.3% (green if > 70%)
F1 (Macro): 84.1%
Precision: 85.6%
Per-Class Metrics expandable
Tested on 182 validation samples
```

**Important Notes:**
- ✅ Uses same train/val split as during training (reproducible)
- ✅ Filters out synthetic images automatically (not in DB)
- ✅ Applies same normalization as during training
- ⚠️ **For models trained with augmentation**: Test runs on original images (not augmented) for speed
  - This is faster and avoids timeout
  - Test metrics may differ from training validation metrics
  - Training validation metrics (shown in metadata) are the "true" performance
- ⚠️ Validation samples must still exist in database (unchanged since training)

**Why do test metrics differ?**
- Model trained on augmented data (6x multiplier: rotations, translations, warping)
- Test evaluates on original images only (no augmentation)
- This is a pragmatic trade-off: Speed vs. exact reproducibility
- **Use training validation metrics** (R²: 0.917) as the accurate performance measure

**Interpreting Results:**

**Good Model:**
- Regression: R² > 0.7, RMSE/MAE reasonable
- Classification: Accuracy > 70%, F1 > 0.7
- Train and Val metrics similar (no overfitting)

**Overfitting:**
- Train metrics much better than Val metrics
- Large gap indicates model memorized training data

**Underfitting:**
- Both Train and Val metrics poor
- Model too simple or not trained enough

### Classification Training

**Interactive Class Creation:**

1. Navigate to http://localhost/ai_training_train.html
2. Select "Total_Score" as target feature
3. Click the preview icon (👁️) next to sample count
4. Choose number of classes (2-5)
5. Customize class names (e.g., "Poor", "Fair", "Good")
6. Adjust class boundaries interactively
7. Save to database as `Custom_Class_N` feature

**Key Features:**
- **Equal-sample distribution**: Classes automatically balanced for equal sample counts
- **Editable boundaries**: Click on numbers to adjust score ranges
- **Custom names**: Rename classes with auto-suggestions
- **Visual preview**: Histogram and distribution bars
- **Non-overlapping ranges**: Ensures disjoint score intervals

**Example:**
```python
# Create 3-class classification for Total_Score [0-60]
# Automatic boundaries: [0-43], [44-51], [52-60]
# Custom names: "Poor", "Fair", "Good"
# Result: 919 samples → ~306 per class
```

### Component training sources (2026-06)

The component model trains on **TELEFRED + SYNTHETIC** plus **any human-`validated` row** of any
source — including the **LOWSCORER** set (662 manually-rated real low-score figures, imported
`validated=True`) and OXFORD/DRAWN rows labelled via the data-view modal or the Review queue. Rows still need
component sub-labels, so score-only images are auto-excluded; `pos_weight` is computed over the same
inclusion set. Machine-generated labels (ALGORITHM/OCS_MACHINE) stay out until validated. Set the
live model with the ⭐ **current-model** marker (see API above).

### Evaluator — inter-rater & model-vs-human reliability

`ai_training_evaluator.html` runs a small study to show the component model deviates from ground
truth no more than human raters do:

1. **Build** — sample **held-out** TELEFRED images (the current model's validation set) with
   ground-truth components, per score band (configurable count; default 20/band). Presentation
   order is **randomized** (not score-sorted) to avoid rater bias; ground truth is snapshotted.
2. **Rate** — N fixed raters (A/B/…) **blindly** score every image via the shared component editor
   (no model/GT shown). Ratings persist in the DB; pause/resume (first-unrated) and review.
3. **Analyse** — pick the AI-rater model (default current), then get a headline verdict, a deviation
   table (each rater + model vs GT: MAE/RMSE/bias/Pearson/ICC + sub-label agreement & Cohen's κ),
   inter-rater stats, Bland-Altman + most-critical-component charts, and CSV export. Total_Score is
   the sum of the 60 sub-labels for GT, raters and model alike (fair, apples-to-apples).

Tables: `evaluation_studies / items / ratings / model_runs` (auto-created). Tested in
`api/test_evaluator.py`.

---

## 🔬 Data Extraction Tools

### MAT File Extractor

**Purpose:** Extract images from MATLAB `.mat` files (machine tablet recordings)

**Features:**
- Extracts reference images and COPY/RECALL drawings
- Auto-cropping with configurable padding (default: 5px)
- Line normalization to 2.00px thickness
- Batch processing of directory trees

**Usage:**

```bash
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \
  --input /app/templates/bsp_ocsplus_202511/Machine_rater/matfiles \
  --output /app/data/output
```

**Output:**
- `PC56_REFERENCE_20251111.png` - Reference template
- `PC56_COPY_drawn_20251111.png` - Immediate copy
- `PC56_RECALL_drawn_20251111.png` - Memory recall

**Configuration (`mat_extractor.conf`):**

```json
{
  "canvas_width": 568,
  "canvas_height": 274,
  "auto_crop": true,
  "padding_px": 5
}
```

### OCS Image Extractor

**Purpose:** Extract red-pixel drawings from human rating images

**Features:**
- Red pixel detection (R≥200, G≤100, B≤100)
- Removes grids, reference figures, annotations
- Auto-cropping and line normalization
- Batch processing

**Usage:**

```bash
docker exec npsketch-api python3 /app/ocs_extraction/ocs_extractor.py \
  --input /app/templates/bsp_ocsplus_202511/Human_rater/imgs \
  --output /app/data/output
```

**Input Files:**
- `Park_16_COPY.png`
- `TeamD178_RECALL.png`

**Output:**
- `Park_16_COPY_ocs_20251111.png` (568×274, 2px lines)
- `TeamD178_RECALL_ocs_20251111.png` (568×274, 2px lines)

### Oxford Dataset Normalizer

**Purpose:** Normalize pre-processed PNG images with CSV labels (TotalScore)

**Features:**
- Normalizes images to 568×274px with 2.00px line thickness
- Auto-cropping with configurable padding (default: 5px)
- Matches MAT/OCS normalization process
- Batch processing of directory trees
- Database import with CSV label matching

**Usage:**

```bash
# DEPRECATED (2026-06): per-source import was replaced by the unified path.
# All data now lives in templates/labels.csv + img/ and is imported via:
docker exec -e PYTHONPATH=/app npsketch-api python3 /app/data_consolidation/import_unified.py
```

**Input:**
- Original PNG files in `imgs/` directory (various resolutions)
- CSV file: `Rater1_simple.csv` with ID, Cond (COPY/RECALL), TotalScore

**Output:**
- Normalized images: `imgs_normalized_568x274/{ID}_{Cond}.png` (568×274, 2px lines)
- Database entries with TotalScore from CSV

**Key Features:**
- Images already black lines (no color extraction needed)
- Labels from CSV file (TotalScore included automatically)
- Same normalization as MAT/OCS (568×274, 2px lines)
- Command-line import (not via web interface)

### Unified Output Format

All three extractors produce identical characteristics:
- **Resolution**: 568×274 pixels
- **Line Thickness**: 2.00px (normalized via Zhang-Suen + dilation)
- **Margins**: ~5-7px white border
- **Format**: Black lines on white background
- **Purpose**: CNN training compatibility

---

## 📡 API Endpoints

### Image Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload and evaluate image (algorithm) |
| `/api/normalize-image` | POST | Auto-crop + scale + center (same as training data) |
| `/api/register-image` | POST | Optional registration + optional thinning |
| `/api/check-duplicate` | POST | Check duplicates (both upload & training databases) |

### Evaluations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/evaluations/recent` | GET | List recent evaluations |
| `/api/evaluations/{id}` | GET | Get evaluation details |
| `/api/evaluations/{id}` | DELETE | Delete evaluation |
| `/api/evaluations/{id}/evaluate` | PUT | Add manual evaluation |

### References

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/references` | GET | List reference images |
| `/api/references/{id}/image` | GET | Get reference image |
| `/api/visualizations/{file}` | GET | Get visualization |

### Training Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/training-data/upload` | POST | Upload training data |
| `/api/extract-training-data` | POST | Extract MAT/OCS files (web interface) |
| `/api/training-data-evaluations` | GET | List training images |
| `/api/training-data-image/{id}/evaluate` | POST | Run line detection |
| `/api/training-data-image/{id}/ground-truth` | POST | Save ground truth |
| `/api/training-data-image-quality-check/start` | POST | Start background quality check |
| `/api/training-data-image-quality-check/status` | GET | Quality check progress |
| `/api/training-data-image/{id}/quality-status` | POST | Manually set quality status |

**Note:** Oxford dataset uses command-line scripts (`oxford_extraction/oxford_normalizer.py` + `oxford_extraction/oxford_db_populator.py`) instead of web interface.

### AI Training

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ai-training/dataset-info` | GET | Get dataset statistics (all source formats) |
| `/api/ai-training/available-features` | GET | List available features with stats |
| `/api/ai-training/feature-distribution/{feature}` | GET | Get distribution & histogram |
| `/api/ai-training/custom-class-distribution/{feature}` | GET | Get custom class info |
| `/api/ai-training/generate-classes` | POST | Generate balanced classes |
| `/api/ai-training/recalculate-class-counts` | POST | Recalculate after boundary change |
| `/api/ai-training/start-training` | POST | Start model training (regression or classification) |
| `/api/ai-training/training-status` | GET | Get training progress |
| `/api/ai-training/models` | GET | List trained models (incl. `is_current` flag) |
| `/api/ai-training/models/{filename}/metadata` | GET | Get model metadata |
| `/api/ai-training/models/current` | GET | The pinned "current" model per mode |
| `/api/ai-training/models/set-current` | POST | Pin a model as current for its mode |
| `/api/ai-training/models/test` | POST | Test model on validation set |
| `/api/ai-training/models/predict-single` | POST | Predict single image (`return_normalized=true` also returns the 568×274 normalized PNG) |
| `/api/ai-training/models/{filename}` | DELETE | Delete model |

**Current model marker:** `data/models/current_models.json` records the live model per mode
(components / regression / classification). It's honored by `precompute_predictions.py`, the
`evaluate.html` default selection, and the ⭐ UI in overview / data-view.

### Evaluator (rater reliability study)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/evaluator/studies` | POST | Build a study (held-out TELEFRED per score band; `n_raters`, `per_band`) |
| `/api/evaluator/studies` | GET | List studies + per-rater progress |
| `/api/evaluator/studies/{id}` | GET / DELETE | Study detail (per-band, progress) / delete |
| `/api/evaluator/studies/{id}/queue?rater=A` | GET | Blind ordered queue for a rater (no GT/model leaked) |
| `/api/evaluator/studies/{id}/rating` | POST | Upsert a rater's blind score (stored in DB) |
| `/api/evaluator/studies/{id}/analysis?model=` | GET | MAE/RMSE/bias/Pearson/ICC + per-sub-label agreement/κ + inter-rater + headline |
| `/api/evaluator/studies/{id}/export.csv?model=` | GET | Per-image GT / each rater / model totals |

### Example: Train Model via API

```bash
curl -X POST "http://localhost/api/ai-training/start-training" \
  -H "Content-Type: application/json" \
  -d '{
    "target_feature": "Total_Score",
    "train_split": 0.8,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "batch_size": 8
  }'
```

**Response:**

```json
{
  "success": true,
  "message": "Training started successfully",
  "training_id": "20251112_143211",
  "progress": {
    "epoch": 0,
    "total_epochs": 10,
    "split_info": {
      "train_samples": 24,
      "val_samples": 6,
      "split_strategy": "stratified"
    }
  }
}
```

---

## 🔧 Configuration

### Line Detection Parameters

Edit `api/image_processing/line_detector.py`:

```python
LineDetector(
    rho=1.0,                 # Distance resolution (pixels)
    theta=np.pi/180,         # Angle resolution (radians)
    threshold=18,            # Hough threshold
    min_line_length=35,      # Minimum line length (px)
    max_line_gap=35,         # Max gap between segments
    final_min_length=30      # Final filter threshold
)
```

### Image Registration

Edit `api/image_processing/image_registration.py`:

```python
ImageRegistration(
    max_rotation_degrees=30,     # Search range
    rotation_step=3,             # Angular resolution
    scale_range=(0.75, 1.30),    # Scale search range
    scale_step=0.05              # Scale resolution
)
```

### Comparison Tolerances

Edit `api/image_processing/comparator.py`:

```python
LineComparator(
    position_tolerance=120.0,    # Max position difference (px)
    angle_tolerance=50.0,        # Max angle difference (°)
    length_tolerance=0.8,        # Max length difference (ratio)
    similarity_threshold=0.5     # Min similarity for match
)
```

---

## 🗄️ Database Schema

### Core Tables

**reference_images**: Reference templates with manually defined lines
```python
- id: int (PK)
- name: str (unique)
- image_data: bytes (original)
- processed_image_data: bytes (normalized)
- lines_data: str (JSON array of lines)
- width, height: int
```

**uploaded_images**: Uploaded drawings for evaluation
```python
- id: int (PK)
- filename: str
- image_data: bytes (original)
- processed_image_data: bytes (normalized 256×256)
- image_hash: str (SHA256, for duplicate detection)
- uploader: str (optional)
```

**evaluation_results**: Comparison results
```python
- id: int (PK)
- image_id: int (FK)
- reference_id: int (FK)
- correct_lines, missing_lines, extra_lines: int
- similarity_score: float
- registration_info: str (JSON)
- user_evaluated, evaluated_correct, evaluated_missing, evaluated_extra: optional
```

### Training Tables

**training_data_images**: CNN-optimized training data
```python
- id: int (PK)
- patient_id: str (e.g., "PC56", "Park_16", "C0078")
- task_type: str (COPY, RECALL, REFERENCE)
- source_format: str (MAT, OCS, OXFORD, DRAWN, UPLOAD)
- original_file_data: bytes (original)
- processed_image_data: bytes (568×274, 2px lines)
- image_hash: str (SHA256 of ORIGINAL file for consistent duplicate checking)
- ground_truth_correct, ground_truth_extra: int (optional)
- features_data: str (JSON with nested Custom_Class structure)
- quality_check_status: str (valid/invalid/NULL)
- quality_check_date: datetime
```

**Example features_data:**
```json
{
  "Total_Score": 45,
  "Custom_Class": {
    "3": {
      "label": 1,
      "name_custom": "Fair",
      "name_generic": "Class_1 [44-51]",
      "boundaries": [0, 44, 52, 60]
    }
  }
}
```

---

## 🐳 Docker Details

### Services

- **nginx**: Frontend server + reverse proxy (Port 80)
- **api**: FastAPI backend with OpenCV, PyTorch (Port 8000)

### Volumes

- `./api:/app:rw` - API code (hot-reload)
- `./data:/app/data:rw` - Persistent data
- `./webapp:/usr/share/nginx/html:ro` - Frontend
- `./templates:/app/templates:ro` - Input data (optional)

### Common Commands

```bash
# Start services
docker compose up -d

# Rebuild after changes
docker compose up --build -d

# View logs
docker compose logs -f api

# Execute commands in container
docker exec npsketch-api python3 /app/your_script.py

# Restart services
docker compose restart

# Stop services
docker compose down
```

---

## 🧪 Testing

### Manual Testing

1. **Upload Test**: http://localhost/upload.html
2. **Reference Editor**: http://localhost/reference.html
3. **Draw Test Image**: http://localhost/draw_testimage.html
4. **Run Tests**: http://localhost/run_test.html

### API Testing

Use interactive docs: http://localhost/api/docs

### Creating Test Dataset

1. Create test images via draw tool
2. Score manually (correct/missing/extra lines)
3. Run automated tests
4. Compare expected vs actual results

---

## 📊 Performance & Metrics

### Line Detection Accuracy

- **Test Rating >90%**: Excellent
- **Test Rating 70-90%**: Good
- **Test Rating 50-70%**: Needs improvement
- **Test Rating <50%**: Poor

### AI Model Performance

**Good Performance:**
- R² Score > 0.7
- MAPE < 15%
- Validation performance close to training

**Signs of Overfitting:**
- Training R² >> Validation R²
- Large gap between train and validation loss

**Recommended Dataset Sizes:**
- Minimum: 50 images (with augmentation)
- Good: 100+ images
- Optimal: 200+ images

---

## 🛠️ Development

### Local Development (without Docker)

```bash
cd api
pip install -r requirements.txt
mkdir -p ../data/visualizations
mkdir -p ../data/logs
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access at: http://localhost:8000/api/docs

### Configuration

**Centralized Configuration:**
- All settings in `api/config/training_config.yaml`
- No hardcoded values in code
- Environment variable overrides supported

**Example:**
```bash
# Override batch size
export NPSKETCH_TRAINING_DEFAULTS_BATCH_SIZE=16

# Override log level
export NPSKETCH_LOGGING_LEVEL=DEBUG
```

**Configuration Structure:**
```yaml
training:
  defaults: {batch_size, learning_rate, num_epochs, ...}
augmentation: {rotation_range, translation_range, ...}
synthetic: {n_samples, complexity_levels, score_threshold, ...}
logging: {level, format, file rotation, ...}
```

### Logging

**Structured Logging:**
```python
from utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Training started")
logger.error("Error occurred", exc_info=True)
```

**Log Files:**
- Location: `/app/data/logs/training.log`
- Rotation: 10 MB max, 5 backups
- Format: `2025-12-29 14:30:15 - module - LEVEL - message`

### Type Safety

**Pydantic Models:**
```python
from config.models import TrainingConfig
from pydantic import ValidationError

try:
    config = TrainingConfig(**request_data)
    # Automatic validation!
except ValidationError as e:
    logger.error(f"Invalid config: {e}")
```

### Adding New Features

1. **New Router**: Add to `api/routers/`
2. **Include Router**: Import and include in `api/main.py`
3. **Database Changes**: Update `api/database.py`
4. **Frontend**: Add HTML to `webapp/`
5. **Configuration**: Add defaults to `api/config/training_config.yaml`
6. **Type Models**: Add Pydantic models to `api/config/models.py` (if needed)

---

## ✅ Recent Features

### Core & Processing
- [x] Iterative line detection with pixel subtraction
- [x] Hungarian algorithm for optimal matching
- [x] SHA256 duplicate detection (checks both upload & training databases)
- [x] Separate thinning & registration options (11x faster default upload)
- [x] Improved edge preservation (larger padding, intelligent border cleaning)
- [x] Consistent normalization across all data sources

### AI Training & Models
- [x] Dual training modes: Regression & Classification
- [x] Interactive class creation with distribution preview
- [x] Customizable class names and boundaries
- [x] Equal-sample class balancing algorithm
- [x] Confusion matrix visualization for classification
- [x] Single image prediction with trained models
- [x] AI model selector in upload interface
- [x] Custom class label display in predictions
- [x] Automatic denormalization for regression outputs
- [x] Local warping augmentation (TPS with 9 control points)
- [x] Hybrid augmentation strategy (3× global + 2× warp+global)
- [x] Post-augmentation line normalization (consistent 2px lines)
- [x] Optimal binarization threshold (175) for augmented images
- [x] **NEW (2025-12-29): Synthetic bad images generation for data imbalance**
- [x] **NEW (2025-12-29): Real bad line extraction (105 lines from 12 images)**
- [x] **NEW (2025-12-29): Multi-complexity synthetic generation (5 levels)**

### Data Management
- [x] MAT/OCS/OXFORD extraction tools with auto-cropping
- [x] Line thickness normalization (2.00px via Zhang-Suen)
- [x] OXFORD dataset hash correction script
- [x] Training data management interface with feature labeling
- [x] Dynamic dataset statistics (all source formats)

### Architecture & UI
- [x] Modular API architecture (3 focused routers for AI training)
- [x] Global CSS system (common.css + ai_training_common.css)
- [x] Responsive design (4 breakpoints: 1200px, 768px, 480px)
- [x] Consistent navigation hierarchy across all 13 pages
- [x] Unified text colors for readability (#333 on light backgrounds)

### Code Quality & Infrastructure (2025-12-29)
- [x] **NEW: Type Safety with Pydantic Models** (automatic validation)
- [x] **NEW: Centralized Configuration** (training_config.yaml)
- [x] **NEW: Structured Logging** (logging module with file rotation)
- [x] **NEW: Config Loader** (singleton pattern with env overrides)
- [x] **NEW: Zero hardcoded values** (all in config.yaml)

---

## 📝 License

MIT License - feel free to use and modify for your projects.

---

## 👤 Author

**Stefan Brodoehl**  
Date: October-December 2025, January–June 2026  
Version: 2.3.0

---

## 📞 Support

For issues or questions:
- Check API documentation: http://localhost/api/docs
- Review code comments and docstrings
- Inspect database: `sqlite3 data/npsketch.db`
- Check logs: `docker compose logs -f api`

---

**Happy Sketching & Training! 🎨🤖**
