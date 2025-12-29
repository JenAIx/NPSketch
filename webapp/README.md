# NPSketch v1.0

**Automated Line Detection & AI Training for Hand-Drawn Images**

NPSketch is a computer vision and machine learning application that automatically compares hand-drawn images to reference templates, provides detailed feedback on drawing accuracy, and trains CNN models to predict clinical features from neuropsychological drawings.

---

## 🎯 Features

### Core Functionality
- **Automated Line Detection**: OpenCV Hough Transform with iterative pixel subtraction
- **Smart Comparison**: Hungarian algorithm for optimal line matching
- **Visual Feedback**: Color-coded visualizations showing matches and differences
- **Dual Analysis Methods**: 
  - Algorithm-based: Line detection and matching
  - AI-based: Trained CNN model prediction (regression or classification)
- **Duplicate Detection**: SHA256 hash-based checking across both upload and training databases
- **Optimized Processing**: Separate thinning and registration options for faster uploads

### AI Training Pipeline
- **CNN Model Training**: ResNet-18 for both regression and classification tasks
- **Dual Training Modes**: 
  - Regression: Predict continuous scores (Total_Score, MMSE)
  - Classification: Predict score ranges with custom class boundaries
- **Interactive Class Creation**: Visual distribution preview with customizable class names and boundaries
- **Data Augmentation**: Realistic image transformations (rotation, translation, scaling)
- **Training Data Management**: Upload, label, and manage training datasets
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
│   ├── upload.html               # Upload & analyze (Algorithm or AI)
│   ├── reference.html            # Reference line editor
│   ├── ai_training.html          # AI training menu
│   ├── ai_training_overview.html # Dataset overview & models
│   ├── ai_training_train.html    # Model training interface
│   ├── ai_training_data_view.html # View & label data
│   ├── ai_training_data_upload.html # Upload MAT/OCS
│   └── training_evaluations.html # Algorithm evaluation
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
- **http://localhost/upload.html** - Upload & analyze drawings (Algorithm or AI Model)
- **http://localhost/reference.html** - Define reference lines
- **http://localhost/ai_training.html** - AI training menu
- **http://localhost/ai_training_overview.html** - Dataset overview & trained models
- **http://localhost/ai_training_train.html** - Train new models
- **http://localhost/ai_training_data_view.html** - View & label training data
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

### 5. AI Model Prediction (Alternative to Algorithm)

**Upload & Analyze with Trained Models:**

In `upload.html`, users can choose between:

**Method 1: Algorithm (Traditional)**
- Line detection + Hungarian matching
- Shows: Correct, Missing, Extra Lines, Similarity Score

**Method 2: AI Model (Neural Network)**
- Select trained CNN model from dropdown
- **Regression**: Predicts continuous score (e.g., 37.5 out of 60)
  - Shows score visualization bar with denormalized value
  - Displays raw model output and normalization info
- **Classification**: Predicts score range/class (e.g., "Good [52-60]")
  - Shows predicted class with confidence percentage
  - Displays probability bars for all classes with custom names
  - Highlights predicted class with checkmark

**Features:**
- Model dropdown shows all available trained models
- Displays model metadata (target feature, mode, accuracy/MAE)
- Real-time prediction (~0.5 sec)
- Supports both regression and classification models
- Uses custom class names (e.g., "Poor", "Fair", "Good")

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

### Data Augmentation

**Purpose:** Increase dataset size and reduce overfitting

**Augmentation Techniques:**

| Technique | Range | Purpose |
|-----------|-------|---------|
| Rotation | ±1-3° | Simulates paper tilt |
| Translation | ±5-10px | Simulates position shifts |
| Scaling | 95-105% | Simulates size variations |

**Configuration:**

```python
augmentation_config = {
    'rotation_range': (-3, 3),      # degrees
    'translation_range': (-10, 10),  # pixels
    'scale_range': (0.95, 1.05),    # scale factor
    'num_augmentations': 5           # per image
}
```

**Example Output:**
- Original dataset: 20 images
- With 5x augmentation: 120 images (20 original + 100 augmented)
- Effective multiplier: 6x

**API Usage:**

```python
from ai_training.data_loader import TrainingDataLoader
from database import SessionLocal

db = SessionLocal()
loader = TrainingDataLoader(db)

# Prepare augmented dataset
stats, output_dir = loader.prepare_augmented_training_data(
    target_feature='Total_Score',
    train_split=0.8,
    augmentation_config={
        'num_augmentations': 5
    },
    output_dir='/app/data/ai_training_data'
)

print(f"Train: {stats['train']['total']} images")
print(f"Val: {stats['val']['total']} images")
```

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
- **Loss Functions**: 
  - Regression: MSE (Mean Squared Error)
  - Classification: CrossEntropyLoss
- **Regularization**: Dropout (0.5)
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
# Step 1: Normalize images
docker exec npsketch-api python3 /app/oxford_extraction/oxford_normalizer.py \
  /app/templates/training_data_oxford_manual_rater_202512/imgs \
  /app/templates/training_data_oxford_manual_rater_202512/imgs_normalized_568x274

# Step 2: Import to database
docker exec npsketch-api python3 /app/oxford_extraction/oxford_db_populator.py
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
| `/api/ai-training/models` | GET | List trained models |
| `/api/ai-training/models/{filename}/metadata` | GET | Get model metadata |
| `/api/ai-training/models/test` | POST | Test model on validation set |
| `/api/ai-training/models/predict-single` | POST | Predict single image |
| `/api/ai-training/models/{filename}` | DELETE | Delete model |

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
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access at: http://localhost:8000/api/docs

### Adding New Features

1. **New Router**: Add to `api/routers/`
2. **Include Router**: Import and include in `api/main.py`
3. **Database Changes**: Update `api/database.py`
4. **Frontend**: Add HTML to `webapp/`

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

---

## 📝 License

MIT License - feel free to use and modify for your projects.

---

## 👤 Author

**Stefan Brodoehl**  
Date: October-November 2025  
Version: 1.0

---

## 📞 Support

For issues or questions:
- Check API documentation: http://localhost/api/docs
- Review code comments and docstrings
- Inspect database: `sqlite3 data/npsketch.db`
- Check logs: `docker compose logs -f api`

---

**Happy Sketching & Training! 🎨🤖**
