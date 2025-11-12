# NPSketch v1.0

**Automated Line Detection & AI Training for Hand-Drawn Images**

NPSketch is a computer vision and machine learning application that automatically compares hand-drawn images to reference templates, provides detailed feedback on drawing accuracy, and trains CNN models to predict clinical features from neuropsychological drawings.

---

## 🎯 Features

### Core Functionality
- **Automated Line Detection**: OpenCV Hough Transform with iterative pixel subtraction
- **Smart Comparison**: Hungarian algorithm for optimal line matching
- **Visual Feedback**: Color-coded visualizations showing matches and differences
- **Duplicate Detection**: SHA256 hash-based duplicate checking

### AI Training Pipeline
- **CNN Model Training**: ResNet-18 for predicting clinical features from drawings
- **Data Augmentation**: Realistic image transformations (rotation, translation, scaling)
- **Training Data Management**: Upload, label, and manage training datasets
- **Performance Metrics**: R², RMSE, MAE, MAPE tracking

### Data Extraction Tools
- **MAT Extractor**: Extract images from MATLAB `.mat` files (machine recordings)
- **OCS Extractor**: Extract red-pixel drawings from human rating images
- **Line Normalization**: Consistent 2.00px line thickness via Zhang-Suen thinning + dilation
- **Auto-Cropping**: Intelligent content-aware cropping with minimal white space

### Web Interface
- **Modern UI**: Responsive design with drag & drop upload
- **Multiple Tools**: Reference editor, test image creator, evaluation viewer
- **AI Training Interface**: Dataset upload, model training, performance monitoring
- **Real-time Feedback**: Live processing status and results

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
│   └── requirements.txt          # Python dependencies
├── webapp/                       # Frontend (HTML/JS)
│   ├── index.html                # Landing page
│   ├── upload.html               # Upload & align interface
│   ├── reference.html            # Reference line editor
│   ├── ai_training.html          # AI training overview
│   ├── ai_training_train.html    # Model training interface
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
- **http://localhost/upload.html** - Upload & analyze drawings
- **http://localhost/reference.html** - Define reference lines
- **http://localhost/ai_training.html** - AI training overview
- **http://localhost/ai_training_train.html** - Train new models
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

**STEP 3: Auto Match (Optional)**
- Image registration: Automatic alignment
- ECC or feature-based (ORB, RANSAC)
- Limited to ±30° rotation
- Line thinning: 1px skeleton for consistency

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
- **Output**: Single value (regression)

**Training Details:**
- **Optimizer**: Adam
- **Loss Function**: MSE (Mean Squared Error)
- **Regularization**: Dropout (0.5)
- **Input Preprocessing**: Normalization to [0, 1]

### Performance Metrics

**Regression Metrics:**
- **R² Score**: Coefficient of determination (0-1, higher better)
- **RMSE**: Root Mean Squared Error (lower better)
- **MAE**: Mean Absolute Error (lower better)
- **MAPE**: Mean Absolute Percentage Error (lower better)

**Example Output:**

```json
{
  "train_metrics": {
    "r2_score": 0.856,
    "rmse": 2.341,
    "mae": 1.823,
    "mape": 6.42
  },
  "val_metrics": {
    "r2_score": 0.792,
    "rmse": 2.876,
    "mae": 2.154,
    "mape": 7.89
  }
}
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

### Unified Output Format

Both extractors produce identical characteristics:
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
| `/api/upload` | POST | Upload and evaluate image |
| `/api/normalize-image` | POST | Auto-crop + scale + center |
| `/api/register-image` | POST | Image registration + thinning |
| `/api/check-duplicate` | POST | Check for duplicate images |

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
| `/api/training-data-evaluations` | GET | List training images |
| `/api/training-data-image/{id}/evaluate` | POST | Run line detection |
| `/api/training-data-image/{id}/ground-truth` | POST | Save ground truth |

### AI Training

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ai-training/dataset-info` | GET | Get dataset statistics |
| `/api/ai-training/available-features` | GET | List available features |
| `/api/ai-training/start-training` | POST | Start model training |
| `/api/ai-training/training-status` | GET | Get training progress |
| `/api/ai-training/models` | GET | List trained models |
| `/api/ai-training/models/{filename}` | GET | Get model metadata |

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
- patient_id: str (e.g., "PC56", "Park_16")
- task_type: str (COPY, RECALL, REFERENCE)
- source_format: str (MAT, OCS, DRAWN)
- original_file_data: bytes (original)
- processed_image_data: bytes (568×274, 2px lines)
- image_hash: str (SHA256)
- ground_truth_correct, ground_truth_extra: int (optional)
- features_data: str (JSON: {"Total_Score": 28, "MMSE": 24})
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

- [x] Iterative line detection with pixel subtraction
- [x] Hungarian algorithm for optimal matching
- [x] SHA256 duplicate detection
- [x] Modular API architecture (focused routers)
- [x] MAT/OCS extraction tools with auto-cropping
- [x] Line thickness normalization (2.00px)
- [x] AI training pipeline (ResNet-18)
- [x] Data augmentation system
- [x] Stratified train/test splits
- [x] Training data management interface
- [x] Model performance tracking

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
