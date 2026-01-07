# NPSketch - Agent Quick Reference Guide

**Purpose:** This document provides AI agents with essential information to quickly understand the project structure, Docker setup, and file access patterns.

---

## 📋 Project Description

**NPSketch** is a computer vision and machine learning application for neuropsychological drawing analysis:

- **Core Functionality:** Automated line detection, comparison, and evaluation of hand-drawn images
- **AI Training:** CNN models (ResNet-18) to predict clinical features (TotalScore, MMSE) from drawings
- **Data Sources:** Three import methods:
  1. **MAT Files** - MATLAB `.mat` files (machine tablet recordings)
  2. **OCS Images** - PNG/JPG with red-pixel annotations (human ratings)
  3. **Oxford Dataset** - Pre-processed PNG images with CSV labels (command-line import)

**Tech Stack:**
- Backend: FastAPI (Python 3.10+)
- Frontend: HTML/JavaScript (served by nginx)
- Database: SQLite (`npsketch.db`)
- ML: PyTorch (ResNet-18 CNN)
- Image Processing: OpenCV, PIL

---

## 📁 Project Structure

```
NPSketch/
├── api/                          # FastAPI Backend (mounted to /app in container)
│   ├── main.py                   # Application entry point
│   ├── database.py               # SQLAlchemy models & DB setup
│   ├── routers/                  # API endpoint modules
│   │   ├── upload.py             # Image upload & processing
│   │   ├── training_data.py      # Training data management (MAT/OCS)
│   │   ├── ai_training.py        # AI model training endpoints
│   │   └── ...
│   ├── image_processing/         # Computer vision library
│   │   ├── line_detector.py      # Hough Transform line detection
│   │   ├── comparator.py         # Hungarian algorithm matching
│   │   └── ...
│   ├── ai_training/              # ML pipeline
│   │   ├── model.py              # ResNet-18 CNN
│   │   ├── trainer.py            # Training orchestration
│   │   ├── data_loader.py        # Database → PyTorch dataset
│   │   └── ...
│   ├── mat_extraction/           # MAT file extractor
│   │   └── mat_extractor.py
│   ├── ocs_extraction/           # OCS image extractor
│   │   └── ocs_extractor.py
│   ├── oxford_extraction/        # Oxford dataset extractor
│   │   ├── oxford_normalizer.py  # Image normalization
│   │   ├── oxford_db_populator.py # Database import
│   │   └── validate_oxford_*.py  # Validation scripts
│   ├── line_normalizer.py        # Shared line thickness normalization
│   └── requirements.txt          # Python dependencies
│
├── webapp/                       # Frontend (HTML/JS, served by nginx)
│   ├── index.html                # Landing page
│   ├── upload.html               # Upload & analyze interface
│   ├── ai_training_data_upload.html  # MAT/OCS upload
│   └── ...
│
├── data/                         # Persistent data (Docker volume)
│   ├── npsketch.db               # SQLite database
│   ├── models/                   # Trained CNN models (.pth files)
│   ├── visualizations/           # Generated comparison images
│   └── tmp/                      # Temporary uploads/extractions
│
├── templates/                    # Input data (read-only volume)
│   ├── bsp_ocsplus_202511/       # MAT/OCS source data
│   └── training_data_oxford_manual_rater_202512/  # Oxford dataset
│       ├── imgs/                 # Original PNG images
│       ├── imgs_normalized_568x274/  # Normalized images
│       └── Rater1_simple.csv     # Labels (ID, Cond, TotalScore)
│
├── nginx/                        # Reverse proxy config
├── docker-compose.yml            # Container orchestration
├── README.md                     # Full documentation
├── DATA_IMPORT_FLOW.md           # Data import workflows
└── AGENTS.md                     # This file
```

---

## 🐳 Docker Container Setup

### Container Information

- **Container Name:** `npsketch-api`
- **Service Name:** `api` (in docker-compose.yml)
- **Base Image:** Python 3.10+ with OpenCV, PyTorch, FastAPI
- **Working Directory:** `/app`
- **Port:** 8000 (mapped to host 8000)

### Volume Mounts

| Host Path | Container Path | Access | Purpose |
|-----------|----------------|--------|---------|
| `./api` | `/app` | **RW** | API code (hot-reload enabled) |
| `./data` | `/app/data` | **RW** | Database, models, visualizations |
| `./templates` | `/app/templates` | **RO** | Input data (read-only) |
| `./webapp` | `/usr/share/nginx/html` | **RO** | Frontend (nginx) |

**Key Points:**
- ✅ **Writable:** `/app` (code) and `/app/data` (persistent data)
- ❌ **Read-only:** `/app/templates` (input data)
- ⚠️ **Important:** When writing to templates, use `/app/data/tmp/` then copy to host

---

## 💻 Running Python Code in Docker

### Basic Command Pattern

```bash
docker exec npsketch-api python3 /app/path/to/script.py [args]
```

### Common Examples

```bash
# Run a script in the api directory
docker exec npsketch-api python3 /app/oxford_extraction/oxford_db_populator.py --test

# Run with arguments
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \
  --input /app/templates/bsp_ocsplus_202511/Machine_rater/matfiles \
  --output /app/data/output

# Interactive Python shell
docker exec -it npsketch-api python3

# Run with environment variables
docker exec -e VAR=value npsketch-api python3 /app/script.py
```

### Accessing Database

```bash
# Python script accessing database
docker exec npsketch-api python3 -c "
from database import get_db, TrainingDataImage
db = next(get_db())
entries = db.query(TrainingDataImage).all()
print(f'Total entries: {len(entries)}')
db.close()
"
```

### Multi-line Python Commands

```bash
# Use heredoc for complex scripts
docker exec npsketch-api python3 << 'EOF'
from database import get_db, TrainingDataImage
import json

db = next(get_db())
entries = db.query(TrainingDataImage).filter(
    TrainingDataImage.source_format == 'OXFORD'
).all()

for entry in entries:
    print(f"ID {entry.id}: {entry.patient_id}")
db.close()
EOF
```

---

## 📂 File Access Patterns

### Path Mapping Reference

| What You Need | Host Path | Container Path | Notes |
|---------------|-----------|----------------|-------|
| **API Code** | `./api/main.py` | `/app/main.py` | Direct edit works (hot-reload) |
| **Database** | `./data/npsketch.db` | `/app/data/npsketch.db` | Use `/app/data/` in code |
| **Templates (read)** | `./templates/...` | `/app/templates/...` | Read-only, use for input |
| **Templates (write)** | `./templates/...` | ❌ **Cannot write** | Use `/app/data/tmp/` then `docker cp` |
| **Models** | `./data/models/` | `/app/data/models/` | Trained CNN models |
| **Visualizations** | `./data/visualizations/` | `/app/data/visualizations/` | Generated images |

### Writing Files from Container

**Problem:** `/app/templates` is read-only. You need to write files there.

**Solution:** Write to writable location, then copy to host:

```bash
# Step 1: Write to writable location in container
docker exec npsketch-api python3 -c "
with open('/app/data/tmp/output.csv', 'w') as f:
    f.write('data')
"

# Step 2: Copy from container to host
docker cp npsketch-api:/app/data/tmp/output.csv ./templates/training_data_oxford_manual_rater_202512/
```

### Reading Files in Python Code

```python
# ✅ CORRECT: Use container paths
import os
from pathlib import Path

# Database (writable)
db_path = "/app/data/npsketch.db"

# Templates (read-only)
csv_path = "/app/templates/training_data_oxford_manual_rater_202512/Rater1_simple.csv"
img_dir = "/app/templates/training_data_oxford_manual_rater_202512/imgs"

# Temporary files (writable)
tmp_dir = "/app/data/tmp"
```

### Common File Operations

```python
# Read from templates (read-only)
with open("/app/templates/path/to/file.csv", "r") as f:
    data = f.read()

# Write to data directory (writable)
with open("/app/data/tmp/output.png", "wb") as f:
    f.write(image_data)

# Database access
from database import get_db
db = next(get_db())
# ... use db ...
db.close()
```

---

## 🔑 Key Database Tables

### `training_data_images` (Main Training Data Table)

```python
- id: int (PK)
- patient_id: str (e.g., "PC56", "C0078")
- task_type: str ("COPY", "RECALL", "REFERENCE")
- source_format: str ("MAT", "OCS", "OXFORD", "DRAWN")
- original_filename: str
- original_file_data: bytes (BLOB)
- processed_image_data: bytes (BLOB, 568×274 PNG)
- image_hash: str (SHA256 for duplicate detection)
- extraction_metadata: str (JSON)
- features_data: str (JSON, e.g., {"Total_Score": 60})
- session_id: str
- uploaded_at: datetime
```

**Access Pattern:**
```python
from database import get_db, TrainingDataImage
db = next(get_db())
entries = db.query(TrainingDataImage).filter(
    TrainingDataImage.source_format == 'OXFORD'
).all()
```

---

## 🚀 Common Tasks Quick Reference

### 1. Import Oxford Dataset

```bash
# Normalize images
docker exec npsketch-api python3 /app/oxford_extraction/oxford_normalizer.py \
  /app/templates/training_data_oxford_manual_rater_202512/imgs \
  /app/templates/training_data_oxford_manual_rater_202512/imgs_normalized_568x274

# Import to database
docker exec npsketch-api python3 /app/oxford_extraction/oxford_db_populator.py
```

### 2. Validate Database Entries

```bash
docker exec npsketch-api python3 /app/oxford_extraction/validate_oxford_import.py
```

### 3. Query Database

```bash
docker exec npsketch-api python3 -c "
from database import get_db, TrainingDataImage
db = next(get_db())
count = db.query(TrainingDataImage).count()
print(f'Total entries: {count}')
db.close()
"
```

### 4. Extract MAT Files

```bash
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \
  --input /app/templates/bsp_ocsplus_202511/Machine_rater/matfiles \
  --output /app/data/tmp/extracted
```

### 5. Check Container Status

```bash
docker ps | grep npsketch
docker logs npsketch-api
```

---

## 📝 Important Notes for Agents

### File Paths
- ✅ **Always use container paths** (`/app/...`) in Python code
- ✅ **Database is at** `/app/data/npsketch.db`
- ✅ **Templates are read-only** - use `/app/data/tmp/` for writes, then `docker cp`

### Database Access
- Always use `get_db()` dependency or `SessionLocal()` from `database.py`
- Close database sessions: `db.close()`
- Use transactions: `db.commit()` after changes

### Image Processing
- All normalized images: **568×274 pixels**
- Line thickness: **2.00px** (Zhang-Suen + dilation)
- Format: **RGB PNG** (black lines on white background)
- Augmentation: **3× global + 2× local warping** (per image)
  - Global: Rotation ±3°, Translation ±10px, Scaling 0.95-1.05×
  - Warping: TPS with 9 control points, ±15px displacement
  - Pipeline: Transform → Binarize (threshold 175) → Line normalize (2px)

### Classification Training
- **Class Weights**: Automatically calculated for imbalanced datasets
  - Formula: `weight = total_samples / (num_classes * class_count)`
  - Applied to CrossEntropyLoss automatically
  - Stored in model metadata (`class_weights.weights`)
  - Logged during training with distribution analysis
- **Split Strategy**: Stratified by class labels (ensures balanced train/val split)
- **Imbalance Detection**: Warns if class ratio > 3:1

### Learning Rate Optimization
- **LR Scheduling**: ReduceLROnPlateau automatically reduces LR when val_loss plateaus
  - Enabled by default (`use_lr_scheduling: true`)
  - Factor: 0.5 (halve LR), Patience: 5 epochs, Min LR: 1e-6
  - Logs LR changes: `Learning Rate reduced: 0.001000 → 0.000500`
  - Stored in metadata (`lr_scheduling.final_lr`)
- **Differential LR**: Different learning rates for backbone vs head
  - Enabled by default (`use_differential_lr: true`)
  - Backbone: `LR × 0.1` (slow adaptation of pre-trained weights)
  - Head: `LR × 1.0` (fast learning of new task-specific layers)
  - Stored in metadata (`differential_lr.backbone_lr`, `differential_lr.head_lr`)
- **Configuration**: Set in `training_config.yaml` or via TrainingConfig Pydantic model

### Import Methods
- **MAT/OCS:** Web interface (`/api/extract-training-data`)
- **Oxford:** Command-line scripts (`oxford_extraction/oxford_normalizer.py` + `oxford_extraction/oxford_db_populator.py`)

### Container Commands
- Use `docker exec npsketch-api` for running Python code
- Use `docker cp` to transfer files between container and host
- Container name: `npsketch-api` (not `api`)

---

## 🔍 Quick Debugging

### Check if container is running
```bash
docker ps | grep npsketch-api
```

### View logs
```bash
docker logs npsketch-api
docker logs -f npsketch-api  # Follow logs
```

### Access container shell
```bash
docker exec -it npsketch-api /bin/bash
```

### Check file permissions
```bash
docker exec npsketch-api ls -la /app/templates
docker exec npsketch-api ls -la /app/data
```

### Test database connection
```bash
docker exec npsketch-api python3 -c "
from database import get_db
db = next(get_db())
print('Database connected!')
db.close()
"
```

---

## 📚 Additional Documentation

- **README.md** - Full project documentation
- **DATA_IMPORT_FLOW.md** - Detailed data import workflows
- **templates/training_data_oxford_manual_rater_202512/OXFORD_IMPORT_SUMMARY.md** - Oxford import details
- **api/ai_training/LOCAL_WARPING.md** - Local warping augmentation documentation

---

---

## 🆕 New Infrastructure (2025-12-29)

### Configuration System

**Centralized Config:** `api/config/training_config.yaml`
```python
from config import get_config

config = get_config()
batch_size = config.get("training.defaults.batch_size")  # 8
rotation_range = config.get("augmentation.rotation_range")  # [-3, 3]
synthetic_n_samples = config.get("synthetic.defaults.n_samples")  # 50
```

**Environment Overrides:**
```bash
export NPSKETCH_TRAINING_DEFAULTS_BATCH_SIZE=16
export NPSKETCH_LOGGING_LEVEL=DEBUG
```

### Type Safety (Pydantic Models)

**Models:** `api/config/models.py`
```python
from config.models import TrainingConfig
from pydantic import ValidationError

# Automatic validation
try:
    config = TrainingConfig(
        target_feature="Total_Score",
        train_split=0.8,
        num_epochs=50,
        use_lr_scheduling=True,      # Enable LR scheduling
        use_differential_lr=True,    # Enable differential LR
        backbone_lr_multiplier=0.1  # Backbone LR = LR × 0.1
    )
    # Type-safe access: config.batch_size → guaranteed int
except ValidationError as e:
    logger.error(f"Invalid config: {e}")
```

**Available Models:**
- `TrainingConfig` - Training parameters (includes LR scheduling & differential LR)
- `AugmentationConfig` - Augmentation settings
- `SyntheticImageConfig` - Synthetic image generation
- `ModelMetadata` - Model metadata structure

### Structured Logging

**Logger Setup:** `api/utils/logger.py`
```python
from utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Operation started")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)
logger.debug("Debugging only")
```

**Log Configuration:**
- Level: INFO (configurable in YAML)
- File: `/app/data/logs/training.log` (10 MB rotation, 5 backups)
- Format: `2025-12-29 14:30:15 - module - LEVEL - message`
- Module-specific levels in config.yaml

**Log Files:**
- Container: `/app/data/logs/training.log`
- Host: `./data/logs/training.log`
- Rotation: Automatic at 10 MB

---

**Last Updated:** 2026-01-07 
**Container Name:** `npsketch-api` 
**Working Directory:** `/app` 
**Version:** 1.1.4

