# NPSketch v1.0

**Automated Line Detection for Hand-Drawn Images**

NPSketch is a computer vision application that automatically compares hand-drawn images to reference templates, providing detailed feedback on drawing accuracy.

---

## 🎯 Motivation

The goal of this project is to create an automated solution for comparing hand-drawn images --- for example, the "House of Nikolaus" --- to a reference drawing. The system automatically identifies how many lines in the drawing are correct, missing, or misplaced, and provides both numerical and visual feedback.

---

## ✨ Features

- **Automated Line Detection**: Uses OpenCV Hough Transform to detect and extract line features from drawings
- **Smart Comparison**: Compares uploaded drawings to reference templates with configurable tolerance
- **Visual Feedback**: Generates side-by-side visualizations showing matches and differences
- **REST API**: Full-featured FastAPI backend with automatic documentation
- **Modern Web Interface**: Beautiful, responsive web UI with drag & drop upload
- **Database Storage**: SQLite database for storing images, features, and evaluation results
- **Persistent Data**: Volume-mounted data directory for database and visualizations
- **Hot Reload**: Development mode with automatic code reloading

---

## 🏗️ Architecture

```
npsketch/
├── data/                         # Persistent data (volume mounted)
│   ├── npsketch.db              # SQLite database
│   ├── visualizations/          # Generated comparison images
│   └── .gitignore               # Ignore DB & visualizations
├── api/                          # FastAPI Backend
│   ├── main.py                  # Main application (startup, health, router includes)
│   ├── routers/                 # Modular API endpoints
│   │   ├── __init__.py          # Router exports
│   │   ├── admin.py             # Administrative endpoints (migrations)
│   │   ├── upload.py            # Image upload and processing endpoints
│   │   ├── evaluations.py       # Evaluation management endpoints
│   │   ├── references.py        # Reference image management endpoints
│   │   └── test_images.py       # Test image management and testing endpoints
│   ├── database.py              # SQLAlchemy models and setup
│   ├── models.py                # Pydantic request/response models
│   ├── image_processing/        # Image processing library
│   │   ├── line_detector.py     # Line detection using Hough Transform
│   │   ├── comparator.py        # Line comparison and similarity metrics
│   │   └── utils.py             # Image preprocessing utilities
│   ├── services/                # Business logic layer
│   │   ├── reference_service.py # Reference image management
│   │   └── evaluation_service.py# Evaluation workflow
│   ├── mat_extraction/          # MATLAB .mat file extraction tools
│   │   ├── mat_extractor.py     # Extract reference and drawn images from .mat files
│   │   ├── mat_extractor.conf   # Resolution configuration (fixed, manual)
│   │   └── README.md            # MAT extraction documentation
│   ├── ocs_extraction/          # OCS image extraction tools
│   │   ├── ocs_extractor.py     # Extract red-pixel drawings from OCS rating images
│   │   ├── ocs_extractor.conf   # Resolution and threshold configuration
│   │   └── README.md            # OCS extraction documentation
│   ├── line_normalizer.py       # Line thickness normalization utility (Zhang-Suen + dilation)
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile               # API container definition
├── webapp/                      # Frontend (served by nginx)
│   ├── index.html               # Landing page with stats
│   ├── upload.html              # Upload & align interface (3-step workflow)
│   ├── reference.html           # Manual reference line editor
│   ├── draw_testimage.html      # Test image creation tool
│   ├── run_test.html            # Automated test runner
│   └── evaluations.html         # Evaluation viewer & rating
├── nginx/                       # Nginx configuration
│   └── nginx.conf               # Reverse proxy config
├── docker-compose.yml           # Container orchestration
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

### Communication Flow

```
Browser → nginx (Port 80) → Static HTML files (webapp/)
        ↓
        JavaScript → /api/* → nginx → FastAPI (Port 8000)
        ↓
        Response with JSON data
```

---

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- (Optional) Python 3.10+ for local development

### Installation & Running

1. **Clone or navigate to the project directory:**
   ```bash
   cd npsketch
   ```

2. **Start the application:**
   ```bash
   docker compose up --build -d
   ```

3. **Access the application:**
   - **Landing Page**: http://localhost (stats and overview)
   - **Upload & Align**: http://localhost/upload.html (upload and process drawings)
   - **Reference Editor**: http://localhost/reference.html (define reference lines manually)
   - **Draw Test Images**: http://localhost/draw_testimage.html (create test dataset)
   - **Run Tests**: http://localhost/run_test.html (automated testing with metrics)
   - **Evaluations**: http://localhost/evaluations.html (view and rate evaluations)
   - **API Documentation**: http://localhost/api/docs (interactive Swagger UI)

4. **Stop the application:**
   ```bash
   docker compose down
   ```

---

## 📖 How It Works

### 1. Environment and Setup

The system uses:
- **FastAPI** for creating the REST API
- **OpenCV** for image processing and line detection
- **scikit-image** for advanced image registration and morphological operations
- **SQLite** for database storage and metadata management
- **Nginx** as reverse proxy and static file server

### 2. Reference Image Definition (Manual)

**NPSketch uses a manual, interactive approach for defining reference features:**

On first startup:
1. The system detects that no reference image is initialized
2. The webapp displays only the "Reference Image" option
3. The user opens the **Reference Editor** (`reference.html`)

**Manual Line Definition:**
- The reference image is displayed at 256×256 pixels
- User clicks two points on the image to define a line
- Each line is automatically categorized as:
  - **Horizontal** (angle 0° ± 10°)
  - **Vertical** (angle 90° ± 10°)
  - **Diagonal** (45°, 135°, 60°, 120°, etc.)
- Lines are color-coded and displayed as an overlay
- Lines can be deleted individually from the feature list
- Once all lines are defined, they are stored in the database

**Why Manual Definition?**
- 100% accuracy for ground truth (no detection errors)
- Full control over what constitutes a "correct" line
- Works for any drawing, regardless of complexity
- Simple and intuitive for non-technical users

### 3. Test Image Creation

**NPSketch includes a powerful test image creator** (`draw_testimage.html`):

**Drawing Tools:**
- 256×256 pixel canvas (matches reference resolution)
- Adjustable brush size (1-10px)
- Draw and erase modes
- Clear all functionality
- **Rotation buttons** (±10°) to create variations

**Manual Scoring:**
- User manually scores the drawing:
  - **Correct Lines**: How many reference lines are present
  - **Missing Lines**: Auto-calculated (reference_total - correct)
  - **Extra Lines**: Additional lines not in reference
- This creates ground truth data for testing the detection algorithm

**Test Image Management:**
- All test images are listed with their scores
- Click to load and edit
- Update existing or save as new
- Delete unwanted test images
- Automatic timestamp-based naming

**Why Manual Test Images?**
- Creates a labeled dataset for algorithm validation
- Tests edge cases (rotations, missing lines, extra strokes)
- Measures "Prediction Accuracy" (expected vs actual detection)

### 4. Image Upload & Processing (3-Step Workflow)

**NPSketch provides an intuitive 3-step workflow for image upload and processing:**

**STEP 1: Upload & Auto-Normalization (Backend)**
When you upload an image via `upload.html`:
1. **Auto-Crop**: Removes white space around the drawing
2. **Scale to Fit**: Scales object to fill 256×256 canvas (maintains aspect ratio using `min(height_ratio, width_ratio)`)
3. **Center**: Places scaled drawing centered on white 256×256 canvas
4. **Display**: Shows normalized image immediately in the workspace
5. **Duplicate Detection**: Checks if image already exists using SHA256 hash
   - Compares against original uploaded files in database
   - Shows warning banner if duplicate detected
   - Analysis continues but duplicate evaluations are marked

**STEP 2: Manual Adjustments (Frontend)**
After upload, you can fine-tune the image interactively:
- **Scale Control** (50-300%): Adjust image size with slider
- **Rotation Control** (-180° to +180°): Rotate with slider or ±10° buttons
- **Translation Controls**: Move image with arrow buttons (X/Y in pixels)
- **Overlay Mode**: Toggle to see your drawing overlaid on reference
- All adjustments are applied in real-time on the canvas

**STEP 3: Auto Match (Optional, Backend)**
If "Apply Auto Match after upload" is checked (default):
- **Image Registration**: Automatically aligns your drawing to the reference
  - Uses ECC or feature-based registration (ORB, RANSAC)
  - Applies optimal translation, rotation, and scale
  - Configurable motion type (Translation/Euclidean/Similarity/Affine)
  - Limited to ±30° rotation for natural drawings
- **Line Thinning**: Skeletonizes lines to 1px thickness for consistent detection
- Manual transformations are reset after auto match completes

**STEP 4: Analysis & Evaluation**
Click "Analyze Drawing" to:
- Extract line features from your processed drawing
- Compare detected lines to reference features
- Calculate accuracy metrics (correct, missing, extra lines)
- Generate side-by-side visualization
- Store results in database
- Show **"View in Evaluations"** button (green) to navigate to evaluations page
- Button automatically filters to show "Unevaluated" entries

**UX Enhancements:**
- Spinner overlay on canvas during processing (upload and analysis)
- "Analyze Drawing" button disabled while processing
- Real-time feedback with status messages
- Duplicate warning banner with detailed information

**Stage 2: Line Detection (Iterative with Pixel Subtraction)**
1. **Binary Threshold**: Strong black/white separation (threshold=127)
2. **Iterative Detection** (up to 20 iterations):
   - **Multi-Pass Strategy**:
     - Pass 1 (Iter 1-10): Strict threshold (18), longer lines (35px)
     - Pass 2 (Iter 11-20): Relaxed threshold (10), shorter lines (25px)
   - **Longest-First**: Sorts detected lines by length, picks longest
   - **Overlap Check**: Rejects duplicates using angle (±8°) and position (±25px)
   - **Special Handling**: Crossing lines (X pattern) are NOT considered duplicates if angle difference is 80-100°
   - **Pixel Subtraction**:
     - Draws detected line on mask with 8px buffer
     - **Dilates** mask with 5×5 ellipse kernel (expands by 2-3px)
     - **Subtracts** dilated region from image
     - This removes the line completely, preventing re-detection
   - Stops when no more lines found or 12 lines detected
3. **Final Filter**: Removes lines shorter than 30px (noise reduction)

**Why This Approach?**
- Iterative detection finds all lines systematically
- Pixel subtraction eliminates duplicates naturally
- Dilate ensures complete removal (no artifacts)
- Multi-pass catches both strong and weak lines
- Crossing lines are properly handled

### 5. Comparison & Evaluation (Hungarian Algorithm)

**NPSketch uses the Hungarian Algorithm for optimal line matching:**

**Line Similarity Calculation:**
For each pair of detected and reference lines, calculate:
1. **Position Distance**: Euclidean distance between line midpoints
   - Normalized by tolerance (default: 120px)
2. **Angle Difference**: Angular difference in degrees
   - Normalized to 0-90° range (lines have no direction)
   - Normalized by tolerance (default: 50°)
3. **Length Ratio**: Relative length difference
   - Calculated as: `|len1 - len2| / max(len1, len2)`
   - Normalized by tolerance (default: 0.8 or 80%)
4. **Combined Similarity**: Weighted average
   - Position: 40%, Angle: 30%, Length: 30%

**Optimal Matching (Hungarian Algorithm):**
- Creates a cost matrix (1 - similarity for all pairs)
- Uses `scipy.optimize.linear_sum_assignment` for optimal bipartite matching
- Ensures best global assignment (not greedy)
- Filters matches by similarity threshold (default: 0.5)

**Metrics Calculated:**
- **Correct Lines**: Number of matched pairs (similarity ≥ threshold)
- **Missing Lines**: Reference lines with no match
- **Extra Lines**: Detected lines >30px with no match (filters noise)
- **Reference Match Score**: `correct_lines / total_reference_lines`

**Accuracy Calculation:**
```python
if missing == 0:
    accuracy = 100%
else:
    # Penalize: each extra line cancels one correct line
    adjusted_correct = max(0, correct - extra)
    accuracy = adjusted_correct / total_reference_lines
```

**Why Hungarian Algorithm?**
- Guarantees optimal matching (no local optima)
- Fairer than greedy algorithms
- Standard approach in computer vision
- Better handles ambiguous cases

### 6. Automated Testing & Validation

**Run Tests Page** (`run_test.html`):
- Displays count of available test images
- Configurable settings:
  - **Image Registration**: Enable/disable, motion type (Translation/Euclidean/Similarity/Affine)
  - **Line Matching Tolerances**: Position, angle, length
  - **Max Rotation**: Limit for registration search
- **Runs all test images** through the pipeline
- Compares **Expected vs Actual** detection results

**Two Key Metrics:**
1. **Reference Match**: How well detected lines match the reference (like normal evaluation)
2. **Test Rating** (Prediction Accuracy): How well actual detection matches expected scores
   - Perfect when detected lines match user's manual scoring
   - Measures algorithm reliability

**Test Results Display:**
- Overall statistics (average Test Rating, perfect tests, etc.)
- Individual results (collapsed/expandable)
- Side-by-side visualization: Original → Registered → Reference
- Shows: Expected (user), Actual (detected), Difference

### 7. Visualization and Debugging

Each processed image generates a 3-panel visualization:

**Panel 1: Original**
- Shows the uploaded/test image as-is

**Panel 2: Registered**
- Shows the image after registration (if enabled)
- Displays transformation info (rotation, scale, translation)
- Detected lines color-coded:
  - 🟢 **Green**: Matched lines (in both detected and reference)
  - 🔵 **Blue**: Extra lines (detected but not in reference)

**Panel 3: Reference**
- Shows the reference image with manually defined lines
- Reference lines color-coded:
  - 🟢 **Green**: Matched lines (found in detection)
  - 🔴 **Red**: Missing lines (not detected)

**Registration Info:**
- Translation (X/Y in pixels)
- Rotation (degrees)
- Scale factor (e.g., 1.77x)
- Overlap score (quality metric)

Visualizations are stored in `data/visualizations/` and accessible through the API.

### 8. Database Schema

**Tables:**

1. **reference_images**
   - Stores reference templates with manually defined features
   - Fields: id, name (unique), image_data (BLOB), processed_image_data (BLOB), lines_data (JSON), num_lines, width, height, created_at
   - `lines_data`: JSON array of manually defined lines: `[{"x1": ..., "y1": ..., "x2": ..., "y2": ...}, ...]`

2. **test_images**
   - Stores test images with expected ground truth scores
   - Fields: id, test_name (unique), image_data (BLOB), expected_correct, expected_missing, expected_extra, created_at
   - Used for algorithm validation and testing

3. **uploaded_images**
   - Stores uploaded drawings for live evaluation
   - Fields: id, filename, image_data (BLOB), processed_image_data (BLOB), image_hash (SHA256), uploader, uploaded_at
   - `image_data`: Original uploaded file (before normalization)
   - `processed_image_data`: Normalized 256×256 image
   - `image_hash`: SHA256 hash of original file for duplicate detection

4. **extracted_features**
   - Stores automatically detected line features
   - Fields: id, image_id (FK), feature_data (JSON), num_lines, extracted_at
   - `feature_data`: JSON with detected lines and metadata

5. **evaluation_results**
   - Stores comparison results (for uploads and tests)
   - Fields: id, image_id (FK), reference_id (FK), test_image_id (FK), correct_lines, missing_lines, extra_lines, similarity_score, detection_score, visualization_path, evaluated_at, registration_info (JSON)
   - `registration_info`: Contains transformation details (rotation, scale, translation)
   - Optional: `user_evaluated`, `evaluated_correct`, `evaluated_missing`, `evaluated_extra` for manual validation

---

## 🔧 Configuration

### Line Detection Parameters (Iterative Method)

Edit `api/image_processing/line_detector.py`:

```python
LineDetector(
    rho=1.0,                # Distance resolution (pixels)
    theta=np.pi/180,        # Angle resolution (radians)
    threshold=18,           # Base threshold (Pass 1: strict)
    min_line_length=35,     # Initial min length (Pass 1)
    max_line_gap=35,        # Max gap between segments
    final_min_length=30     # Final filter (removes short artifacts)
)
```

**Multi-Pass Strategy:**
- **Pass 1** (Iterations 1-10): `threshold=15`, `min_line_length=35px` (strong lines)
- **Pass 2** (Iterations 11-20): `threshold=10`, `min_line_length=25px` (weak lines)

**Overlap Detection:**
- `angle_threshold=8.0°` (lines with similar angles)
- `position_threshold=25.0px` (lines at similar positions)
- Special case: 80-100° difference = crossing lines (both kept)

### Image Registration Parameters

Edit `api/image_processing/image_registration.py` or configure via UI:

```python
ImageRegistration(
    max_rotation_degrees=30,    # Search range for rotation
    rotation_step=3,            # Angular resolution
    scale_range=(0.75, 1.30),   # Scale search range
    scale_step=0.05,            # Scale resolution
    translation_range=(-10, 10), # X/Y search range
    translation_step=5          # Translation resolution
)
```

**Pre-Scaling:**
- Calculates `min(height_ratio, width_ratio)` to prevent clipping
- Clips to `0.5-2.5x` range
- Centers result on white canvas

**Thinning (Skeletonization):**
- Applied if `total_scale > 1.1x`
- Uses `skimage.morphology.skeletonize` for 1-pixel lines
- Includes Gaussian smoothing for anti-aliasing

### Comparison Tolerance (Hungarian Algorithm)

Edit `api/image_processing/comparator.py` or configure via UI:

```python
LineComparator(
    position_tolerance=120.0,   # Max position difference (pixels)
    angle_tolerance=50.0,       # Max angle difference (degrees)
    length_tolerance=0.8,       # Max length difference (ratio)
    similarity_threshold=0.5    # Min similarity for match
)
```

**Optimized Values:**
- These values were determined through automated grid search testing
- Balance between precision and recall
- Configurable in real-time via Run Tests page

---

## 📡 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check and stats |
| `/api/upload` | POST | Upload and evaluate image (with duplicate detection) |
| `/api/check-duplicate` | POST | Check if image already exists (SHA256 hash) |
| `/api/normalize-image` | POST | STEP 1: Auto-crop + scale + center to 256×256 |
| `/api/register-image` | POST | STEP 3: Registration + line thinning to 1px |
| `/api/evaluations/recent` | GET | List recent evaluations |
| `/api/evaluations/{id}` | GET | Get specific evaluation |
| `/api/evaluations/{id}` | DELETE | Delete evaluation (cascades to uploaded_images) |
| `/api/evaluations/{id}/evaluate` | PUT | Add manual evaluation scores |
| `/api/references` | GET | List reference images |
| `/api/references/{id}/image` | GET | Get reference image data |
| `/api/visualizations/{file}` | GET | Get visualization image |
| `/api/admin/migrate-add-image-hash` | POST | Admin: Add image_hash column and populate |
| `/api/docs` | GET | Interactive API documentation |

### Example: Evaluate Test Image

```bash
curl -X POST "http://localhost/api/test-images/1/evaluate?use_registration=true" \
  -H "Content-Type: application/json"
```

Response:
```json
{
  "test_id": 1,
  "test_name": "test_house_rotated",
  "correct_lines": 7,
  "missing_lines": 1,
  "extra_lines": 0,
  "similarity_score": 0.875,
  "detection_score": 0.875,
  "visualization_path": "/api/visualizations/test_1.png",
  "registration_info": {
    "used": true,
    "method": "brute_force_with_scale",
    "translation_x": 10.0,
    "translation_y": 5.0,
    "rotation_degrees": -6.0,
    "scale": 1.77,
    "overlap_score": 0.82
  }
}
```

---

## 🧪 Testing

### Manual Testing

1. Navigate to http://localhost/upload.html
2. Drag and drop or click to upload a hand-drawn image
3. View instant feedback with:
   - Numerical metrics (correct, missing, extra lines)
   - Similarity score percentage
   - Side-by-side visual comparison

### API Testing

Use the interactive API documentation at http://localhost/api/docs

### Creating Test Images

Draw the "House of Nikolaus" pattern:
- Start at bottom left corner
- Draw to bottom right (base)
- Continue up to form a square
- Add roof triangle
- Draw diagonals inside
- Complete as one continuous line if possible

---

## 🛠️ Development

### Local Development (without Docker)

1. **Install dependencies:**
   ```bash
   cd api
   pip install -r requirements.txt
   ```

2. **Create data directory:**
   ```bash
   mkdir -p ../data/visualizations
   ```

3. **Run the API:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access at:** http://localhost:8000/api/docs

### Modular API Architecture

The API is organized into focused router modules for better maintainability:

```python
# api/routers/
├── admin.py        # Database migrations, system admin
├── upload.py       # Image upload, duplicate check, normalization, registration  
├── evaluations.py  # Evaluation CRUD, user feedback
├── references.py   # Reference management, manual feature editing
└── test_images.py  # Test image management, automated testing
```

Each router handles a specific domain, making the codebase more:
- **Maintainable**: Easy to locate and modify functionality
- **Scalable**: New features can be added to appropriate modules
- **Testable**: Individual routers can be tested in isolation
- **Readable**: Clear separation of concerns

### Adding New Reference Images

```python
from services import ReferenceService
from database import get_db

db = next(get_db())
ref_service = ReferenceService(db)

# Load your image
import cv2
image = cv2.imread('your_reference.png')

# Store as reference
ref_service.store_reference('my_reference', image)
```

---

## 🔬 Data Extraction Tools

NPSketch includes two specialized extraction tools for processing neuropsychological assessment data:

1. **MAT Extractor**: Extracts from MATLAB `.mat` files (machine tablet recordings)
2. **OCS Extractor**: Extracts from PNG images (human expert ratings)

Both tools produce **identical output format** (568×274 PNG, 2.00px lines) for seamless CNN training integration.

### Quick Commands

```bash
# MAT Extractor - Process all .mat files in directory
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \
  --input /app/templates/source_folder \
  --output /app/data/target_folder

# OCS Extractor - Process all PNG images in directory  
docker exec npsketch-api python3 /app/ocs_extraction/ocs_extractor.py \
  --input /app/templates/source_folder \
  --output /app/data/target_folder
```

---

## 🔬 MATLAB .mat File Extraction

Specialized tool for extracting reference images and hand-drawn lines from MATLAB `.mat` files (e.g., from neuropsychological assessments like Figure Copy tasks).

### MAT Extractor Tool

**Location:** `api/mat_extraction/`

**Features:**
- **Auto-Cropping**: Automatically crops to actual drawing content with configurable padding (5px)
- **Line Normalization**: Zhang-Suen thinning + dilation for consistent 2.00px line thickness
- **Batch Processing**: Recursively processes all `.mat` files in directory tree
- Extracts reference images (templates) from `.mat` files
- Extracts hand-drawn lines from patient recordings (COPY and RECALL tasks)
- Renders drawing strokes as PNG images with uniform resolution
- Configuration-based resolution management (default: 568×274 landscape format)
- Optimized for CNN training with minimal white space and consistent line thickness

### Quick Start

**Process all .mat files in a directory:**

```bash
# Basic usage - process entire directory tree
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \
  --input /app/templates/bsp_ocsplus_202511 \
  --output /app/data/tmp

# Using mounted volumes from host machine
# Assumes: ./templates is mounted to /app/templates
#          ./data is mounted to /app/data
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \
  --input /app/templates/your_matfiles_folder \
  --output /app/data/output_folder
```

**With custom config:**

```bash
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \
  --input /path/to/matfiles \
  --output /path/to/output \
  --config /path/to/custom.conf
```

**Example with real paths:**

```bash
# Process MAT files from templates/bsp_ocsplus_202511/Machine_rater/matfiles
# Output to data/tmp
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \
  --input /app/templates/bsp_ocsplus_202511/Machine_rater/matfiles \
  --output /app/data/tmp
```

### Configuration File

The tool uses `mat_extractor.conf` (read-only, manually configured):

```json
{
  "canvas_width": 568,
  "canvas_height": 274,
  "auto_crop": true,
  "padding_px": 5,
  "created_at": "2025-11-11T08:00:00.000000",
  "source": "Optimized for CNN training - landscape format with auto-cropping"
}
```

**Configuration Options:**
- `canvas_width` / `canvas_height`: Target output resolution (default: 568×274)
- `auto_crop`: Enable automatic content cropping (default: true)
- `padding_px`: White border padding around content (default: 5px)

**Behavior:**
- On first run, tool creates default config and exits
- User reviews and adjusts config as needed
- Config is never modified by the tool (read-only)
- Ensures uniform canvas size for consistent analysis

### Output Structure

For each `.mat` file, the tool generates:

1. **Reference Image** (1 per patient)
   - `{PatientID}_REFERENCE_{date}.png` (e.g., `PC56_REFERENCE_20251111.png`)
   - The template/original figure to be drawn
   - Auto-cropped to non-white pixels + padding
   - Scaled to configured resolution (e.g., 568×274)

2. **COPY Image** (drawn while viewing reference)
   - `{PatientID}_COPY_drawn_{date}.png`
   - Contains all drawing strokes as black lines on white background
   - Auto-cropped to actual drawing bounds + padding
   - Scaled to configured resolution

3. **RECALL Image** (drawn from memory)
   - `{PatientID}_RECALL_drawn_{date}.png`
   - Contains all drawing strokes as black lines on white background
   - Auto-cropped to actual drawing bounds + padding
   - Scaled to configured resolution

**Result:** All outputs have consistent dimensions (e.g., 568×274) with ~5-6px white margins, optimized for CNN training.

### Technical Details

**Data Extraction:**
- Reference images from `figs` field (RGB, 568×568)
- Drawing data from `trails.cont_lines` (separate stroke arrays)
- Patient ID extraction from filename (prefers `PC` over `Pro` identifiers)

**Auto-Cropping Algorithm:**
- **Drawn Images**: Bounding box calculated from actual line coordinates
- **Reference Images**: Bounding box calculated from non-white pixels (threshold < 250)
- Configurable padding added around content (default: 5px)
- Preserves aspect ratio while scaling to target resolution
- Ensures optimal content density for machine learning

**Drawing Reconstruction:**
- Each stroke stored as (x, y, timestamp) coordinates
- Stroke separation detected via time gaps (mean + 2×std)
- Coordinates mapped from screen space to canvas space
- Lines rendered with 2px width for visibility

**Why Extract from .mat Files?**
- Preserves original data from neuropsychological assessments
- Enables automated analysis of clinical drawings
- Creates standardized dataset for algorithm development
- Supports batch processing of patient data

### Example Output

```
Processing: FIGURECOPY_data_OCSPlus_210426_Pro1003_PC56_German_20210426T125245.mat
  ✓ Saved reference: PC56_REFERENCE_20251111.png (568x274)
  ✓ Saved COPY  : PC56_COPY_drawn_20251111.png (15 lines, 1245 points)
  ✓ Saved RECALL: PC56_RECALL_drawn_20251111.png (10 lines, 1091 points)
```

---

## 🎨 OCS Image Extraction

NPSketch includes a specialized tool for extracting red-pixel drawings from OCS (Observer-rated Clinical Scale) rating images. These images typically contain multiple elements (reference figures, grids, annotations), and the tool isolates only the hand-drawn red lines.

### OCS Extractor Tool

**Location:** `api/ocs_extraction/`

**Features:**
- **Red Pixel Detection**: Isolates red pixels using configurable RGB thresholds (R≥200, G≤100, B≤100)
- **Auto-Cropping**: Automatically crops to actual drawing bounds with padding (5px)
- **Line Normalization**: Zhang-Suen thinning + dilation for consistent 2.00px line thickness
- **Noise Removal**: Removes all non-red elements (reference figures, grids, text)
- **Batch Processing**: Recursively processes all PNG files in directory tree
- **Fixed Resolution**: All outputs scaled to 568×274 (landscape format)
- **CNN-Optimized**: Minimal white space, consistent margins (3-7px), and uniform line thickness

### Quick Start

**Process all PNG images in a directory:**

```bash
# Basic usage - process entire directory tree
docker exec npsketch-api python3 /app/ocs_extraction/ocs_extractor.py \
  --input /app/templates/bsp_ocsplus_202511/Human_rater/imgs \
  --output /app/data/tmp

# Using mounted volumes from host machine
# Assumes: ./templates is mounted to /app/templates
#          ./data is mounted to /app/data
docker exec npsketch-api python3 /app/ocs_extraction/ocs_extractor.py \
  --input /app/templates/your_images_folder \
  --output /app/data/output_folder
```

**With custom config:**

```bash
docker exec npsketch-api python3 /app/ocs_extraction/ocs_extractor.py \
  --input /path/to/images \
  --output /path/to/output \
  --config /path/to/custom.conf
```

**Example with real paths:**

```bash
# Process OCS PNG files from templates/bsp_ocsplus_202511/Human_rater/imgs
# Output to data/tmp
docker exec npsketch-api python3 /app/ocs_extraction/ocs_extractor.py \
  --input /app/templates/bsp_ocsplus_202511/Human_rater/imgs \
  --output /app/data/tmp
```

### Configuration File

The tool uses `ocs_extractor.conf` (read-only, manually configured):

```json
{
  "canvas_width": 568,
  "canvas_height": 274,
  "auto_crop": true,
  "padding_px": 5,
  "red_threshold": {
    "r_min": 200,
    "g_max": 100,
    "b_max": 100
  }
}
```

**Configuration Options:**
- `canvas_width` / `canvas_height`: Target output resolution (default: 568×274)
- `auto_crop`: Enable automatic content cropping (default: true)
- `padding_px`: White border padding around content (default: 5px)
- `red_threshold`: RGB thresholds for red pixel detection

### Input Files

Expected naming convention:
- `{PatientID}_COPY.png` - Immediate copy task
- `{PatientID}_RECALL.png` - Memory recall task

Examples: `Park_16_COPY.png`, `TeamD178_RECALL.png`, `TEAMK299_COPY.png`

### Output Structure

For each input image, the tool generates:

**Output Filename:** `{PatientID}_{TaskType}_ocs_{date}.png`

Examples:
- `Park_16_COPY_ocs_20251111.png`
- `TeamD178_RECALL_ocs_20251111.png`
- `TEAMK299_COPY_ocs_20251111.png`

**Output Characteristics:**
- Resolution: 568×274 pixels (consistent)
- Format: Black lines on white background
- Margins: ~3-7px white border
- Content: Only red pixels from original, converted to black

### Processing Pipeline

1. **Load Image**: Read OCS rating image (typically 520×420 RGBA)
2. **Red Pixel Detection**: Apply RGB threshold (R≥200, G≤100, B≤100)
3. **Bounding Box**: Find minimal bounds around red pixels
4. **Add Padding**: Extend bounds by 5px
5. **Crop**: Extract only the bounded region
6. **Color Conversion**: Red pixels → Black, background → White
7. **Resize**: Scale to 568×274
8. **Line Normalization**: Skeletonize (Zhang-Suen) + dilate to 2.00px thickness
9. **Save**: Output as PNG

### Example Results

```
Processing: Park_16_COPY.png
  Found 6,251 red pixels in 520x420 image
  Bounding box: x=120..421, y=98..260 (302x163)
  ✓ Saved: Park_16_COPY_ocs_20251111.png (568x274)

Processing: TEAMK276_RECALL.png
  Found 9,021 red pixels in 520x420 image
  Bounding box: x=60..447, y=90..301 (388x212)
  ✓ Saved: TEAMK276_RECALL_ocs_20251111.png (568x274)
```

### Why Extract from OCS Images?

- **Human Ratings**: OCS images come from human raters who annotate on screen
- **Multi-Element Images**: Original images contain grids, reference figures, and annotations
- **Red Pixel Isolation**: Only the red hand-drawn lines are the actual rating
- **Standardization**: Converts diverse input formats to uniform output for CNN training
- **Quality Control**: Enables automated analysis of human rating quality

### Comparison: MAT vs OCS Extractors

| Feature | MAT Extractor | OCS Extractor |
|---------|---------------|---------------|
| **Input** | MATLAB .mat files | PNG images |
| **Source** | Machine recordings | Human ratings |
| **Data Type** | Coordinate arrays | Pixel data |
| **Extraction** | Line rendering from (x,y,t) | Color filtering (red pixels) |
| **Reference** | Included in .mat | Separate file |
| **Output** | 568×274 PNG | 568×274 PNG |
| **Line Thickness** | 2.00px (normalized) | 2.00px (normalized) |
| **Use Case** | Patient tablet recordings | Expert visual ratings |

Both tools produce identical output characteristics (568×274 resolution, 2.00px line thickness) for seamless CNN training pipeline integration.

### Line Thickness Normalization

Both extractors implement **automatic line thickness normalization** to ensure consistent 2.00px lines across all outputs:

**Algorithm:**
1. **Skeletonization**: Zhang-Suen thinning algorithm reduces all lines to 1-pixel thickness
2. **Dilation**: Morphological dilation with ellipse kernel expands to target thickness (2px)
3. **Result**: Perfectly consistent line thickness across all images

**Benefits for CNN Training:**
- ✓ Eliminates line thickness as a confounding variable
- ✓ Reduces input variance → faster convergence
- ✓ Better generalization across different data sources
- ✓ Consistent feature extraction

**Verification:**
```bash
# All outputs verified to have 2.00px ± 0.00px line thickness
# MAT Extractor: Reference, COPY, RECALL all @ 2.00px
# OCS Extractor: COPY, RECALL all @ 2.00px
```

### Complete Workflow Example

**Process both MAT files and OCS images:**

```bash
# Step 1: Process MAT files (machine recordings)
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \
  --input /app/templates/bsp_ocsplus_202511/Machine_rater/matfiles \
  --output /app/data/output

# Step 2: Process OCS images (human ratings)
docker exec npsketch-api python3 /app/ocs_extraction/ocs_extractor.py \
  --input /app/templates/bsp_ocsplus_202511/Human_rater/imgs \
  --output /app/data/output

# Result: All images in /app/data/output with:
#   - 568×274 resolution
#   - 2.00px line thickness
#   - 5px white padding
#   - Black lines on white background
```

**Output Structure:**

```
data/output/
├── PC56_REFERENCE_20251111.png      # MAT: Reference image
├── PC56_COPY_drawn_20251111.png     # MAT: COPY drawn
├── PC56_RECALL_drawn_20251111.png   # MAT: RECALL drawn
├── Park_16_COPY_ocs_20251111.png    # OCS: COPY rating
├── Park_16_RECALL_ocs_20251111.png  # OCS: RECALL rating
└── ... (more files)
```

### Important Notes

**Configuration Management:**
- Both tools use **read-only** configuration files
- On first run, a default config is created and the tool exits
- Review and adjust the config file as needed, then run again
- Config files are never modified by the tools (ensures reproducibility)

**Volume Mounting:**
- The Docker container has volumes mounted:
  - `./api` → `/app` (code)
  - `./data` → `/app/data` (output)
  - `./templates` → `/app/templates` (input, optional)
- All paths in commands use container paths (`/app/...`)
- Output files are accessible on host in `./data/` directory

**Batch Processing:**
- Both tools recursively search input directories
- MAT Extractor: Finds all `.mat` files
- OCS Extractor: Finds all PNG files with `COPY` or `RECALL` in filename
- Processing summary shown at end (successful/failed counts)

**Error Handling:**
- Invalid files are skipped with error messages
- Processing continues for remaining files
- Check console output for detailed error information

**Performance:**
- Line normalization adds ~0.5-1 second per image
- Typical processing: 1-2 seconds per MAT file (3 images each)
- OCS images: ~0.5 seconds per image

### Troubleshooting

**"Config file not found":**
```bash
# First run creates default config, then exits
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \
  --input /app/templates/test --output /app/data/tmp

# Review and edit config if needed:
# api/mat_extraction/mat_extractor.conf

# Run again after reviewing config
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \
  --input /app/templates/test --output /app/data/tmp
```

**"No .mat files found" or "No OCS images found":**
- Check that input path is correct (use container paths: `/app/...`)
- Verify files exist: `docker exec npsketch-api ls /app/templates/your_folder`
- MAT files must have `.mat` extension
- OCS images must have `COPY` or `RECALL` in filename and `.png` extension

**Output files not appearing on host:**
- Check volume mounting in `docker-compose.yml`
- Verify output directory is inside mounted volume (e.g., `/app/data/`)
- Check container permissions: `docker exec npsketch-api ls -la /app/data/`

**Line thickness not 2.00px:**
- Verify `normalize_lines: true` in config file
- Check that `cv2.ximgproc.thinning` is available (requires opencv-contrib-python)
- Look for normalization errors in console output

**Memory errors with large datasets:**
- Process in smaller batches
- Increase Docker memory limit if needed

---

## 🐳 Docker Details

### Services

- **nginx**: Serves frontend (webapp/) and proxies API requests (Port 80)
- **api**: FastAPI backend with OpenCV and image processing (Port 8000)

### Volumes

- `./api:/app:rw` - API code (hot-reload enabled)
- `./data:/app/data:rw` - Persistent data (database & visualizations)
- `./webapp:/usr/share/nginx/html:ro` - Frontend files
- `./nginx/nginx.conf:/etc/nginx/nginx.conf:ro` - Nginx configuration

### Rebuilding

```bash
# Rebuild after code changes
docker compose up --build

# Rebuild specific service
docker compose build api

# View logs
docker compose logs -f api

# Restart services
docker compose restart

# Clean rebuild
docker compose down
docker compose up --build -d
```

### Data Persistence

The `data/` directory is mounted as a volume, ensuring:
- Database persists across container restarts
- Visualizations are kept
- Easy backup by copying the `data/` folder

---

## 🔍 Algorithms Used

### Image Registration
- **Bounding Box Analysis**: Calculates optimal pre-scaling from black pixel distribution
- **Brute-Force Search**: Exhaustively tests rotation/scale/translation combinations
- **Overlap Scoring**: Uses intersection of black pixels for quality metric
- **Similarity Transform**: Applies rotation + scale (preserves shape, no shear)
- **Skeletonization**: Morphological thinning to 1-pixel lines (`skimage.morphology.skeletonize`)
- **Gaussian Filtering**: Smoothing for anti-aliasing

### Line Detection (Iterative Method)
- **Binary Thresholding**: Strong black/white separation (threshold=127)
- **Probabilistic Hough Transform**: Detects line segments with configurable sensitivity
- **Longest-First Selection**: Prioritizes important/major lines
- **Morphological Dilation**: Expands detected line masks to capture nearby pixels
- **Pixel Subtraction**: Removes detected lines from image to prevent duplicates
- **Multi-Pass Strategy**: Two-phase detection (strict → relaxed) for completeness
- **Crossing Detection**: Special handling for X-patterns using angle analysis

### Comparison Algorithm (Hungarian Method)
- **Multi-Metric Similarity**: Combines position, angle, and length
- **Hungarian Algorithm**: Optimal bipartite matching (`scipy.optimize.linear_sum_assignment`)
- **Cost Matrix**: Precomputes similarity for all line pairs
- **Global Optimization**: Finds best overall assignment (not greedy)
- **Length Filtering**: Ignores extra lines <30px (noise reduction)

---

## ✅ Recent Improvements

- [x] **Duplicate Detection**: SHA256 hash-based duplicate checking
- [x] **Smart Data Storage**: Original and processed images stored separately
- [x] **Cascading Deletes**: Deleting evaluations cleans up orphaned data
- [x] **Enhanced UX**: Spinner overlays, button states, status feedback
- [x] **Evaluations Filtering**: Direct link to unevaluated entries
- [x] **Admin Tools**: Migration endpoints for database updates
- [x] **Modular API Architecture**: Refactored main.py (1323→75 lines) into focused router modules
- [x] **Clean Code Structure**: Organized endpoints by domain (admin, upload, evaluations, references, test_images)
- [x] **MATLAB .mat Extraction**: Automated tool for extracting reference and drawn images from neuropsychological assessment data
- [x] **OCS Image Extraction**: Red-pixel isolation from human rating images with noise removal
- [x] **Auto-Cropping**: Intelligent content-aware cropping with configurable padding (5px default)
- [x] **Line Thickness Normalization**: Zhang-Suen skeletonization + dilation for consistent 2.00px lines
- [x] **Batch Processing**: Process multiple `.mat` files and OCS images with unified resolution configuration (default: 568×274)
- [x] **Drawing Reconstruction**: Convert stroke data (x, y, timestamp) to visual PNG images
- [x] **Red Pixel Detection**: Configurable RGB thresholds for isolating hand-drawn annotations
- [x] **CNN-Optimized Output**: Consistent resolution, line thickness, and minimal white space for machine learning
- [x] **Dual Extraction Pipelines**: Separate tools for machine recordings (MAT) and human ratings (OCS)
- [x] **Comprehensive Documentation**: Detailed Docker commands, configuration guides, and troubleshooting

## 🚧 Future Enhancements

- [ ] Neural network for tolerating drawing variations
- [ ] Multiple reference templates support
- [ ] Template difficulty levels (easy, medium, hard)
- [ ] Export results as JSON/CSV
- [ ] Real-time drawing evaluation (webcam integration)
- [ ] User accounts and drawing history
- [ ] Mobile app integration
- [ ] Drawing tutorials and step-by-step hints
- [ ] Batch processing for multiple images
- [ ] Advanced statistics and analytics dashboard
- [ ] Custom reference image upload via UI

---

## 📝 License

MIT License - feel free to use and modify for your projects.

---

## 👤 Author

**Stefan Brodoehl**  
Date: October 2025  
Version: 1.0

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Algorithm optimization and fine-tuning
- Additional image processing techniques
- UI/UX enhancements
- Test coverage and automated testing
- Documentation and tutorials
- Performance optimization
- Multi-language support

---

## 📞 Support

For issues or questions:
- Check the API documentation at http://localhost/api/docs
- Review the code comments and docstrings
- Test with the provided examples
- Inspect the database with: `sqlite3 data/npsketch.db`

---

**Happy Sketching! 🎨**
