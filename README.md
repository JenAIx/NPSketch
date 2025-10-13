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
│   ├── main.py                  # Main application with endpoints
│   ├── database.py              # SQLAlchemy models and setup
│   ├── models.py                # Pydantic request/response models
│   ├── image_processing/        # Image processing library
│   │   ├── line_detector.py     # Line detection using Hough Transform
│   │   ├── comparator.py        # Line comparison and similarity metrics
│   │   └── utils.py             # Image preprocessing utilities
│   ├── services/                # Business logic layer
│   │   ├── reference_service.py # Reference image management
│   │   └── evaluation_service.py# Evaluation workflow
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile               # API container definition
├── webapp/                      # Frontend (served by nginx)
│   ├── index.html               # Landing page
│   ├── app.html                 # Main application
│   └── upload.html              # Upload interface
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
   - **Reference Editor**: http://localhost/reference.html (define reference lines manually)
   - **Draw Test Images**: http://localhost/draw_testimage.html (create test dataset)
   - **Run Tests**: http://localhost/run_test.html (automated testing with metrics)
   - **Evaluations**: http://localhost/evaluations.html (view past evaluations)
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

### 4. Image Upload & Processing (Automated)

When you upload an image (or run tests):

**Stage 1: Normalization & Registration**
1. **Resize**: Image is resized to 256×256 (matches reference)
2. **Registration** (optional, enabled by default):
   - **Pre-Scaling**: Calculates optimal scale based on bounding box (min of height/width ratio)
   - **Centering**: Places image on white canvas to prevent clipping
   - **Brute-Force Search**: Tests rotations (-30° to +30°, 3° steps) and scales (0.75-1.3x, 0.05 steps)
   - **Translation Search**: Tests small translations (-10 to +10px, 5px steps)
   - **Overlap Scoring**: Calculates intersection of black pixels
   - **Best Match**: Applies transformation with highest overlap score
   - **Line Thinning**: Uses skeletonization to reduce thick lines to 1px (if scaled >1.1x)

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
   - Fields: id, filename, image_data (BLOB), processed_image_data (BLOB), uploader, uploaded_at

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
| `/api/upload` | POST | Upload and evaluate image |
| `/api/evaluations/recent` | GET | List recent evaluations |
| `/api/evaluations/{id}` | GET | Get specific evaluation |
| `/api/references` | GET | List reference images |
| `/api/references/{id}/image` | GET | Get reference image data |
| `/api/visualizations/{file}` | GET | Get visualization image |
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

**ste**  
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
