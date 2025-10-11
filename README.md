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
   - **Landing Page**: http://localhost
   - **Main Application**: http://localhost/app.html
   - **Upload Interface**: http://localhost/upload.html
   - **API Documentation**: http://localhost/api/docs
   - **API Direct Access**: http://localhost:8000/api/docs

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
- **SQLite** for database storage and metadata management
- **Nginx** as reverse proxy and static file server

### 2. Reference Image Initialization

On startup, the system:
- Checks if the SQLite database exists, otherwise initializes it
- Loads the reference image (default: "House of Nikolaus" pattern)
- Normalizes the reference to 512×512 pixels
- Uses OpenCV to extract line features (edges, contours)
- Stores extracted reference features and images as BLOBs in the database

### 3. Image Upload & Processing

When you upload an image:
1. **Normalization**: Image is resized to 512×512 with aspect ratio preservation and padding
2. **Preprocessing**: Converted to grayscale, Gaussian blur applied, adaptive thresholding
3. **Line Detection**: Probabilistic Hough Transform detects line segments
4. **Feature Extraction**: Lines, angles, and lengths are extracted and stored
5. **Storage**: Original and processed images stored in database

### 4. Comparison & Evaluation

The system compares detected lines to reference lines using:

- **Position Similarity**: Euclidean distance between line midpoints (tolerance: 20px)
- **Angle Similarity**: Angle difference in degrees (tolerance: 15°)
- **Length Similarity**: Relative length ratio (tolerance: 30%)
- **Weighted Similarity**: Combines metrics (40% position, 30% angle, 30% length)

Metrics calculated:
- **Correct Lines**: Lines that match the reference (similarity > 70%)
- **Missing Lines**: Reference lines not found in drawing
- **Extra Lines**: Lines drawn but not in reference
- **Similarity Score**: Overall accuracy (0-100%)

### 5. Visualization and Debugging

Each processed image generates an overlay visualization where detected lines are highlighted:

- 🟢 **Green**: Matched lines (present in both)
- 🔴 **Red**: Missing lines (in reference only)
- 🔵 **Blue**: Extra lines (in upload only)

Visualizations are stored in `data/visualizations/` and accessible through the API.

### 6. Database Schema

**Tables:**

1. **reference_images**
   - Stores reference templates and their features
   - Fields: id, name (unique), image_data (BLOB), processed_image_data (BLOB), feature_data (JSON), width, height, created_at

2. **uploaded_images**
   - Stores uploaded drawings
   - Fields: id, filename, image_data (BLOB), processed_image_data (BLOB), uploader, uploaded_at

3. **extracted_features**
   - Stores detected line features
   - Fields: id, image_id (FK), feature_data (JSON), num_lines, extracted_at

4. **evaluation_results**
   - Stores comparison results
   - Fields: id, image_id (FK), reference_id (FK), correct_lines, missing_lines, extra_lines, similarity_score, visualization_path, evaluated_at

---

## 🔧 Configuration

### Line Detection Parameters

Edit `api/image_processing/line_detector.py`:

```python
LineDetector(
    rho=1.0,              # Distance resolution (pixels)
    theta=np.pi/180,      # Angle resolution (radians)
    threshold=50,         # Minimum votes for line
    min_line_length=30,   # Minimum line length (pixels)
    max_line_gap=10       # Max gap between segments (pixels)
)
```

### Comparison Tolerance

Edit `api/image_processing/comparator.py`:

```python
LineComparator(
    position_tolerance=20.0,  # Max position difference (pixels)
    angle_tolerance=15.0,     # Max angle difference (degrees)
    length_tolerance=0.3      # Max length difference (ratio)
)
```

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

### Example: Upload Image

```bash
curl -X POST "http://localhost/api/upload" \
  -F "file=@my_drawing.png" \
  -F "uploader=John" \
  -F "reference_name=default_reference"
```

Response:
```json
{
  "success": true,
  "message": "Image uploaded and evaluated successfully",
  "image_id": 1,
  "evaluation": {
    "id": 1,
    "correct_lines": 8,
    "missing_lines": 2,
    "extra_lines": 1,
    "similarity_score": 0.8,
    "visualization_path": "/api/visualizations/eval_1.png",
    "evaluated_at": "2025-10-11T12:00:00"
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

### Line Detection
- **Gaussian Blur**: Reduces noise before edge detection
- **Adaptive Thresholding**: Converts to binary image
- **Canny Edge Detection**: Identifies edges in the image
- **Probabilistic Hough Transform**: Detects line segments efficiently
- **Contour Detection**: Additional feature extraction

### Comparison Algorithm
- **Euclidean Distance**: For position matching (line midpoints)
- **Angular Distance**: For orientation matching (line angles)
- **Length Ratio**: For size matching (relative lengths)
- **Weighted Similarity**: Combines all metrics with configurable weights
- **Greedy Matching**: Finds best matches between detected and reference lines

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
