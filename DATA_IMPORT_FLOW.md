# Data Import Flow Documentation

## Overview

NPSketch supports importing training data from three sources:
1. **MAT Files** (`.mat`) - MATLAB files containing machine tablet recordings
2. **OCS Images** (`.png`/`.jpg`) - Human expert rating images with red-pixel annotations
3. **Oxford Dataset** (`.png` + `.csv`) - Pre-processed PNG images with CSV labels (TotalScore)

All workflows produce normalized images (568×274px, 2px line thickness) and store them in the `training_data_images` database table.

---

## Frontend Flow

### Entry Point: `webapp/ai_training_data_upload.html`

**Step 1: Format Selection**
- User selects either "MATLAB .mat Files" or "OCS PNG/JPG Images"
- Frontend sets `selectedFormat` variable (`'mat'` or `'ocs'`)
- File input `accept` attribute is updated:
  - MAT: `.mat` only
  - OCS: `.png,.jpg,.jpeg`

**Step 2: File Selection**
- User can:
  - Click to browse files
  - Drag & drop files
- Files are validated:
  - MAT: Must end with `.mat`
  - OCS: Must match `/\.(png|jpg|jpeg)$/i`
- Valid files added to `selectedFiles` array

**Step 3: Processing**
```javascript
async function processFiles() {
    const formData = new FormData();
    formData.append('format', selectedFormat);  // 'mat' or 'ocs'
    
    selectedFiles.forEach(file => {
        formData.append('files', file);  // Multiple files supported
    });
    
    const response = await fetch('/api/extract-training-data', {
        method: 'POST',
        body: formData
    });
}
```

**Step 4: Results Display**
- Shows statistics: total files, success, duplicates, errors
- Lists extracted images with patient ID and task type
- Provides link to view all training data

---

## Backend Flow

### Entry Point: `api/routers/training_data.py`

**Endpoint:** `POST /api/extract-training-data`

**Request:**
- `format`: `'mat'` or `'ocs'` (Form field)
- `files`: List of `UploadFile` objects

**Processing Steps:**

1. **Validation**
   - Format must be `'mat'` or `'ocs'`
   - At least one file required

2. **Session Creation**
   - Unique session ID: `YYYYMMDD_HHMMSS_microseconds`
   - Temporary directories:
     - Upload: `/app/data/tmp/uploads/{session_id}`
     - Output: `/app/data/tmp/extracted/{session_id}`

3. **Per-File Processing Loop**

   **a) Duplicate Detection**
   
   **MAT Files:**
   - Check if any image from this exact `.mat` file already exists
   - Compare by `original_filename` + SHA256 hash of original file
   - If duplicate: Skip entire file (both COPY and RECALL)
   
   **OCS Files:**
   - Check by SHA256 hash of original image file
   - If duplicate: Skip file

   **b) File Storage**
   - Save original file to temporary upload directory
   - Create per-file output directory: `file_{index}`

   **c) Extraction**
   - **MAT:** Call `run_mat_extractor()` → subprocess to `mat_extractor.py`
   - **OCS:** Call `run_ocs_extractor()` → subprocess to `ocs_extractor.py`

   **d) Process Extracted Images**
   - Find all `.png` files in output directory
   - For each extracted image:
     - Extract metadata: `patient_id`, `task_type` from filename
     - Skip REFERENCE images (MAT only - we only want COPY/RECALL)
     - Calculate image hash:
       - **MAT:** Hash of processed image (each COPY/RECALL is unique)
       - **OCS:** Hash of original file
     - Check for duplicate image by hash
     - Read processed image (568×274, 2px lines)
     - Get image dimensions
     - Create metadata JSON
     - Save to database

4. **Database Storage**
   ```python
   TrainingDataImage(
       patient_id=patient_id,           # e.g., "PC56", "Park_16"
       task_type=task_type,             # "COPY", "RECALL", "REFERENCE"
       source_format=format.upper(),    # "MAT", "OCS"
       original_filename=uploaded_file.filename,
       original_file_data=original_content,      # Full original file (BLOB)
       processed_image_data=processed_content,  # 568×274 PNG (BLOB)
       image_hash=image_hash,                   # SHA256
       extraction_metadata=json.dumps(metadata),
       session_id=session_id
   )
   ```

5. **Response**
   ```json
   {
       "success": true,
       "session_id": "20251113_143211_123456",
       "statistics": {
           "total_files": 5,
           "success": 4,
           "duplicates": 1,
           "errors": 0,
           "total_images_extracted": 8
       },
       "results": [
           {
               "original_filename": "PC56.mat",
               "status": "success",
               "message": "Extracted and saved 2 image(s) (COPY + RECALL only)",
               "extracted_images": [
                   {
                       "id": 123,
                       "patient_id": "PC56",
                       "task_type": "COPY",
                       "filename": "PC56_COPY_drawn_20251111.png",
                       "width": 568,
                       "height": 274
                   },
                   ...
               ]
           },
           ...
       ]
   }
   ```

6. **Cleanup**
   - Remove temporary upload directory
   - Keep extracted images in output directory (for inspection)
   - Clean up any loose PNG files in tmp root

---

## MAT File Extractor

### Script: `api/mat_extraction/mat_extractor.py`

**Command:**
```bash
python3 /app/mat_extraction/mat_extractor.py \
    --input <input_dir> \
    --output <output_dir> \
    --config /app/mat_extraction/mat_extractor.conf
```

**Processing:**

1. **Load MAT File**
   - Uses `scipy.io.loadmat()` to read MATLAB `.mat` file
   - Extracts data structures:
     - `data_complex_copy` → COPY condition
     - `data_complex_memory_copy` → RECALL condition

2. **Extract Reference Image** (optional, usually skipped)
   - From `data_complex_copy.figs` or `data_complex_memory_copy.figs`
   - Output: `{patient_id}_REFERENCE_{timestamp}.png`

3. **Extract Drawn Lines** (main output)
   - From `data_complex_copy.trails.cont_lines` (COPY)
   - From `data_complex_memory_copy.trails.cont_lines` (RECALL)
   - Drawing area from `data_complex_copy.draw_area.rect`

4. **Rendering Process**
   - Calculate bounding box from line coordinates
   - Auto-crop with 5px padding
   - Render lines on temporary canvas
   - Resize to 568×274 (landscape)
   - Normalize line thickness to 2.00px:
     - Zhang-Suen skeletonization (1px)
     - Dilation to 2px

5. **Output Files**
   - `{patient_id}_COPY_drawn_{timestamp}.png`
   - `{patient_id}_RECALL_drawn_{timestamp}.png`
   - `{patient_id}_REFERENCE_{timestamp}.png` (if extracted, but usually skipped)

**Configuration (`mat_extractor.conf`):**
```json
{
    "canvas_width": 568,
    "canvas_height": 274,
    "auto_crop": true,
    "padding_px": 5
}
```

**Key Features:**
- Extracts XY coordinates from MATLAB structures
- Handles both COPY and RECALL conditions
- Auto-crops to content with padding
- Normalizes line thickness for CNN compatibility
- Each MAT file produces 2 images (COPY + RECALL)

---

## OCS Image Extractor

### Script: `api/ocs_extraction/ocs_extractor.py`

**Command:**
```bash
python3 /app/ocs_extraction/ocs_extractor.py \
    --input <input_dir> \
    --output <output_dir> \
    --config /app/ocs_extraction/ocs_extractor.conf
```

**Processing:**

1. **Load Image**
   - Read PNG/JPG file using PIL
   - Convert to RGB if needed

2. **Red Pixel Detection**
   - Threshold: `R ≥ 200, G ≤ 100, B ≤ 100`
   - Creates binary mask of red pixels
   - Removes grids, reference figures, annotations (non-red content)

3. **Extract Drawing**
   - Extract only red pixels (the actual drawing)
   - Convert to black lines on white background
   - Remove all non-red content

4. **Normalization**
   - Auto-crop to content with 5px padding
   - Resize to 568×274 (landscape)
   - Normalize line thickness to 2.00px:
     - Zhang-Suen skeletonization (1px)
     - Dilation to 2px

5. **Output File**
   - `{patient_id}_{task_type}_ocs_{timestamp}.png`
   - Task type extracted from filename: `*_COPY.png` → COPY, `*_RECALL.png` → RECALL

**Configuration (`ocs_extractor.conf`):**
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

**Key Features:**
- Extracts red-pixel annotations from human rating images
- Removes all non-red content (grids, reference figures)
- One image per file (COPY or RECALL)
- Auto-crops and normalizes for CNN compatibility

---

## Oxford Dataset Import

### Overview

Oxford dataset consists of pre-processed PNG images that are already black lines (no color extraction needed) with labels stored in a CSV file. This is a **command-line import** method (not via web interface).

### Script: `api/oxford_extraction/oxford_db_populator.py`

**Command:**
```bash
# Test run with first 5 files
python3 /app/oxford_extraction/oxford_db_populator.py --test

# Full import
python3 /app/oxford_extraction/oxford_db_populator.py

# Limited import
python3 /app/oxford_extraction/oxford_db_populator.py --limit 50
```

### Prerequisites

1. **Normalized Images**: Images must be normalized to 568×274px first
   ```bash
   python3 /app/oxford_extraction/oxford_normalizer.py \
     /app/templates/training_data_oxford_manual_rater_202512/imgs \
     /app/templates/training_data_oxford_manual_rater_202512/imgs_normalized_568x274
   ```

2. **CSV File**: `Rater1_simple.csv` with columns:
   - `ID`: Patient identifier (e.g., "C0078")
   - `Cond`: Condition ("COPY" or "RECALL")
   - `TotalScore`: Clinical score (0-60)

### Processing:

1. **Load CSV Data**
   - Read `Rater1_simple.csv`
   - Create mapping: `{ID}_{Cond}` → TotalScore

2. **Process Each Image**
   - For each CSV row:
     - Construct filename: `{ID}_{Cond}.png`
     - Load original image from `imgs/`
     - Load normalized image from `imgs_normalized_568x274/`
     - Calculate SHA256 hash of normalized image
     - Check for duplicates
     - Create database entry

3. **Database Storage**
   ```python
   TrainingDataImage(
       patient_id=patient_id,           # e.g., "C0078"
       task_type=task_type,             # "COPY" or "RECALL"
       source_format="OXFORD",
       original_filename=filename,      # e.g., "C0078_COPY.png"
       original_file_data=original_data,      # Original PNG (BLOB)
       processed_image_data=processed_data,  # Normalized 568×274 PNG (BLOB)
       image_hash=image_hash,                 # SHA256 of processed image
       extraction_metadata=json.dumps({
           "width": 568,
           "height": 274,
           "line_thickness": 2.0,
           "auto_crop": True,
           "padding_px": 5,
           "source": "Oxford manual rater dataset"
       }),
       features_data=json.dumps({
           "Total_Score": total_score
       }),
       session_id="oxford_YYYYMMDD"
   )
   ```

**Key Features:**
- Images already black lines (no color extraction)
- Labels from CSV file (TotalScore)
- One image per CSV row (COPY or RECALL)
- Both original and normalized images stored
- Automatic duplicate detection by image hash

### Normalization Process

Before importing, images must be normalized using `oxford_extraction/oxford_normalizer.py`:

1. **Auto-crop**: Find bounding box of content, add 5px padding
2. **Resize**: Scale to 568×274 (landscape)
3. **Normalize line thickness**: Zhang-Suen skeletonization + dilation to 2.00px

This matches exactly the process used by MAT and OCS extractors.

---

## Database Schema

### Table: `training_data_images`

**Core Fields:**
- `id`: Primary key
- `patient_id`: Patient identifier (e.g., "PC56", "Park_16", "C0078")
- `task_type`: "COPY", "RECALL", or "REFERENCE"
- `source_format`: "MAT", "OCS", "OXFORD", or "DRAWN"
- `original_filename`: Original uploaded filename
- `original_file_data`: Full original file (BLOB)
- `processed_image_data`: Normalized 568×274 PNG (BLOB)
- `image_hash`: SHA256 hash for duplicate detection
- `extraction_metadata`: JSON with technical details
- `session_id`: Upload session identifier
- `uploaded_at`: Timestamp

**Optional Fields:**
- `ground_truth_correct`: Expected correct lines (for algorithmic evaluation)
- `ground_truth_extra`: Expected extra lines (for algorithmic evaluation)
- `features_data`: JSON with clinical scores (for CNN training)

---

## Key Differences: MAT vs OCS vs Oxford

| Aspect | MAT Files | OCS Images | Oxford Dataset |
|--------|-----------|------------|----------------|
| **Input** | `.mat` MATLAB files | `.png`/`.jpg` image files | `.png` + `.csv` |
| **Source** | Machine tablet recordings | Human expert ratings | Manual rater (pre-processed) |
| **Images per File** | 2 (COPY + RECALL) | 1 (COPY or RECALL) | 1 (COPY or RECALL) |
| **Extraction Method** | XY coordinates from MATLAB structures | Red pixel detection | Direct normalization (already black) |
| **Labels** | None (separate CSV) | None (separate CSV) | **In CSV (TotalScore)** |
| **Duplicate Check** | Per MAT file (by filename + hash) | Per image (by file hash) | Per image (by processed image hash) |
| **Image Hash** | Hash of processed image | Hash of original file | Hash of processed image |
| **REFERENCE Images** | Extracted but usually skipped | Not applicable | Not applicable |
| **Processing** | Render lines from coordinates | Extract red pixels | Normalize existing black lines |
| **Output Format** | `{patient_id}_{task_type}_drawn_{timestamp}.png` | `{patient_id}_{task_type}_ocs_{timestamp}.png` | `{patient_id}_{task_type}.png` |
| **Import Method** | Web interface | Web interface | **Command-line script** |

---

## Unified Output Format

All three import methods produce identical image characteristics:

- **Resolution:** 568×274 pixels (landscape)
- **Line Thickness:** 2.00px (normalized via Zhang-Suen + dilation)
- **Format:** Black lines on white background (RGB PNG)
- **Margins:** ~5-7px white border (from auto-cropping with padding)
- **Purpose:** CNN training compatibility

---

## Error Handling

**Duplicate Detection:**
- MAT: If MAT file already processed → Skip entire file
- OCS: If image hash exists → Skip file
- Per-image: If processed image hash exists → Skip that specific image

**Error Cases:**
- Invalid file format → Error status
- Extraction fails → Error status
- No images extracted → Error status
- File read error → Error status

**Cleanup:**
- Temporary upload directories removed after processing
- Extracted images kept in output directory for inspection
- Loose PNG files in tmp root cleaned up

---

## Example Workflow

### MAT File Upload:

1. User selects "MATLAB .mat Files"
2. User uploads: `PC56.mat`, `PC0460.mat`
3. Backend processes:
   - `PC56.mat` → Extracts COPY + RECALL → 2 images saved
   - `PC0460.mat` → Extracts COPY + RECALL → 2 images saved
4. Total: 4 images in database

### OCS Image Upload:

1. User selects "OCS PNG/JPG Images"
2. User uploads: `Park_16_COPY.png`, `Park_16_RECALL.png`, `TeamD178_COPY.png`
3. Backend processes:
   - `Park_16_COPY.png` → Extracts red pixels → 1 image saved
   - `Park_16_RECALL.png` → Extracts red pixels → 1 image saved
   - `TeamD178_COPY.png` → Extracts red pixels → 1 image saved
4. Total: 3 images in database

### Oxford Dataset Import:

1. **Normalize images** (if not already done):
   ```bash
   python3 /app/oxford_extraction/oxford_normalizer.py \
     /app/templates/training_data_oxford_manual_rater_202512/imgs \
     /app/templates/training_data_oxford_manual_rater_202512/imgs_normalized_568x274
   ```

2. **Import to database**:
   ```bash
   python3 /app/oxford_extraction/oxford_db_populator.py
   ```

3. Script processes:
   - Reads `Rater1_simple.csv` (962 rows)
   - Matches CSV rows to PNG files
   - Loads original + normalized images
   - Creates database entries with TotalScore from CSV
4. Total: 889 images in database (with TotalScore labels)

---

## API Endpoints

**Main Upload:**
- `POST /api/extract-training-data` - Upload and extract MAT/OCS files
- **Oxford**: Command-line scripts (`oxford_extraction/oxford_normalizer.py` + `oxford_extraction/oxford_db_populator.py`) - Direct database import

**View Data:**
- `GET /api/training-data-images` - List all training images
- `GET /api/training-data-image/{id}/original` - Get original file
- `GET /api/training-data-image/{id}/processed` - Get processed image

**Features:**
- `POST /api/training-data-features-upload` - Upload CSV with clinical scores

---

## Frontend Pages

- **Upload:** `http://localhost/ai_training_data_upload.html`
- **View Data:** `http://localhost/ai_training_data_view.html`
- **Training:** `http://localhost/ai_training_train.html`

