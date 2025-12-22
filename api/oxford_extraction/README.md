# Oxford Dataset Extraction

This folder contains scripts for importing the Oxford manual rater dataset into the NPSketch training database. The Oxford dataset consists of pre-processed PNG images with clinical scores stored in a CSV file.

---

## 📁 Contents

### Core Scripts

1. **`oxford_normalizer.py`** - Image Normalization
   - Normalizes PNG images to 568×274px with 2.00px line thickness
   - Auto-crops images to content with 5px padding
   - Uses Zhang-Suen skeletonization + dilation (same as MAT/OCS extractors)
   - Processes entire directories of images

2. **`oxford_db_populator.py`** - Database Import
   - Reads CSV file with patient IDs, conditions (COPY/RECALL), and TotalScore
   - Matches CSV rows to original and normalized images
   - Creates database entries with both original and processed images
   - Includes duplicate detection and error handling

### Validation Scripts

3. **`validate_oxford_entries.py`** - Quick Validation
   - View database entries for a specific session
   - Check basic field correctness
   - Display metadata and features

4. **`validate_oxford_import.py`** - Comprehensive Validation
   - Validates database entries against:
     - Original images in `imgs/` directory
     - Normalized images in `imgs_normalized_568x274/` directory
     - CSV TotalScore values
     - Image hashes and file integrity
   - Reports missing entries and inconsistencies

---

## 🚀 Workflow

### Step 1: Normalize Images

Before importing to the database, images must be normalized to match the format used by MAT/OCS extractors (568×274px, 2px lines).

```bash
docker exec npsketch-api python3 /app/oxford_extraction/oxford_normalizer.py \
  /app/templates/training_data_oxford_manual_rater_202512/imgs \
  /app/templates/training_data_oxford_manual_rater_202512/imgs_normalized_568x274
```

**What it does:**
- Processes all PNG files in the input directory
- Auto-crops each image to content (5px padding)
- Resizes to 568×274 pixels
- Normalizes line thickness to 2.00px
- Saves normalized images to output directory

**Output:**
- Normalized images: `imgs_normalized_568x274/{ID}_{Cond}.png`
- Progress logging with ETA and statistics

### Step 2: Import to Database

Import normalized images with CSV labels into the `training_data_images` table.

```bash
# Test run (first 5 files)
docker exec npsketch-api python3 /app/oxford_extraction/oxford_db_populator.py --test

# Full import
docker exec npsketch-api python3 /app/oxford_extraction/oxford_db_populator.py

# Limited import (first N files)
docker exec npsketch-api python3 /app/oxford_extraction/oxford_db_populator.py --limit 50
```

**What it does:**
- Reads `Rater1_simple.csv` (columns: ID, Cond, TotalScore)
- For each CSV row:
  - Loads original image from `imgs/`
  - Loads normalized image from `imgs_normalized_568x274/`
  - Calculates SHA256 hash for duplicate detection
  - Creates database entry with:
    - `patient_id`: From CSV ID column
    - `task_type`: "COPY" or "RECALL" (from CSV Cond)
    - `source_format`: "OXFORD"
    - `original_file_data`: Original PNG (BLOB)
    - `processed_image_data`: Normalized PNG (BLOB)
    - `features_data`: `{"Total_Score": <value>}` from CSV
    - `extraction_metadata`: Normalization details (JSON)

**Output:**
- Database entries in `training_data_images` table
- Statistics: success, errors, duplicates, skipped
- Session ID for tracking: `oxford_YYYYMMDD`

### Step 3: Validate Import

Verify that the import was successful and data is correct.

```bash
# Quick validation (view entries)
docker exec npsketch-api python3 /app/oxford_extraction/validate_oxford_entries.py

# Comprehensive validation (check everything)
docker exec npsketch-api python3 /app/oxford_extraction/validate_oxford_import.py
```

**What it checks:**
- ✅ All TotalScore values match CSV
- ✅ All original files exist and match database
- ✅ All normalized files exist and match database
- ✅ Image hashes are correct
- ✅ Resolution is 568×274px
- ✅ Line thickness is 2.00px
- ✅ Metadata is complete

---

## 📋 Example: Complete Import Workflow

```bash
# 1. Normalize all images
docker exec npsketch-api python3 /app/oxford_extraction/oxford_normalizer.py \
  /app/templates/training_data_oxford_manual_rater_202512/imgs \
  /app/templates/training_data_oxford_manual_rater_202512/imgs_normalized_568x274

# Output:
# Processing 906 images...
# [1/906] Processing C0078_COPY.png... ✓
# [2/906] Processing C0078_RECALL.png... ✓
# ...
# Complete: 891 successful, 15 errors

# 2. Test import with first 5 files
docker exec npsketch-api python3 /app/oxford_extraction/oxford_db_populator.py --test

# Output:
# Processing first 5 RECALL rows...
# [1/5] Processing C0078_RECALL.png... ✓ Success! Database ID: 45
# ...
# Total processed: 5
# Success: 5, Errors: 0

# 3. Validate test entries
docker exec npsketch-api python3 /app/oxford_extraction/validate_oxford_entries.py

# 4. If validation passes, run full import
docker exec npsketch-api python3 /app/oxford_extraction/oxford_db_populator.py

# 5. Comprehensive validation
docker exec npsketch-api python3 /app/oxford_extraction/validate_oxford_import.py
```

---

## 📊 Data Format

### CSV File: `Rater1_simple.csv`

```csv
ID,Cond,TotalScore
C0078,COPY,60
C0078,RECALL,43
C0083,COPY,60
C0083,RECALL,46
...
```

**Columns:**
- `ID`: Patient identifier (e.g., "C0078")
- `Cond`: Condition ("COPY" or "RECALL")
- `TotalScore`: Clinical score (0-60)

### Image Files

**Original:** `imgs/{ID}_{Cond}.png`
- Various resolutions
- Black lines on white background

**Normalized:** `imgs_normalized_568x274/{ID}_{Cond}.png`
- Fixed resolution: 568×274 pixels
- Line thickness: 2.00px
- Auto-cropped with 5px padding

### Database Entry

```python
TrainingDataImage(
    patient_id="C0078",
    task_type="COPY",
    source_format="OXFORD",
    original_filename="C0078_COPY.png",
    original_file_data=<BLOB: original PNG>,
    processed_image_data=<BLOB: normalized 568×274 PNG>,
    image_hash="6493e5c7a6a3762f...",
    extraction_metadata='{"width": 568, "height": 274, "line_thickness": 2.0, ...}',
    features_data='{"Total_Score": 60}',
    session_id="oxford_20251222"
)
```

---

## 🔧 Configuration

### Paths (in `oxford_db_populator.py`)

```python
BASE_DIR = Path("/app/templates/training_data_oxford_manual_rater_202512")
CSV_PATH = BASE_DIR / "Rater1_simple.csv"
ORIGINAL_IMGS_DIR = BASE_DIR / "imgs"
NORMALIZED_IMGS_DIR = BASE_DIR / "imgs_normalized_568x274"
```

### Normalization Parameters (in `oxford_normalizer.py`)

```python
target_size = (568, 274)      # Output resolution
auto_crop = True              # Auto-crop to content
padding = 5                   # Padding in pixels
target_thickness = 2.0        # Line thickness in pixels
```

---

## ⚠️ Important Notes

1. **Order Matters:** Always normalize images before importing to database
2. **Duplicate Detection:** Scripts check for duplicate images by SHA256 hash
3. **Error Handling:** Missing files are skipped with error messages
4. **Session ID:** All entries from one import share the same session ID
5. **Read-Only Templates:** `/app/templates` is read-only; normalized images must be written there before import

---

## 🐛 Troubleshooting

### Images not found
- Check that images exist in `imgs/` directory
- Verify filenames match CSV: `{ID}_{Cond}.png`

### Normalization errors
- Some images may be corrupted (15 files in original dataset)
- Check logs for specific error messages

### Database import errors
- Ensure normalized images exist before running populator
- Check CSV file format (must have ID, Cond, TotalScore columns)
- Verify database connection

### Validation failures
- Run `validate_oxford_import.py` for detailed error messages
- Check that all files are accessible
- Verify CSV matches database entries

---

## 📚 Related Documentation

- **Main README:** `/README.md` - Full project documentation
- **Data Import Flow:** `/DATA_IMPORT_FLOW.md` - All import methods
- **Agent Guide:** `/AGENTS.md` - Quick reference for AI agents
- **Oxford Summary:** `/templates/training_data_oxford_manual_rater_202512/OXFORD_IMPORT_SUMMARY.md`

---

**Last Updated:** 2025-12-22  
**Location:** `/app/oxford_extraction/` (in Docker container)

