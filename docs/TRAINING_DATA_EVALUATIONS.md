# Training Data Evaluations System

## Overview

The Training Data Evaluations system allows you to test the algorithmic line detection approach by comparing automated detection results against ground truth values from CNN-optimized images.

## Purpose

- **Evaluate** the algorithmic line detection performance
- **Compare** automated detection vs manually verified ground truth
- **Measure** algorithm accuracy on real patient data
- **Tune** detection parameters based on performance metrics

## Key Difference from Old Evaluations System

### Old System (`evaluations.html`)
- Works with **UploadedImage** + **EvaluationResult** tables
- Automated detection runs **at upload time**
- Results **stored permanently** in database
- Purpose: Manual review and correction of individual drawings

### New System (`training_evaluations.html`)
- Works with **TrainingDataImage** table (CNN-optimized)
- Automated detection runs **on-demand** (not at upload)
- Results **NOT stored** in database (ephemeral)
- Purpose: Evaluate algorithmic performance against ground truth

## Database Schema

### TrainingDataImage Table
```python
class TrainingDataImage(Base):
    id: int                             # Primary key
    patient_id: str                      # e.g., "PC56", "Park_16"
    task_type: str                       # COPY, RECALL, REFERENCE, DRAWN
    source_format: str                   # MAT, OCS, DRAWN
    
    # Image data (CNN-optimized: 568×274, 2px lines)
    processed_image_data: bytes          # Normalized image
    
    # Ground truth (manually verified)
    ground_truth_correct: int            # Expected correct lines
    ground_truth_extra: int              # Expected extra/wrong lines
    
    # Clinical features (for CNN training)
    features_data: str (JSON)            # Total_Score, MMSE, etc.
```

## API Endpoints

### 1. List Training Data Evaluations
```bash
GET /api/training-data-evaluations?limit=100&has_ground_truth=true
```

**Response:**
```json
{
  "total": 31,
  "offset": 0,
  "limit": 100,
  "evaluations": [
    {
      "id": 32,
      "patient_id": "PC56",
      "task_type": "COPY",
      "source_format": "MAT",
      "has_ground_truth": true,
      "ground_truth_correct": 8,
      "ground_truth_extra": 1
    }
  ]
}
```

**Filters:**
- `has_ground_truth`: true/false (filter by ground truth presence)
- `task_type`: COPY/RECALL/DRAWN
- `source_format`: MAT/OCS/DRAWN

### 2. Evaluate Training Image (Run Detection)
```bash
POST /api/training-data-image/{image_id}/evaluate
```

**Response:**
```json
{
  "image_id": 32,
  "patient_id": "PC56",
  "task_type": "COPY",
  "source_format": "MAT",
  "total_reference_lines": 11,
  
  "automated": {
    "correct_lines": 7,
    "missing_lines": 4,
    "extra_lines": 2,
    "similarity_score": 0.636
  },
  
  "ground_truth": {
    "correct_lines": 8,
    "extra_lines": 1,
    "missing_lines": 3,
    "has_ground_truth": true
  },
  
  "comparison": {
    "accuracy": 0.909,
    "details": {
      "correct_diff": 1,
      "missing_diff": 1,
      "extra_diff": 1,
      "total_error": 3,
      "max_error": 33
    }
  },
  
  "visualization_path": "/api/visualizations/test_training_32.png"
}
```

**Accuracy Calculation:**
```python
# Total error = sum of absolute differences
total_error = |detected_correct - gt_correct| + 
              |detected_missing - gt_missing| + 
              |detected_extra - gt_extra|

# Max error = reference_lines × 3 (one for each metric)
max_error = total_reference_lines × 3

# Accuracy (0.0 to 1.0)
accuracy = 1.0 - (total_error / max_error)
```

### 3. Save/Update Ground Truth
```bash
POST /api/training-data-image/{image_id}/ground-truth
Content-Type: application/json

{
  "ground_truth_correct": 8,
  "ground_truth_extra": 1
}
```

**Note:** `ground_truth_missing` is calculated automatically as:
```python
ground_truth_missing = total_reference_lines - ground_truth_correct
```

## Web Interface

### Access
```
http://localhost/training_evaluations.html
```

### Features

#### Left Panel: Image List
- **Filters:**
  - 📋 All
  - ✓ Has GT (has ground truth)
  - ⚠ No GT (no ground truth)
  - MAT / OCS / Drawn (by source)
- **Display:**
  - Patient ID, Task Type
  - Ground truth values (if set)
  - Source format badge
  - Upload timestamp

#### Right Panel: Evaluation Details
1. **Ground Truth Section**
   - Set/update ground truth values
   - Correct Lines (0-20)
   - Extra Lines (0-20)
   - Save button

2. **Evaluation Button**
   - "🔍 Run Line Detection" - triggers evaluation
   - Does NOT store results in DB
   - Shows real-time comparison

3. **Results Display**
   - **Accuracy Badge**: Overall detection accuracy
   - **3-Column Comparison:**
     - 🤖 Automated Detection
     - 👤 Ground Truth
     - 📊 Differences
   - **Visualization**: Color-coded line detection
     - 🟢 Green: Correct lines
     - 🔴 Red: Missing lines
     - 🔵 Blue: Extra lines

## Workflow

### Use Case 1: Known Ground Truth (MAT/OCS Data)
1. Upload MAT/OCS files via AI Training Data Upload
2. Images extracted with consistent format (568×274, 2px lines)
3. Expert provides ground truth values manually or via CSV
4. Navigate to Training Evaluations
5. Select image → Run Line Detection
6. Compare automated vs ground truth
7. Use accuracy metrics to tune algorithm

### Use Case 2: Manual Drawings
1. Create drawing via Draw Test Image tool
2. Save with ground truth values
3. Navigate to Training Evaluations
4. Select image → Run Line Detection
5. Verify algorithm performance

### Use Case 3: Images Without Ground Truth
1. Navigate to Training Evaluations
2. Filter by "No GT"
3. Select image
4. Set ground truth values manually
5. Run Line Detection
6. Compare and iterate

## Performance Metrics

### Reference Match Score
```python
similarity_score = correct_lines / total_reference_lines
```
- Measures how well detection matches reference
- Same as regular evaluation system

### Detection Accuracy (vs Ground Truth)
```python
accuracy = 1.0 - (total_error / max_error)

where:
  total_error = |correct_diff| + |missing_diff| + |extra_diff|
  max_error = reference_lines × 3
```
- Measures how well automated detection matches ground truth
- **Perfect match**: 1.0 (100%)
- **No match**: 0.0 (0%)

### Interpretation
- **Accuracy ≥ 90%**: Excellent detection
- **Accuracy 70-90%**: Good detection
- **Accuracy 50-70%**: Needs improvement
- **Accuracy < 50%**: Poor detection

## Algorithm Tuning

Based on evaluation results, you can tune:

### Line Detection Parameters
- `threshold`: Hough Transform threshold (18 → stricter/looser)
- `min_line_length`: Minimum line length (35px → shorter/longer)
- `max_line_gap`: Gap tolerance (35px → more/less aggressive)

### Comparison Tolerances
- `position_tolerance`: Position matching (120px)
- `angle_tolerance`: Angle matching (50°)
- `length_tolerance`: Length matching (0.8)

### Testing Process
1. Evaluate current algorithm on training data
2. Note systematic errors (e.g., always missing diagonal lines)
3. Adjust parameters
4. Re-evaluate
5. Measure improvement in accuracy metrics
6. Iterate until satisfactory

## Example Session

```bash
# 1. Check training data stats
curl http://localhost/api/training-data-evaluations | jq '.total'
# Output: 31

# 2. Evaluate image #32
curl -X POST http://localhost/api/training-data-image/32/evaluate | jq '.comparison.accuracy'
# Output: 0.968

# 3. Set ground truth for image #14
curl -X POST http://localhost/api/training-data-image/14/ground-truth \
  -H "Content-Type: application/json" \
  -d '{"ground_truth_correct": 9, "ground_truth_extra": 0}'

# 4. Re-evaluate
curl -X POST http://localhost/api/training-data-image/14/evaluate | jq
```

## Technical Notes

### Why NOT Store Automated Results?
- Training data is for **testing the algorithm**
- Automated results are **ephemeral** (change with parameter tuning)
- Ground truth is **permanent** (manually verified)
- Storing automated results would pollute training data

### Visualization Generation
- Created on-demand during evaluation
- Stored temporarily in `/app/data/visualizations/`
- Filename: `test_training_{image_id}.png`
- Same format as test image visualizations

### Reference Image
- Currently uses **default reference** (first in database)
- All training images compared against same reference
- Ensures consistent evaluation metrics

## Integration with Existing Systems

### Related Pages
- **AI Training Data Upload** (`ai_training_data_upload.html`) - Upload MAT/OCS/drawn images
- **AI Training Data View** (`ai_training_data_view.html`) - Browse all training data
- **Training Evaluations** (`training_evaluations.html`) - **NEW** - Evaluate algorithmic detection
- **Old Evaluations** (`evaluations.html`) - User correction of uploaded drawings

### Data Flow
```
MAT/OCS Files → Extract → TrainingDataImage (with ground truth)
                            ↓
                      Run Detection (on-demand)
                            ↓
                      Compare vs Ground Truth
                            ↓
                      Accuracy Metrics (ephemeral)
```

## Future Enhancements

- [ ] Batch evaluation (run on all training data)
- [ ] Statistical analysis (mean, std, distribution)
- [ ] Parameter optimization (grid search)
- [ ] Export results as CSV/JSON
- [ ] Compare multiple algorithm versions
- [ ] Visualization comparison (before/after tuning)

## Summary

The Training Data Evaluations system provides a **scientific approach** to measuring and improving the algorithmic line detection system by:

1. ✅ Using CNN-optimized images with known ground truth
2. ✅ Running detection on-demand (not stored)
3. ✅ Calculating accuracy metrics
4. ✅ Providing visual feedback
5. ✅ Enabling iterative improvement

This complements the existing evaluation system which focuses on **user correction of individual drawings** rather than **algorithm performance measurement**.

