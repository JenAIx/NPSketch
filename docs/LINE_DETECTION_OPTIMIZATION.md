# Line Detection & Testing Guide

## Overview

NPSketch v1.0 uses a **manual, iterative approach** for line detection optimization:

1. **Manual Reference Definition** - Define ground truth lines interactively
2. **Test Image Creation** - Draw test images with expected scores
3. **Iterative Line Detection** - Automatic detection with pixel subtraction
4. **Automated Testing** - Validate detection accuracy

---

## Current Approach: Iterative Detection with Pixel Subtraction

### How It Works

The new line detection algorithm eliminates the need for constant parameter tuning:

```
┌─────────────────────────────────────────────────────────┐
│  1. Binary Threshold (127) → Strong black/white         │
│  2. ITERATION (up to 20 times):                         │
│     a) Hough Transform detects all lines                │
│     b) Pick LONGEST line                                │
│     c) Check for duplicates (angle ±8°, position ±25px) │
│     d) Draw line on mask with 8px buffer                │
│     e) DILATE mask (5×5 ellipse, +2-3px)               │
│     f) SUBTRACT from image → Line removed!              │
│  3. Repeat until no more lines or 12 lines found        │
│  4. Final filter: Remove lines < 30px                   │
└─────────────────────────────────────────────────────────┘
```

### Key Features

- **Multi-Pass Strategy**: 
  - Pass 1 (Iter 1-10): Strict threshold (15), longer lines (35px)
  - Pass 2 (Iter 11-20): Relaxed threshold (10), shorter lines (25px)
  
- **Longest-First**: Prioritizes major/important lines

- **Pixel Subtraction**: Dilate + Subtract prevents duplicate detection

- **Crossing Detection**: Lines with 80-100° angle difference are both kept (X pattern)

### Current Parameters

```python
LineDetector(
    threshold=18,           # Base threshold
    min_line_length=35,     # Initial min length
    max_line_gap=35,        # Connect segments
    final_min_length=30     # Final noise filter
)
```

**Why These Work:**
- Iterative approach is self-correcting
- Pixel subtraction eliminates most tuning needs
- Multi-pass catches both strong and weak lines
- Parameters are stable across different images

---

## Workflow: Creating & Testing

### Step 1: Define Reference (One-Time Setup)

1. Go to http://localhost/reference.html
2. Click two points to define each line
3. Lines are automatically categorized (H/V/D)
4. Delete any mistakes
5. Reference is saved in database

**Ground Truth**: Manual definition ensures 100% accuracy for reference lines.

### Step 2: Create Test Images

1. Go to http://localhost/draw_testimage.html
2. Draw your test image (256×256 canvas)
3. Use rotation buttons (±10°) to create variations
4. **Manual Scoring**:
   - **Correct Lines**: How many reference lines you drew
   - **Extra Lines**: Additional lines not in reference
   - **Missing Lines**: Auto-calculated
5. Save with descriptive name

**Purpose**: Creates labeled dataset for algorithm validation.

### Step 3: Run Automated Tests

1. Go to http://localhost/run_test.html
2. See available test image count
3. Configure settings (optional):
   - Image Registration: Enable/disable
   - Line Matching Tolerances
   - Max Rotation
4. Click **Run All Tests**

**Metrics Displayed:**
- **Test Rating** (Prediction Accuracy): Expected vs Actual detection
- **Reference Match**: How well lines match reference
- Individual results with visualizations

### Step 4: Analyze Results

**Per-Image Results:**
- Original → Registered → Reference visualization
- Expected (your scoring) vs Actual (detected)
- Difference breakdown

**Overall Stats:**
- Average Test Rating (target: >90%)
- Perfect Tests (100% match)
- Success rate

---

## When to Adjust Parameters

### ❌ Don't Adjust If:
- Test Rating is >90%
- Most test images are correctly detected
- Only 1-2 problem cases

### ✅ Consider Adjusting If:
- Test Rating consistently <80%
- Many test images have wrong line counts
- Systematic over/under-detection

### How to Adjust

Edit `api/image_processing/line_detector.py`:

```python
def __init__(
    self,
    threshold=18,           # Higher = fewer lines detected
    min_line_length=35,     # Higher = ignore shorter lines
    max_line_gap=35,        # Higher = connect more gaps
    final_min_length=30     # Higher = stricter final filter
):
```

**After Changes:**
```bash
docker compose restart api
```

Then re-run tests to verify improvement.

---

## Comparison Tolerance Optimization

### Current Tolerances

```python
LineComparator(
    position_tolerance=120.0,   # Max distance in pixels
    angle_tolerance=50.0,       # Max angle difference in degrees
    length_tolerance=0.8,       # Max length difference (ratio)
    similarity_threshold=0.5    # Min similarity for match
)
```

### Real-Time Tuning

You can adjust these in the **Run Tests UI** without code changes:
1. Expand "Advanced Settings"
2. Adjust sliders for Position, Angle, Length
3. Click "Run All Tests" to see immediate impact

### Grid Search Optimization (Advanced)

If you want to systematically optimize tolerances:

```bash
# Run grid search
docker compose exec api python3 /app/test_runner.py
```

This will:
1. Test multiple tolerance combinations
2. Find parameters with best Test Rating
3. Generate HTML report in `data/test_output/`

**Note**: This optimizes comparison tolerances, NOT line detection parameters.

---

## Understanding Metrics

### 1. Reference Match (Detection Score)

**Formula**: `correct_lines / total_reference_lines`

**Meaning**: How well detected lines match the reference image.

**Example**:
- Reference has 8 lines
- Detected 7 correct, 1 missing, 0 extra
- Reference Match = 7/8 = 87.5%

### 2. Test Rating (Prediction Accuracy)

**Formula**: Compares Expected (manual) vs Actual (detected)

**Meaning**: How well the algorithm predicts what you told it to expect.

**Example**:
- Expected: Correct=7, Missing=1, Extra=0
- Actual: Correct=7, Missing=1, Extra=0
- Test Rating = 100% (perfect match!)

**Why Important**: 
- Measures algorithm reliability
- Validates manual scoring accuracy
- Shows if detection is consistent

---

## Best Practices

### ✅ Do's

- **Create diverse test images** - Different patterns, rotations, missing lines
- **Score test images carefully** - Accurate expected values are crucial
- **Run tests after changes** - Always verify improvements
- **Use registration for rotated images** - Handles misalignment
- **Review visualizations** - Understand what's being detected
- **Keep test images clean** - Avoid ambiguous/messy drawings

### ❌ Don'ts

- **Don't over-optimize** - 90%+ Test Rating is excellent
- **Don't change parameters frequently** - Stability matters
- **Don't ignore visualizations** - They show exactly what's detected
- **Don't create too-easy tests** - Challenge the algorithm
- **Don't skip verification** - Always run full test suite after changes

---

## Troubleshooting

### Problem: Low Test Rating (<80%)

**Causes:**
1. Test image scoring is incorrect
2. Detection parameters need adjustment
3. Registration is failing
4. Test images are too different from reference

**Solutions:**
1. Review test image expected values
2. Check individual visualizations
3. Try with/without registration
4. Create more varied test images

### Problem: Inconsistent Results

**Causes:**
1. Image registration varies slightly
2. Lines near detection threshold
3. Ambiguous line patterns

**Solutions:**
1. Increase `final_min_length` to filter noise
2. Adjust comparison tolerances
3. Make test images more distinct

### Problem: Missing Diagonal Lines

**Causes:**
1. Diagonal lines fragmented by rotation
2. Threshold too high
3. Lines too thin after skeletonization

**Solutions:**
1. Enable registration (helps alignment)
2. Lower `threshold` slightly
3. Check if test image is too faint

### Problem: Too Many Extra Lines

**Causes:**
1. Noise in image
2. Threshold too low
3. Artifacts from registration

**Solutions:**
1. Increase `threshold`
2. Increase `min_line_length`
3. Increase `final_min_length` (filters short artifacts)
4. Use higher quality test images

---

## Advanced: Iterative Detection Internals

### Overlap Detection Algorithm

```python
def _overlaps_with_existing(line, existing_lines):
    # Calculate angle and position
    angle = atan2(y2-y1, x2-x1) * 180/π
    mid = ((x1+x2)/2, (y1+y2)/2)
    
    for existing_line in existing_lines:
        angle_diff = abs(angle - existing_angle)
        
        # Special: Crossing lines (X pattern)
        if 80° <= angle_diff <= 100°:
            continue  # NOT a duplicate!
        
        # Check angle and position
        if angle_diff < 8° AND distance < 25px:
            return True  # Duplicate!
    
    return False
```

### Pixel Subtraction Process

```python
def _erase_line_pixels(image, line):
    # 1. Create mask
    mask = zeros_like(image)
    cv2.line(mask, (x1,y1), (x2,y2), 255, thickness=8)
    
    # 2. DILATE (expand by 2-3px)
    kernel = cv2.getStructuringElement(MORPH_ELLIPSE, (5,5))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    
    # 3. SUBTRACT
    image[dilated > 0] = 0  # Erase completely!
```

**Why Dilate?**
- Captures nearby pixels
- Removes line completely (no artifacts)
- Prevents re-detection of same line

---

## Migration from Old System

If you're upgrading from a previous version:

### Old System (Deprecated)
- `stepwise_line_optimizer.py` - ❌ Removed
- Manual parameter tuning required
- Separate optimization runs

### New System (Current)
- Iterative detection with pixel subtraction
- Manual reference definition
- Integrated test suite with UI
- Real-time tolerance adjustment

### Migration Steps

1. **Delete old files** (if present):
   ```bash
   rm api/stepwise_line_optimizer.py
   rm api/optimize_detection.py
   ```

2. **Use new reference editor**:
   - Define lines manually at http://localhost/reference.html

3. **Create test images**:
   - Use http://localhost/draw_testimage.html

4. **Run tests**:
   - Use http://localhost/run_test.html

No parameter optimization scripts needed!

---

## Related Documentation

- [Main README](../README.md) - Complete project overview
- [API Endpoints](../README.md#-api-endpoints) - REST API documentation
- [Algorithm Details](../README.md#-algorithms-used) - Technical implementation

---

## Changelog

### 2025-10-13 - Major Refactor
- ✅ Implemented iterative detection with pixel subtraction
- ✅ Manual reference line definition
- ✅ Integrated test suite with UI
- ✅ Real-time tolerance adjustment
- ❌ Removed stepwise optimizer (no longer needed)
- ❌ Removed separate optimization scripts

### 2025-10-12 - Initial Release (Deprecated)
- Created Stepwise Line Detection Optimizer
- Implemented iterative parameter adjustment

---

## Support

For questions or issues:
1. Check test visualizations at http://localhost/run_test.html
2. Review individual test results
3. Examine API logs: `docker compose logs api`
4. Check this documentation

---

**Last Updated:** 2025-10-13  
**Status:** ✅ Current (Iterative Detection Method)
