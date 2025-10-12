# Line Detection Optimization Guide

## Overview

NPSketch provides two complementary optimization approaches for achieving perfect line detection:

### 1. **Grid Search Optimization** (`test_runner.py`)
Tests multiple parameter combinations for comparison metrics (position, angle, length tolerances).

### 2. **Stepwise Line Detection** (`stepwise_line_optimizer.py`) ⭐
Optimizes Hough Transform parameters to match expected line counts in test images.

---

## Stepwise Line Detection Optimizer

### Concept

The optimizer adjusts line detection parameters until the detected line count matches the expected total from your test images.

**Formula:** `Target Lines = Expected Correct + Expected Extra`

**Examples:**
- If `Expected: Correct=8, Extra=0` → Optimize until **8 lines** are detected
- If `Expected: Correct=6, Extra=2` → Optimize until **8 lines** are detected
- If `Expected: Correct=5, Extra=0, Missing=3` → Optimize until **5 lines** are detected

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│  1. Load test image + expected values                   │
│  2. Calculate target: correct + extra                   │
│  3. Detect lines with current parameters                │
│                                                          │
│  IF detected > target:                                  │
│     → Increase threshold (stricter detection)           │
│                                                          │
│  IF detected < target:                                  │
│     → Decrease threshold (more sensitive)               │
│     → Increase max_line_gap (connect broken lines)      │
│                                                          │
│  4. Repeat until detected == target                     │
└─────────────────────────────────────────────────────────┘
```

### Parameters Optimized

| Parameter | Description | Effect |
|-----------|-------------|--------|
| `threshold` | Minimum votes required | **Primary control** - Higher = fewer lines |
| `max_line_gap` | Max gap to connect segments | Connects broken lines |
| `min_line_length` | Minimum line length | Filters short artifacts |

---

## Usage

### Basic Usage

```bash
# Run the optimizer
docker compose exec api python3 /app/stepwise_line_optimizer.py
```

This will:
1. Load all test images from the database
2. Optimize parameters for each image individually
3. Calculate recommended global parameters
4. Generate an HTML report in `/app/data/test_output/`

### Programmatic Usage

```python
from stepwise_line_optimizer import StepwiseLineOptimizer

# Initialize
optimizer = StepwiseLineOptimizer(use_registration=True)

# Optimize all test images
results = optimizer.optimize_all_test_images(
    max_iterations=20,  # Max optimization iterations per image
    tolerance=0         # 0 = exact match required
)

# Get recommended parameters
recommended = results['recommended_parameters']
print(f"Threshold: {recommended['threshold']}")
print(f"Min Length: {recommended['min_line_length']}")
print(f"Max Gap: {recommended['max_line_gap']}")

# Generate HTML report
optimizer.generate_report(results)
```

### Optimize Single Image

```python
# Get test image and reference from database
test_image = db.query(TestImage).filter(TestImage.test_name == "MyTest").first()
reference = db.query(ReferenceImage).first()

# Optimize
result = optimizer.optimize_for_image(
    test_image=test_image,
    reference=reference,
    max_iterations=20,
    tolerance=0
)

# Check result
if result['success']:
    print(f"✅ Optimized in {result['iterations']} iterations")
    print(f"Parameters: {result['optimal_parameters']}")
else:
    print(f"⚠️ Best difference: {result['final_diff']}")
```

---

## Workflow: Adding New Test Images

### Step 1: Create Test Image

Use the webapp to draw and score a test image:
1. Go to http://localhost/draw_testimage.html
2. Draw your test image
3. Set expected values:
   - **Correct Lines**: Lines that match the reference
   - **Extra Lines**: Additional lines you drew
   - **Missing Lines**: Auto-calculated from reference
4. Save the test image

### Step 2: Run Optimization

```bash
docker compose exec api python3 /app/stepwise_line_optimizer.py
```

The optimizer will:
- Analyze your new test image
- Find parameters that detect exactly `Correct + Extra` lines
- Update recommendations if needed

### Step 3: Apply Recommended Parameters

If the optimizer suggests new parameters, update `/app/image_processing/line_detector.py`:

```python
def __init__(
    self,
    rho: float = 1.0,
    theta: float = np.pi / 180,
    threshold: int = 75,        # ← Update this
    min_line_length: int = 65,  # ← Update this
    max_line_gap: int = 50      # ← Update this
):
```

### Step 4: Verify

Run tests to ensure 100% accuracy:
```bash
curl -s -X POST "http://localhost/api/test-images/run-tests" | python3 -m json.tool
```

---

## Understanding Results

### Success Criteria

✅ **Success** = Detected lines == Target lines (within tolerance)  
⚠️ **Partial** = Close but not exact match after max iterations

### HTML Report Sections

1. **Summary Cards**
   - Total test images
   - Successful optimizations
   - Success rate

2. **Recommended Parameters**
   - Global parameters that work best across all images
   - Average of successful optimizations

3. **Individual Results Table**
   - Per-image optimization results
   - Final parameters for each image
   - Iteration count and success status

### Example Output

```
============================================================
📊 OPTIMIZATION RESULTS
============================================================
✅ Test Drawing2       :  6/ 6 lines (diff: +0, iterations:  1)
✅ Test Drawing_rotright:  6/ 6 lines (diff: +0, iterations:  1)
✅ Supper              :  8/ 8 lines (diff: +0, iterations:  1)

✨ Success rate: 3/3 images

💡 RECOMMENDED GLOBAL PARAMETERS:
   threshold:       75
   min_line_length: 65
   max_line_gap:    50
```

---

## Advanced Topics

### Custom Optimization Strategy

You can customize the optimization logic in `optimize_for_image()`:

```python
# Current strategy:
if detected_lines > target_lines:
    # Too many - be stricter
    current_params['threshold'] += max(1, (detected_lines - target_lines) * 2)
else:
    # Too few - be more sensitive
    current_params['threshold'] -= max(1, (target_lines - detected_lines) * 2)
    current_params['max_line_gap'] += 2
```

You might want to:
- Adjust step sizes for faster/slower convergence
- Add constraints (e.g., threshold must be between 20-150)
- Optimize multiple parameters simultaneously
- Use different strategies for different types of images

### Integration with Grid Search

Combine both optimization approaches:

1. **First**: Run Stepwise Optimizer for line detection
2. **Then**: Run Grid Search for comparison tolerances
3. **Result**: Optimal parameters for entire pipeline

```bash
# Step 1: Optimize line detection
docker compose exec api python3 /app/stepwise_line_optimizer.py

# Step 2: Optimize comparison
docker compose exec api python3 /app/test_runner.py
```

---

## Troubleshooting

### Optimizer Can't Reach Target

**Problem:** Max iterations reached without exact match

**Possible Causes:**
1. **Expected values are incorrect**
   - Review your test image scoring
   - Check if `correct + extra` makes sense

2. **Image quality issues**
   - Faint/broken lines may not be detectable
   - Add preprocessing or adjust image quality

3. **Registration artifacts**
   - Try with `use_registration=False`
   - Adjust `max_rotation_degrees`

4. **Parameter range limits**
   - Threshold is constrained to 20-150
   - May need to adjust constraints in code

**Solution:** Check the HTML report to see how close the optimizer got, then manually adjust.

### Different Results Each Run

**Problem:** Optimizer gives different results on repeated runs

**Causes:**
- Image registration may vary slightly
- Random initialization in some CV algorithms

**Solution:** Run multiple times and use average/median of results.

### All Images Need Different Parameters

**Problem:** No single parameter set works for all images

**This is normal!** It means your test images vary significantly. Options:

1. **Accept the average** - Best overall compromise
2. **Use per-image parameters** - Store optimal params per test
3. **Refine test images** - Make them more consistent
4. **Increase tolerance** - Allow ±1 line difference

---

## Best Practices

### ✅ Do's

- **Start with current optimal parameters** (75, 65, 50)
- **Use diverse test images** - Different line counts, patterns
- **Review generated HTML reports** - Understand what changed
- **Run after adding new test images** - Keep parameters optimal
- **Document your expected values** - Why did you set them?

### ❌ Don'ts

- **Don't optimize for one image only** - May not generalize
- **Don't ignore partial successes** - Check what's close enough
- **Don't change parameters too frequently** - Stability is important
- **Don't forget to restart API** - After manual parameter changes
- **Don't skip verification tests** - Always run full test suite

---

## Related Documentation

- [Main README](../README.md) - Project overview
- [API Documentation](../api/README.md) - API endpoints
- [Test Runner Guide](./TEST_RUNNER.md) - Grid search optimization

---

## Changelog

### 2025-10-12 - Initial Release
- Created Stepwise Line Detection Optimizer
- Implemented iterative parameter adjustment
- Added HTML report generation
- Integrated with existing test infrastructure

---

## Support

For questions or issues:
1. Check the HTML report for detailed diagnostics
2. Review this documentation
3. Examine test image expected values
4. Check API logs: `docker compose logs api`

---

**Last Updated:** 2025-10-12  
**Status:** ✅ Production Ready

