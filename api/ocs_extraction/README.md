# OCS Image Extractor

Extract red-pixel drawings from OCS (Observer-rated Clinical Scale) rating images.

## Overview

This tool processes OCS rating images that contain multiple elements (reference figures, grids, annotations) and extracts **only the red-pixel hand-drawn lines**, outputting clean black-on-white PNG images optimized for CNN training.

## Features

- **Red Pixel Detection**: Isolates red pixels using configurable RGB thresholds
- **Auto-Cropping**: Automatically crops to actual drawing bounds with padding
- **Line Normalization**: Zhang-Suen thinning + dilation for consistent 2.00px thickness
- **Noise Removal**: Removes all non-red elements (reference figures, grids, text)
- **Fixed Resolution**: All outputs scaled to consistent dimensions (default: 568×274)
- **Batch Processing**: Processes entire directory trees
- **Configurable**: Adjustable thresholds and output parameters

## Configuration

### ocs_extractor.conf

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

### Configuration Options

- `canvas_width` / `canvas_height`: Target output resolution
- `auto_crop`: Enable automatic content cropping (default: true)
- `padding_px`: White border padding around content (default: 5px)
- `red_threshold`: RGB thresholds for red pixel detection
  - `r_min`: Minimum red channel value (default: 200)
  - `g_max`: Maximum green channel value (default: 100)
  - `b_max`: Maximum blue channel value (default: 100)

## Usage

### Basic Usage

```bash
docker exec npsketch-api python3 /app/ocs_extraction/ocs_extractor.py \
  --input /app/templates/bsp_ocsplus_202511/Human_rater/imgs \
  --output /app/data/tmp
```

### With Custom Config

```bash
docker exec npsketch-api python3 /app/ocs_extraction/ocs_extractor.py \
  --input /path/to/images \
  --output /path/to/output \
  --config /path/to/custom.conf
```

## Input Files

The tool expects PNG images with specific naming conventions:

- `{PatientID}_COPY.png` - Immediate copy task
- `{PatientID}_RECALL.png` - Memory recall task

**Examples:**
- `Park_16_COPY.png`
- `TeamD178_RECALL.png`
- `TEAMK299_COPY.png`

## Output Files

Generated PNG files follow this naming pattern:

```
{PatientID}_{TaskType}_ocs_{date}.png
```

**Examples:**
- `Park_16_COPY_ocs_20251111.png`
- `TeamD178_RECALL_ocs_20251111.png`
- `TEAMK299_COPY_ocs_20251111.png`

### Output Characteristics

- **Resolution**: 568×274 pixels (consistent across all outputs)
- **Format**: Black lines on white background (RGB)
- **Margins**: ~3-7px white border (auto-cropped to content)
- **Content**: Only red pixels from original, converted to black

## Processing Pipeline

1. **Load Image**: Read OCS rating image (typically 520×420 RGBA)
2. **Red Pixel Detection**: Apply RGB threshold to isolate red pixels
3. **Bounding Box Calculation**: Find minimal bounds around red pixels
4. **Add Padding**: Extend bounds by configurable padding
5. **Crop**: Extract only the bounded region
6. **Color Conversion**: Red pixels → Black pixels, background → White
7. **Resize**: Scale to target resolution (568×274)
8. **Line Normalization**: Skeletonize using Zhang-Suen thinning, then dilate to 2.00px thickness
9. **Save**: Output as PNG

## Technical Details

### Red Pixel Detection Algorithm

Red pixels are identified using RGB thresholds:
- Red channel (R) ≥ 200
- Green channel (G) ≤ 100
- Blue channel (B) ≤ 100

This ensures reliable detection of drawing strokes while ignoring other elements.

### Auto-Cropping

- Bounding box calculated from actual red pixel locations
- Configurable padding (default: 5px) added around content
- Ensures optimal content density (typically 4-8% black pixels)
- Maintains consistent margins across all outputs

### Quality Metrics

Typical output characteristics:
- **Black pixel coverage**: 2-5% of total image (after normalization)
- **Line thickness**: 2.00px ± 0.00px (perfectly normalized)
- **Margins**: 3-7px on all sides
- **Edge brightness**: 255/255 (pure white borders)

## Example Processing Results

```
Park_16_COPY.png (520x420) → Park_16_COPY_ocs_20251111.png (568x274)
  - Red pixels found: 6,251
  - Cropped from: 302x163
  - Final margins: 5-7px

TEAMK276_COPY.png (520x420) → TEAMK276_COPY_ocs_20251111.png (568x274)
  - Red pixels found: 9,777
  - Cropped from: 413x237
  - Final margins: 3-4px
```

## Troubleshooting

### No red pixels found

If the tool reports "No red pixels found":
1. Check if the image actually contains red drawing strokes
2. Adjust `red_threshold` values in config (may need lower `r_min` or higher `g_max`/`b_max`)

### Wrong elements extracted

If non-drawing elements are included:
1. Tighten `red_threshold` (increase `r_min`, decrease `g_max`/`b_max`)
2. Verify that drawings are actually red (not orange, pink, etc.)

### Output too small/large

Adjust `canvas_width` and `canvas_height` in config to change output dimensions.

## Comparison to MAT Extractor

| Feature | MAT Extractor | OCS Extractor |
|---------|---------------|---------------|
| Input Format | MATLAB .mat files | PNG images |
| Data Source | Coordinate arrays | Pixel data |
| Extraction Method | Line rendering | Color filtering |
| Target Element | All drawing data | Red pixels only |
| Output Format | 568×274 PNG | 568×274 PNG |
| Line Thickness | 2.00px (normalized) | 2.00px (normalized) |
| Use Case | Machine recordings | Human ratings |

Both tools produce identical output characteristics (resolution, line thickness, margins) for seamless integration in CNN training pipelines.

