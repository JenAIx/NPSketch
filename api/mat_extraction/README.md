# MAT File Extractor

Extract reference images and hand-drawn lines from MATLAB .mat files with automatic content cropping.

## Features

- **Auto-Cropping**: Automatically crops to actual drawing content
- **Smart Padding**: Adds configurable white border (default: 5px)
- **Line Normalization**: Zhang-Suen thinning + dilation for consistent 2.00px thickness
- **Fixed Resolution**: All outputs scaled to consistent dimensions
- **Batch Processing**: Processes entire directory trees

## Configuration

### mat_extractor.conf

The tool requires a configuration file that defines output parameters.

**Important:** The configuration is **read-only**. It will never be modified during extraction.

### Default Configuration (568×274)

```json
{
  "canvas_width": 568,
  "canvas_height": 274,
  "auto_crop": true,
  "padding_px": 5
}
```

### First Run

On first run (when no config exists):
1. Tool creates default configuration (568×274, landscape, auto-crop enabled)
2. Tool stops and prompts you to review
3. Edit config if needed (e.g., change resolution, disable auto-crop)
4. Run tool again

### Configuration Options

- `canvas_width`: Output image width in pixels
- `canvas_height`: Output image height in pixels
- `auto_crop`: Enable/disable automatic content cropping (true/false)
- `padding_px`: White border padding in pixels (default: 5)

**Benefits:**
- Consistent resolution across all extractions
- Optimized for CNN training (minimal white space)
- No accidental changes
- Version control friendly
- Reproducible results

## Usage

```bash
docker exec npsketch-api python3 /app/mat_extraction/mat_extractor.py \\
  --input /app/templates/bsp_ocsplus_202511 \\
  --output /app/data/tmp
```

## Output

The tool generates three types of PNG files per patient:

1. **Reference Image**: `PC56_REFERENCE_20251111.png`
   - Original stimulus figure
   - Auto-cropped to content + padding
   - Scaled to configured resolution (e.g., 568×274)

2. **COPY Drawing**: `PC56_COPY_drawn_20251111.png`
   - Patient's immediate copy attempt
   - Rendered from coordinate data
   - Auto-cropped to actual drawing bounds + padding
   - Scaled to configured resolution

3. **RECALL Drawing**: `PC56_RECALL_drawn_20251111.png`
   - Patient's memory recall drawing
   - Same processing as COPY

### Auto-Cropping Behavior

- **Drawn Images**: Bounding box calculated from actual line coordinates
- **Reference Images**: Bounding box calculated from non-white pixels
- Configurable padding added around content (default: 5px)
- Final image scaled to target resolution while preserving aspect ratio
- Result: Optimal content density for CNN training

### Example Output

```
PC0460_REFERENCE_20251111.png   (568×274, 6px margins)
PC0460_COPY_drawn_20251111.png  (568×274, 5-6px margins)
PC0460_RECALL_drawn_20251111.png (568×274, 5-6px margins)
```

All images use the resolution and parameters from `mat_extractor.conf`.
