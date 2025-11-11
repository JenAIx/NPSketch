#!/usr/bin/env python3
"""
MAT File Extractor - Extract reference images and drawn lines from MATLAB .mat files.

Usage:
    python3 mat_extractor.py --input <input_dir> --output <output_dir>
    
Example:
    python3 mat_extractor.py --input /app/templates/bsp_ocsplus_202511 --output /app/data/tmp
"""

import scipy.io
import numpy as np
from PIL import Image, ImageDraw
import os
import json
import argparse
from datetime import datetime
import cv2


# Configuration file name
CONFIG_FILE = "mat_extractor.conf"


def load_config(config_path):
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"Loaded config from: {config_path}")
                print(f"  Resolution: {config.get('canvas_width')}x{config.get('canvas_height')}")
                return config
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    return None


def create_default_config(config_path):
    """Create default configuration file if it doesn't exist (manual step)."""
    default_config = {
        'canvas_width': 568,
        'canvas_height': 274,
        'created_at': datetime.now().isoformat(),
        'source': 'Optimized for CNN training - landscape format with auto-cropping',
        'padding_px': 5,
        'auto_crop': True
    }
    
    if not os.path.exists(config_path):
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created default config: {config_path}")
            print(f"  Default resolution: {default_config['canvas_width']}x{default_config['canvas_height']}")
            print(f"  → Edit this file to set your desired resolution")
            return True
        except Exception as e:
            print(f"Error: Could not create default config: {e}")
            return False
    return False


def load_mat_file(mat_file_path):
    """Load a MATLAB .mat file."""
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        return mat_data
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return None


def extract_reference_image(mat_data, key_prefix):
    """Extract the reference image (figs field)."""
    task_type = 'COPY' if 'memory' not in key_prefix else 'RECALL'
    
    data = mat_data[key_prefix][0, 0]
    figs = data['figs']
    
    # Handle potential wrapping
    if figs.shape == (1, 1):
        figs = figs[0, 0]
    
    return figs, task_type


def extract_drawn_lines(mat_data, key_prefix):
    """Extract the actual drawn lines from trails.cont_lines field."""
    task_type = 'COPY' if 'memory' not in key_prefix else 'RECALL'
    
    data = mat_data[key_prefix][0, 0]
    
    # Extract drawing area rectangle
    draw_area = data['draw_area'][0, 0]
    rect_data = draw_area['rect']
    
    if isinstance(rect_data, np.ndarray) and rect_data.shape == (1, 4):
        rect = rect_data[0]
    else:
        rect = np.array([0, 0, 568, 568])
    
    # Extract trails - cont_lines is an array of separate line objects
    trails = data['trails'][0, 0]
    cont_lines_array = trails['cont_lines']
    
    # Extract all lines
    lines = []
    num_lines = cont_lines_array.shape[1]
    total_points = 0
    
    for i in range(num_lines):
        line = cont_lines_array[0, i]
        if isinstance(line, np.ndarray) and line.size > 0:
            lines.append(line)
            total_points += line.shape[0]
    
    return lines, rect, task_type, total_points


def normalize_line_thickness(image_array, target_thickness=2, threshold=127):
    """
    Normalize line thickness to a consistent width using skeletonization and dilation.
    
    Args:
        image_array: RGB or grayscale numpy array
        target_thickness: Desired line thickness in pixels (default: 2)
        threshold: Threshold for binarization (default: 127)
    
    Returns:
        Normalized RGB numpy array (uint8)
    """
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()
    
    # Binarize: white background (255), black lines (0)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Invert for processing: black background, white lines
    binary_inv = cv2.bitwise_not(binary)
    
    # Check if there's any content
    if np.sum(binary_inv) == 0:
        # Empty image, return as-is
        result = np.ones_like(gray) * 255
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    # Skeletonize to 1-pixel thin lines using Zhang-Suen thinning
    skeleton = cv2.ximgproc.thinning(binary_inv, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    
    # If target thickness is 1, we're done
    if target_thickness <= 1:
        skeleton_inv = cv2.bitwise_not(skeleton)
        return cv2.cvtColor(skeleton_inv, cv2.COLOR_GRAY2RGB)
    
    # Dilate to target thickness
    kernel_size = max(1, target_thickness - 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(skeleton, kernel, iterations=1)
    
    # Invert back: white background, black lines
    result = cv2.bitwise_not(dilated)
    
    # Convert to RGB
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    return result_rgb


def calculate_bounding_box_from_lines(lines, padding=5):
    """
    Calculate bounding box from line coordinates.
    
    Args:
        lines: List of numpy arrays with shape (n_points, 3) containing x, y, timestamp
        padding: Padding in pixels to add around content
    
    Returns:
        Tuple (min_x, min_y, max_x, max_y) or None if no valid lines
    """
    if not lines or len(lines) == 0:
        return None
    
    all_x = []
    all_y = []
    
    for line in lines:
        if line.shape[0] > 0:
            all_x.extend(line[:, 0])
            all_y.extend(line[:, 1])
    
    if not all_x or not all_y:
        return None
    
    min_x = min(all_x) - padding
    max_x = max(all_x) + padding
    min_y = min(all_y) - padding
    max_y = max(all_y) + padding
    
    return (min_x, min_y, max_x, max_y)


def calculate_bounding_box_from_image(image_data, padding=5, threshold=250):
    """
    Calculate bounding box from image by finding non-white pixels.
    
    Args:
        image_data: RGB numpy array
        padding: Padding in pixels to add around content
        threshold: Pixel value threshold (pixels < threshold are considered content)
    
    Returns:
        Tuple (min_x, min_y, max_x, max_y) or None if no content found
    """
    # Convert to grayscale if needed
    if len(image_data.shape) == 3:
        gray = np.mean(image_data, axis=2)
    else:
        gray = image_data
    
    # Find non-white pixels
    content_mask = gray < threshold
    
    # Find bounding box
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    
    # Add padding
    min_x = max(0, min_x - padding)
    max_x = min(image_data.shape[1] - 1, max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(image_data.shape[0] - 1, max_y + padding)
    
    return (min_x, min_y, max_x, max_y)


def save_reference_image(image_data, output_path, target_size=None, auto_crop=True, padding=5):
    """Save reference image as PNG, with optional auto-cropping and resizing."""
    try:
        if image_data.dtype != np.uint8:
            if image_data.max() <= 1.0:
                image_data = (image_data * 255).astype(np.uint8)
            else:
                image_data = image_data.astype(np.uint8)
        
        # Auto-crop to content if requested
        if auto_crop:
            bbox = calculate_bounding_box_from_image(image_data, padding=padding)
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                image_data = image_data[int(min_y):int(max_y)+1, int(min_x):int(max_x)+1]
        
        if len(image_data.shape) == 3:
            img = Image.fromarray(image_data, mode='RGB')
        else:
            img = Image.fromarray(image_data, mode='L')
        
        # Resize if target size is specified
        if target_size is not None and target_size != (img.width, img.height):
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Normalize line thickness for consistency with drawn images
        if auto_crop:  # Use auto_crop as proxy for normalization
            img_array = np.array(img)
            normalized = normalize_line_thickness(img_array, target_thickness=2)
            img = Image.fromarray(normalized, mode='RGB')
        
        img.save(output_path)
        
        # Verify file was created
        if not os.path.exists(output_path):
            raise Exception(f"File was not created: {output_path}")
        
        return img.size  # Return actual size
    except Exception as e:
        print(f"  ERROR saving reference: {e}")
        import traceback
        traceback.print_exc()
        return None


def render_drawn_lines(lines, rect, output_path, canvas_size=(568, 274),
                       auto_crop=True, padding=5,
                       bg_color=(255, 255, 255), line_color=(0, 0, 0), line_width=2):
    """Render the drawn lines as a PNG image with optional auto-cropping."""
    try:
        if len(lines) == 0:
            # Create empty image
            img = Image.new('RGB', canvas_size, bg_color)
            img.save(output_path)
            return 0
        
        # Calculate bounding box from line coordinates
        if auto_crop:
            bbox = calculate_bounding_box_from_lines(lines, padding=padding)
            if not bbox:
                # Fallback to full canvas
                bbox = (0, 0, canvas_size[0], canvas_size[1])
        else:
            # Use original draw_area rect
            x1, y1, x2, y2 = rect
            bbox = (x1, y1, x2, y2)
        
        min_x, min_y, max_x, max_y = bbox
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        # Create temporary canvas at original size for crisp rendering
        temp_canvas_width = int(bbox_width) + 1
        temp_canvas_height = int(bbox_height) + 1
        
        img = Image.new('RGB', (temp_canvas_width, temp_canvas_height), bg_color)
        draw = ImageDraw.Draw(img)
        
        lines_drawn = 0
        
        # Draw each line
        for line in lines:
            if line.shape[0] < 2:
                continue
            
            # Convert line points to bbox-relative coordinates
            canvas_points = []
            for i in range(line.shape[0]):
                x = line[i, 0] - min_x
                y = line[i, 1] - min_y
                canvas_points.append((x, y))
            
            # Draw this line
            if len(canvas_points) >= 2:
                draw.line(canvas_points, fill=line_color, width=line_width)
                lines_drawn += 1
        
        # Resize to target canvas size
        if canvas_size != (temp_canvas_width, temp_canvas_height):
            img = img.resize(canvas_size, Image.Resampling.LANCZOS)
        
        # Normalize line thickness if requested
        if auto_crop:  # Use auto_crop as proxy for normalization
            img_array = np.array(img)
            normalized = normalize_line_thickness(img_array, target_thickness=2)
            img = Image.fromarray(normalized, mode='RGB')
        
        img.save(output_path)
        
        # Verify file was created
        if not os.path.exists(output_path):
            raise Exception(f"File was not created: {output_path}")
        
        return lines_drawn
    except Exception as e:
        print(f"  ERROR rendering lines: {e}")
        import traceback
        traceback.print_exc()
        return 0


def process_mat_file(mat_file_path, output_directory, canvas_size, auto_crop=True, padding=5):
    """
    Main function to extract both reference and drawn images from MAT file.
    
    Args:
        mat_file_path: Path to the input .mat file
        output_directory: Directory where to save PNG files
        canvas_size: Tuple (width, height) for output images (from config)
        auto_crop: Whether to auto-crop to content
        padding: Padding in pixels around content
    
    Returns:
        bool: Success status
    """
    filename = os.path.basename(mat_file_path)
    print(f"\nProcessing: {filename}")
    print("-" * 80)
    
    # Load MAT file
    mat_data = load_mat_file(mat_file_path)
    if mat_data is None:
        return False
    
    # Extract patient ID from filename
    parts = filename.split('_')
    pc_id = None
    pro_id = None
    for part in parts:
        if part.startswith('PC'):
            pc_id = part
        elif part.startswith('Pro'):
            pro_id = part
    
    patient_id = pc_id if pc_id else (pro_id if pro_id else "UNKNOWN")
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    date_str = datetime.now().strftime('%Y%m%d')
    
    # Extract reference image once (same for COPY and RECALL)
    reference_saved = False
    
    # Process both COPY and RECALL tasks
    for key_prefix in ['data_complex_copy', 'data_complex_memory_copy']:
        if key_prefix not in mat_data:
            continue
        
        task_type = 'COPY' if 'memory' not in key_prefix else 'RECALL'
        
        # Extract and save reference image (only once)
        if not reference_saved:
            try:
                ref_image, _ = extract_reference_image(mat_data, key_prefix)
                
                ref_filename = f"{patient_id}_REFERENCE_{date_str}.png"
                ref_path = os.path.join(output_directory, ref_filename)
                result_size = save_reference_image(ref_image, ref_path, canvas_size, auto_crop, padding)
                
                if result_size:
                    print(f"  ✓ Saved reference: {ref_filename} ({result_size[0]}x{result_size[1]})")
                    reference_saved = True
            except Exception as e:
                print(f"  ✗ Error extracting reference: {e}")
        
        # Extract and save drawn lines
        try:
            drawn_lines, rect, _, total_points = extract_drawn_lines(mat_data, key_prefix)
            drawn_filename = f"{patient_id}_{task_type}_drawn_{date_str}.png"
            drawn_path = os.path.join(output_directory, drawn_filename)
            lines_drawn = render_drawn_lines(drawn_lines, rect, drawn_path, canvas_size, auto_crop, padding)
            
            print(f"  ✓ Saved {task_type:6s}: {drawn_filename} ({lines_drawn} lines, {total_points} points)")
        except Exception as e:
            print(f"  ✗ Error extracting {task_type}: {e}")
    
    return True


def find_mat_files(base_directory):
    """Find all .mat files in the directory tree."""
    import glob
    pattern = os.path.join(base_directory, "**", "*.mat")
    mat_files = glob.glob(pattern, recursive=True)
    return sorted(mat_files)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract reference images and drawn lines from MATLAB .mat files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  python3 mat_extractor.py -i /app/templates/folder -o /app/data/output
  
  # With custom config location
  python3 mat_extractor.py -i /app/templates/folder -o /app/data/output -c /app/mat_extraction/mat_extractor.conf
        '''
    )
    parser.add_argument('--input', '-i', required=True, help='Input directory containing .mat files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for PNG files')
    parser.add_argument('--config', '-c', default=None, help='Config file path (default: script_dir/mat_extractor.conf)')
    
    args = parser.parse_args()
    
    input_directory = args.input
    output_directory = args.output
    
    # Config file path - default to script directory
    if args.config:
        config_path = args.config
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, CONFIG_FILE)
    
    print("=" * 80)
    print("MAT File Extractor - Batch Processing")
    print("=" * 80)
    print(f"Input directory:  {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Config file:      {config_path}")
    print("=" * 80)
    
    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        print(f"\n⚠ Config file not found: {config_path}")
        create_default_config(config_path)
        print(f"\n→ Please review and adjust the config file, then run again.")
        return
    
    # Load configuration (required)
    config = load_config(config_path)
    if not config:
        print(f"\n✗ ERROR: Could not load config from: {config_path}")
        print("→ Please ensure the config file exists and is valid JSON")
        return
    
    canvas_size = (config['canvas_width'], config['canvas_height'])
    auto_crop = config.get('auto_crop', True)
    padding = config.get('padding_px', 5)
    
    print(f"Auto-crop: {'enabled' if auto_crop else 'disabled'}")
    print(f"Padding: {padding}px")
    
    # Find all .mat files
    print(f"\nSearching for .mat files in: {input_directory}")
    mat_files = find_mat_files(input_directory)
    
    print(f"Found {len(mat_files)} .mat file(s)\n")
    
    if len(mat_files) == 0:
        print("No .mat files found!")
        return
    
    # Process each file
    successful = 0
    failed = 0
    
    for mat_file in mat_files:
        try:
            success = process_mat_file(mat_file, output_directory, canvas_size, auto_crop, padding)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ ERROR processing {os.path.basename(mat_file)}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successfully processed: {successful}")
    print(f"Failed:                 {failed}")
    print(f"Output directory:       {output_directory}")
    print(f"Canvas resolution:      {canvas_size[0]}x{canvas_size[1]} (from config)")
    print("=" * 80)


if __name__ == "__main__":
    main()
