#!/usr/bin/env python3
"""
OCS Image Extractor - Extract red-pixel drawings from OCS rating images.

Usage:
    python3 ocs_extractor.py --input <input_dir> --output <output_dir>
    
Example:
    python3 ocs_extractor.py --input /app/templates/bsp_ocsplus_202511/Human_rater/imgs --output /app/data/tmp
"""

import numpy as np
from PIL import Image, ImageDraw
import os
import json
import argparse
from datetime import datetime
import cv2


# Configuration file name
CONFIG_FILE = "ocs_extractor.conf"


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
        'auto_crop': True,
        'red_threshold': {
            'r_min': 200,
            'g_max': 100,
            'b_max': 100
        }
    }
    
    if not os.path.exists(config_path):
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"\n✓ Created default config: {config_path}")
            return True
        except Exception as e:
            print(f"\n✗ Error creating config: {e}")
            return False
    return False


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


def extract_red_pixels(image_path, red_threshold):
    """
    Extract only red pixels from an OCS image.
    
    Args:
        image_path: Path to the input PNG image
        red_threshold: Dict with 'r_min', 'g_max', 'b_max'
    
    Returns:
        Tuple of (red_mask, original_shape)
    """
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        arr = np.array(img)
        
        # Extract red pixels
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        
        # Red mask: high R, low G, low B
        red_mask = (r >= red_threshold['r_min']) & \
                   (g <= red_threshold['g_max']) & \
                   (b <= red_threshold['b_max'])
        
        return red_mask, arr.shape


def calculate_red_bbox(red_mask, padding=5):
    """
    Calculate bounding box around red pixels.
    
    Args:
        red_mask: Boolean array of red pixel locations
        padding: Padding in pixels to add around content
    
    Returns:
        Tuple (min_x, min_y, max_x, max_y) or None if no red pixels
    """
    if not np.any(red_mask):
        return None
    
    rows = np.any(red_mask, axis=1)
    cols = np.any(red_mask, axis=0)
    
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    
    # Add padding
    height, width = red_mask.shape
    min_x = max(0, min_x - padding)
    max_x = min(width - 1, max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(height - 1, max_y + padding)
    
    return (min_x, min_y, max_x, max_y)


def render_red_pixels_to_image(red_mask, bbox, output_path, canvas_size=(568, 274)):
    """
    Render red pixels as black lines on white background.
    
    Args:
        red_mask: Boolean array of red pixel locations
        bbox: Bounding box (min_x, min_y, max_x, max_y)
        output_path: Path to save PNG
        canvas_size: Target canvas size (width, height)
    
    Returns:
        True if successful
    """
    try:
        if bbox is None:
            # Create empty white image
            img = Image.new('RGB', canvas_size, (255, 255, 255))
            img.save(output_path)
            return True
        
        min_x, min_y, max_x, max_y = bbox
        
        # Crop to bounding box
        cropped_mask = red_mask[min_y:max_y+1, min_x:max_x+1]
        
        # Create image from cropped mask (red pixels -> black, others -> white)
        height, width = cropped_mask.shape
        img_array = np.ones((height, width, 3), dtype=np.uint8) * 255
        img_array[cropped_mask] = [0, 0, 0]  # Red pixels become black
        
        # Create PIL image and resize
        img = Image.fromarray(img_array, mode='RGB')
        img = img.resize(canvas_size, Image.Resampling.LANCZOS)
        
        # Normalize line thickness for consistency
        img_array_resized = np.array(img)
        normalized = normalize_line_thickness(img_array_resized, target_thickness=2)
        img = Image.fromarray(normalized, mode='RGB')
        
        img.save(output_path)
        
        # Verify file was created
        if not os.path.exists(output_path):
            raise Exception(f"File was not created: {output_path}")
        
        return True
    except Exception as e:
        print(f"  ERROR rendering image: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_ocs_image(image_path, output_directory, config):
    """
    Process a single OCS image.
    
    Args:
        image_path: Path to input PNG
        output_directory: Directory to save output
        config: Configuration dict
    
    Returns:
        bool: Success status
    """
    filename = os.path.basename(image_path)
    print(f"\nProcessing: {filename}")
    print("-" * 80)
    
    # Parse filename: {ID}_COPY.png or {ID}_RECALL.png
    name_parts = filename.replace('.png', '').split('_')
    
    if len(name_parts) < 2:
        print(f"  ✗ Invalid filename format: {filename}")
        return False
    
    task_type = name_parts[-1]  # COPY or RECALL
    patient_id = '_'.join(name_parts[:-1])  # Everything before task type
    
    if task_type not in ['COPY', 'RECALL']:
        print(f"  ✗ Unknown task type: {task_type}")
        return False
    
    # Extract red pixels
    try:
        red_threshold = config.get('red_threshold', {'r_min': 200, 'g_max': 100, 'b_max': 100})
        red_mask, original_shape = extract_red_pixels(image_path, red_threshold)
        
        red_pixel_count = np.sum(red_mask)
        print(f"  Found {red_pixel_count:,} red pixels in {original_shape[1]}x{original_shape[0]} image")
        
        if red_pixel_count == 0:
            print(f"  ⚠ No red pixels found!")
        
        # Calculate bounding box
        padding = config.get('padding_px', 5)
        bbox = calculate_red_bbox(red_mask, padding)
        
        if bbox:
            min_x, min_y, max_x, max_y = bbox
            print(f"  Bounding box: x={min_x}..{max_x}, y={min_y}..{max_y} ({max_x-min_x+1}x{max_y-min_y+1})")
        
        # Create output filename
        date_str = datetime.now().strftime('%Y%m%d')
        output_filename = f"{patient_id}_{task_type}_ocs_{date_str}.png"
        output_path = os.path.join(output_directory, output_filename)
        
        # Render to PNG
        canvas_size = (config['canvas_width'], config['canvas_height'])
        success = render_red_pixels_to_image(red_mask, bbox, output_path, canvas_size)
        
        if success:
            print(f"  ✓ Saved: {output_filename} ({canvas_size[0]}x{canvas_size[1]})")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  ✗ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_ocs_images(base_directory):
    """Find all OCS PNG files in the directory tree."""
    ocs_files = []
    
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.png') and ('COPY' in file or 'RECALL' in file):
                ocs_files.append(os.path.join(root, file))
    
    return sorted(ocs_files)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Extract red-pixel drawings from OCS images')
    parser.add_argument('--input', required=True, help='Input directory containing OCS PNG images')
    parser.add_argument('--output', required=True, help='Output directory for processed images')
    parser.add_argument('--config', default=None, help='Path to config file (default: ./ocs_extractor.conf)')
    
    args = parser.parse_args()
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        # Use config in same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, CONFIG_FILE)
    
    print("=" * 80)
    print("OCS Image Extractor - Batch Processing")
    print("=" * 80)
    print(f"Input directory:  {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Config file:      {config_path}")
    print("=" * 80)
    
    # Check/create config
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
    
    auto_crop = config.get('auto_crop', True)
    padding = config.get('padding_px', 5)
    
    print(f"Auto-crop: {'enabled' if auto_crop else 'disabled'}")
    print(f"Padding: {padding}px")
    
    # Find all OCS images
    print(f"\nSearching for OCS images in: {args.input}")
    ocs_files = find_ocs_images(args.input)
    print(f"Found {len(ocs_files)} OCS image(s)")
    
    if len(ocs_files) == 0:
        print("No OCS images found!")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process each file
    successful = 0
    failed = 0
    
    for ocs_file in ocs_files:
        try:
            success = process_ocs_image(ocs_file, args.output, config)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ ERROR processing {os.path.basename(ocs_file)}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successfully processed: {successful}")
    print(f"Failed:                 {failed}")
    print(f"Output directory:       {args.output}")
    print(f"Canvas resolution:      {config['canvas_width']}x{config['canvas_height']} (from config)")
    print("=" * 80)


if __name__ == '__main__':
    main()

