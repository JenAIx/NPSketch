#!/usr/bin/env python3
"""
Oxford Image Normalizer

Normalizes Oxford manual rater PNG images to 568×274 with 2px line thickness,
using the same process as MAT/OCS extractors:
1. Auto-crop to content (5px padding)
2. Resize to 568×274
3. Normalize line thickness to 2.00px (Zhang-Suen + dilation)

This script processes all PNG images in the Oxford dataset and produces
normalized versions suitable for CNN training, matching the format used
by MAT and OCS extractors.

Usage:
    python3 oxford_normalizer.py <input_dir> <output_dir>

Example:
    python3 oxford_normalizer.py \\
        /app/templates/training_data_oxford_manual_rater_202512/imgs \\
        /app/templates/training_data_oxford_manual_rater_202512/imgs_normalized_568x274

Author: NPSketch Team
Date: 2025-12-22
"""

import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from line_normalizer import normalize_line_thickness


def calculate_bounding_box_from_image(image_data, padding=5, threshold=250):
    """
    Calculate bounding box from image by finding non-white pixels.
    
    This function identifies the content area of an image by finding all pixels
    that are darker than the threshold (typically non-white pixels representing
    the drawing). It then calculates the bounding box that encompasses all content
    and adds padding around it.
    
    Process:
    1. Convert image to grayscale if needed
    2. Create binary mask of content pixels (pixels < threshold)
    3. Find bounding box coordinates
    4. Add padding to all sides
    
    Args:
        image_data: RGB numpy array (H×W×3) or grayscale array (H×W)
        padding: Padding in pixels to add around content (default: 5)
        threshold: Pixel value threshold - pixels < threshold are considered 
                   content (default: 250, meaning pixels darker than 250/255)
    
    Returns:
        Tuple (min_x, min_y, max_x, max_y) representing bounding box coordinates,
        or None if no content found (empty image)
    
    Example:
        >>> img = np.array(Image.open('drawing.png'))
        >>> bbox = calculate_bounding_box_from_image(img, padding=5)
        >>> min_x, min_y, max_x, max_y = bbox
        >>> cropped = img[min_y:max_y+1, min_x:max_x+1]
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


def normalize_oxford_image(input_path, output_path, target_size=(568, 274), 
                          auto_crop=True, padding=5, target_thickness=2, verbose=False):
    """
    Normalize Oxford image: auto-crop, resize, normalize line thickness.
    
    This function performs the complete normalization pipeline:
    1. Load image and convert to RGB
    2. Auto-crop to content (remove white space, add padding)
    3. Resize to target resolution (568×274)
    4. Normalize line thickness to 2.00px using Zhang-Suen skeletonization + dilation
    
    The process matches exactly what MAT and OCS extractors do, ensuring
    consistent format across all training data sources.
    
    Args:
        input_path: Path to input PNG file
        output_path: Path to output PNG file (will be created)
        target_size: Target resolution as (width, height) tuple (default: (568, 274))
        auto_crop: Enable auto-cropping to content (default: True)
        padding: Padding in pixels to add around content after cropping (default: 5)
        target_thickness: Target line thickness in pixels (default: 2)
        verbose: Enable verbose logging (default: False)
    
    Returns:
        bool: True if successful, False otherwise
    
    Raises:
        No exceptions raised - all errors are caught and logged
    
    Example:
        >>> normalize_oxford_image(
        ...     'input.png',
        ...     'output.png',
        ...     target_size=(568, 274),
        ...     auto_crop=True,
        ...     padding=5,
        ...     target_thickness=2
        ... )
        True
    """
    try:
        if verbose:
            print(f"    Loading image: {os.path.basename(input_path)}")
        
        # Load image
        img = Image.open(input_path)
        original_size = img.size
        
        if verbose:
            print(f"    Original size: {original_size[0]}×{original_size[1]}px")
        
        # Convert to RGB if needed (handles RGBA, L, P modes)
        if img.mode != 'RGB':
            if verbose:
                print(f"    Converting from {img.mode} to RGB")
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Auto-crop to content
        if auto_crop:
            if verbose:
                print(f"    Auto-cropping with {padding}px padding...")
            bbox = calculate_bounding_box_from_image(img_array, padding=padding)
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                cropped_width = max_x - min_x + 1
                cropped_height = max_y - min_y + 1
                if verbose:
                    print(f"    Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")
                    print(f"    Cropped size: {cropped_width}×{cropped_height}px")
                img_array = img_array[int(min_y):int(max_y)+1, int(min_x):int(max_x)+1]
                # Create new PIL image from cropped array
                img = Image.fromarray(img_array, mode='RGB')
            else:
                if verbose:
                    print(f"    Warning: No content found, skipping crop")
        
        # Resize to target size
        if img.size != target_size:
            if verbose:
                print(f"    Resizing from {img.size[0]}×{img.size[1]} to {target_size[0]}×{target_size[1]}px")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img)
        else:
            if verbose:
                print(f"    Already at target size, skipping resize")
        
        # Normalize line thickness to 2.00px
        if verbose:
            print(f"    Normalizing line thickness to {target_thickness}px...")
        normalized = normalize_line_thickness(img_array, target_thickness=target_thickness)
        
        # Save
        if verbose:
            print(f"    Saving to: {os.path.basename(output_path)}")
        result_img = Image.fromarray(normalized, mode='RGB')
        result_img.save(output_path)
        
        if verbose:
            print(f"    ✓ Success!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {os.path.basename(input_path)}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def normalize_oxford_directory(input_dir, output_dir, target_size=(568, 274), verbose=False):
    """
    Normalize all PNG images in a directory.
    
    This function processes all PNG files in the input directory and creates
    normalized versions in the output directory. It provides progress tracking,
    error reporting, and timing information.
    
    Process for each image:
    1. Load and convert to RGB
    2. Auto-crop to content (5px padding)
    3. Resize to target size (568×274)
    4. Normalize line thickness (2.00px)
    5. Save to output directory
    
    Args:
        input_dir: Input directory containing PNG files
        output_dir: Output directory for normalized images (will be created)
        target_size: Target resolution as (width, height) tuple (default: (568, 274))
        verbose: Enable verbose logging for each image (default: False)
    
    Returns:
        dict: Statistics dictionary with keys:
            - 'total': Total number of files
            - 'success': Number of successfully processed files
            - 'errors': Number of files with errors
            - 'duration': Processing duration in seconds
    
    Example:
        >>> stats = normalize_oxford_directory(
        ...     '/app/templates/oxford/imgs',
        ...     '/app/templates/oxford/imgs_normalized',
        ...     target_size=(568, 274)
        ... )
        >>> print(f"Processed {stats['success']}/{stats['total']} files")
    """
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Created output directory: {output_dir}")
    
    # Find all PNG files
    png_files = sorted(glob.glob(os.path.join(input_dir, '*.png')) + 
                      glob.glob(os.path.join(input_dir, '*.PNG')))
    
    if not png_files:
        print(f"⚠ [WARNING] No PNG files found in {input_dir}")
        return {'total': 0, 'success': 0, 'errors': 0, 'duration': 0}
    
    # Print header
    print("=" * 80)
    print("OXFORD IMAGE NORMALIZER")
    print("=" * 80)
    print(f"Started at:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target size:      {target_size[0]}×{target_size[1]}px")
    print(f"Target thickness: 2.00px")
    print(f"Auto-crop:        Enabled (5px padding)")
    print(f"Total files:      {len(png_files)}")
    print("=" * 80)
    print()
    
    success_count = 0
    error_count = 0
    error_files = []
    
    # Process each file
    for i, input_path in enumerate(png_files, 1):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        # Progress indicator
        progress_pct = (i / len(png_files)) * 100
        elapsed = time.time() - start_time
        if i > 1:
            avg_time = elapsed / (i - 1)
            remaining = avg_time * (len(png_files) - i)
            eta_str = f"ETA: {remaining:.0f}s"
        else:
            eta_str = "ETA: calculating..."
        
        print(f"[{i:4d}/{len(png_files)}] ({progress_pct:5.1f}%) {eta_str} | Processing {filename}...", 
              end=' ', flush=True)
        
        file_start = time.time()
        
        success = normalize_oxford_image(
            input_path, 
            output_path,
            target_size=target_size,
            auto_crop=True,
            padding=5,
            target_thickness=2,
            verbose=verbose
        )
        
        file_duration = time.time() - file_start
        
        if success:
            print(f"✓ ({file_duration:.2f}s)")
            success_count += 1
        else:
            print(f"✗ ({file_duration:.2f}s)")
            error_count += 1
            error_files.append(filename)
        
        # Print progress every 50 files
        if i % 50 == 0:
            elapsed_total = time.time() - start_time
            print(f"    [PROGRESS] Processed {i}/{len(png_files)} files "
                  f"({success_count} success, {error_count} errors) "
                  f"in {elapsed_total:.1f}s")
    
    # Print summary
    total_duration = time.time() - start_time
    print()
    print("=" * 80)
    print("NORMALIZATION COMPLETE")
    print("=" * 80)
    print(f"Finished at:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:        {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"Total files:     {len(png_files)}")
    print(f"Success:         {success_count} ({100*success_count/len(png_files):.1f}%)")
    print(f"Errors:          {error_count} ({100*error_count/len(png_files):.1f}%)")
    if error_files:
        print(f"\nFiles with errors ({len(error_files)}):")
        for err_file in error_files[:10]:  # Show first 10 errors
            print(f"  - {err_file}")
        if len(error_files) > 10:
            print(f"  ... and {len(error_files) - 10} more")
    print("=" * 80)
    
    return {
        'total': len(png_files),
        'success': success_count,
        'errors': error_count,
        'duration': total_duration,
        'error_files': error_files
    }


if __name__ == '__main__':
    """
    Main entry point for the Oxford image normalizer.
    
    Command-line arguments:
        input_dir:  Directory containing input PNG files
        output_dir: Directory for normalized output PNG files
        --verbose:  (Optional) Enable verbose logging for each image
    
    Exit codes:
        0: Success
        1: Invalid arguments
        2: Processing errors occurred
    """
    if len(sys.argv) < 3:
        print("Usage: python3 oxford_normalizer.py <input_dir> <output_dir> [--verbose]")
        print()
        print("Arguments:")
        print("  input_dir   Directory containing input PNG files")
        print("  output_dir  Directory for normalized output PNG files")
        print("  --verbose   (Optional) Enable verbose logging for each image")
        print()
        print("Example:")
        print("  python3 oxford_normalizer.py \\")
        print("    /app/templates/training_data_oxford_manual_rater_202512/imgs \\")
        print("    /app/templates/training_data_oxford_manual_rater_202512/imgs_normalized_568x274")
        print()
        print("  python3 oxford_normalizer.py \\")
        print("    /app/templates/training_data_oxford_manual_rater_202512/imgs \\")
        print("    /app/templates/training_data_oxford_manual_rater_202512/imgs_normalized_568x274 \\")
        print("    --verbose")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    # Validate input directory
    if not os.path.isdir(input_dir):
        print(f"✗ [ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Run normalization
    stats = normalize_oxford_directory(
        input_dir, 
        output_dir, 
        target_size=(568, 274),
        verbose=verbose
    )
    
    # Exit with error code if there were processing errors
    if stats['errors'] > 0:
        print(f"\n⚠ [WARNING] {stats['errors']} files had errors during processing")
        sys.exit(2)
    
    print("\n✓ [SUCCESS] All files processed successfully!")
    sys.exit(0)

