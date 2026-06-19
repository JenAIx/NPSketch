#!/usr/bin/env python3
"""
Line Thickness Normalizer

Normalizes line thickness in binary images to a consistent width using
skeletonization and morphological dilation.

This ensures consistent line thickness across different input sources
(MAT files, OCS images, etc.) for CNN training.
"""

import cv2
import numpy as np
from PIL import Image


def normalize_line_thickness(image_array, target_thickness=2, threshold=127):
    """
    Normalize line thickness to a consistent width.
    
    Process:
    1. Binarize image (black lines on white background)
    2. Skeletonize to 1-pixel thin lines
    3. Dilate to target thickness
    
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
    
    # Skeletonize to 1-pixel thin lines
    # Using Zhang-Suen thinning algorithm
    skeleton = cv2.ximgproc.thinning(binary_inv, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    
    # If target thickness is 1, we're done
    if target_thickness <= 1:
        skeleton_inv = cv2.bitwise_not(skeleton)
        return cv2.cvtColor(skeleton_inv, cv2.COLOR_GRAY2RGB)
    
    # Dilate to target thickness.
    # A (k,k) structuring element grows the 1px skeleton to ~k px width;
    # the previous formula (target_thickness - 1) yielded a 1x1 kernel for
    # target 2, i.e. NO dilation - all "2px" images were actually 1px.
    kernel_size = max(1, int(round(target_thickness)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    dilated = cv2.dilate(skeleton, kernel, iterations=1)
    
    # Invert back: white background, black lines
    result = cv2.bitwise_not(dilated)
    
    # Convert to RGB
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    return result_rgb


def normalize_to_canvas(raw_bytes, canvas=(568, 274), padding=5, target_thickness=2):
    """Produce the canonical CNN-ready image from arbitrary input bytes:
    auto-crop to content (+padding), scale & center onto a `canvas`, normalize line
    thickness. Mirrors the training-data import / save-drawn-image normalization so the
    preview (predict) and the stored `processed_image_data` are identical. Returns PNG bytes.
    """
    import io
    W, H = canvas
    img = Image.open(io.BytesIO(raw_bytes))
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255)); bg.paste(img, mask=img.split()[3]); img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    arr = np.array(img)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)

    final = np.ones((H, W, 3), dtype=np.uint8) * 255
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x = max(0, x - padding); y = max(0, y - padding)
        w = min(arr.shape[1] - x, w + 2 * padding); h = min(arr.shape[0] - y, h + 2 * padding)
        crop = arr[y:y + h, x:x + w]
        scale = min(W / w, H / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = np.array(Image.fromarray(crop).resize((nw, nh), Image.Resampling.LANCZOS))
        ox, oy = (W - nw) // 2, (H - nh) // 2
        final[oy:oy + nh, ox:ox + nw] = resized

    normalized = normalize_line_thickness(final, target_thickness=target_thickness)
    out = io.BytesIO(); Image.fromarray(normalized, mode='RGB').save(out, format='PNG')
    return out.getvalue()


def normalize_image_file(input_path, output_path, target_thickness=2):
    """
    Normalize line thickness in an image file.

    Args:
        input_path: Path to input PNG
        output_path: Path to output PNG
        target_thickness: Desired line thickness in pixels
    
    Returns:
        bool: Success status
    """
    try:
        # Load image
        img = Image.open(input_path)
        img_array = np.array(img)
        
        # Normalize
        normalized = normalize_line_thickness(img_array, target_thickness)
        
        # Save
        result_img = Image.fromarray(normalized, mode='RGB')
        result_img.save(output_path)
        
        return True
    except Exception as e:
        print(f"Error normalizing {input_path}: {e}")
        return False


def verify_line_thickness(image_path):
    """
    Verify line thickness after normalization.
    
    Returns:
        Tuple of (avg_thickness, max_thickness, line_pixel_count)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold to binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Get non-zero pixels (lines)
    line_pixels = binary > 0
    line_count = np.sum(line_pixels)
    
    if line_count == 0:
        return 0, 0, 0
    
    # Average distance from center of line
    avg_distance = np.mean(dist_transform[line_pixels])
    max_distance = np.max(dist_transform[line_pixels])
    
    # Thickness is approximately 2 * distance
    avg_thickness = avg_distance * 2
    max_thickness = max_distance * 2
    
    return avg_thickness, max_thickness, line_count


if __name__ == '__main__':
    """
    Test script for line normalization.
    """
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python3 line_normalizer.py <input_image> <output_image> [thickness]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    thickness = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    print(f"Normalizing {input_file} -> {output_file}")
    print(f"Target thickness: {thickness}px")
    
    success = normalize_image_file(input_file, output_file, thickness)
    
    if success:
        print("✓ Normalization complete")
        
        # Verify
        avg, max_t, pixels = verify_line_thickness(output_file)
        print(f"Result: avg={avg:.2f}px, max={max_t:.2f}px, {pixels:,} line pixels")
    else:
        print("✗ Normalization failed")
        sys.exit(1)

