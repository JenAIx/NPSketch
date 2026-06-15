"""
Shared Preprocessing Module for Training and Prediction

Ensures consistent image preprocessing across training and inference pipelines.
This is critical for model performance - train and predict must use identical preprocessing.

Preprocessing Steps:
1. Resize to 568×274 (preserving aspect ratio, centered on white canvas)
2. Binarization (threshold 175, black lines on white background)
3. Line thickness normalization (2.0px)
4. Pre-shrink (optional, configurable factor, creates margins)
5. Grayscale conversion (for model input)

Author: NPSketch Team
Date: 2026-01-20
"""

import numpy as np
import cv2
from PIL import Image
import io
import json
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any
from utils.logger import get_logger

logger = get_logger(__name__)

# Standard target dimensions
TARGET_WIDTH = 568
TARGET_HEIGHT = 274
BINARIZATION_THRESHOLD = 175
LINE_THICKNESS = 2.0


def get_model_input_size() -> Optional[Tuple[int, int]]:
    """
    CNN input resolution (width, height) from config, or None for full
    568x274. Downscaling speeds up CPU training; both dims are halved by
    default so the aspect ratio is preserved.
    """
    try:
        from config import get_config
        cfg = get_config().get('training.model_input', {}) or {}
    except Exception:
        return None
    if not cfg.get('enabled', False):
        return None
    w, h = cfg.get('width'), cfg.get('height')
    if not w or not h:
        return None
    return int(w), int(h)


def resize_for_model_input(gray: np.ndarray, size: Optional[Tuple[int, int]]) -> np.ndarray:
    """
    Resize a grayscale model-input array to (width, height).

    Applied identically in training and inference so train/predict stay in
    parity. INTER_AREA is used (best for downscaling). size=None is a no-op.
    """
    if size is None:
        return gray
    w, h = size
    if gray.shape[1] == w and gray.shape[0] == h:
        return gray
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)


def apply_pre_shrink(image: np.ndarray, factor: float = 0.90) -> np.ndarray:
    """
    Apply pre-shrink to create margins for rotation/translation tolerance.
    
    This shrinks the content and centers it on a white canvas of the same size,
    creating uniform margins around the content.
    
    Args:
        image: Input image (H×W or H×W×3), should be 568×274
        factor: Shrink factor (0.90 = 10% shrink, creates ~28px margins)
    
    Returns:
        Shrunk image centered on white canvas (same size as input)
    """
    if factor >= 1.0:
        return image
    
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    
    # Shrink image
    shrunk = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create white canvas
    if len(image.shape) == 3:
        canvas = np.ones((h, w, image.shape[2]), dtype=np.uint8) * 255
    else:
        canvas = np.ones((h, w), dtype=np.uint8) * 255
    
    # Center shrunk image on canvas
    offset_y, offset_x = (h - new_h) // 2, (w - new_w) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = shrunk
    
    logger.debug(f"Pre-shrink: {w}×{h} → content {new_w}×{new_h}, margins ~{offset_x}px")
    
    return canvas


def apply_binarization(image: np.ndarray, threshold: int = BINARIZATION_THRESHOLD) -> np.ndarray:
    """
    Binarize an image to pure black and white.
    
    Converts to grayscale if needed, applies threshold, then converts back 
    to RGB if input was RGB (for consistency with line normalization).
    
    Args:
        image: Input image (H×W grayscale or H×W×3 RGB)
        threshold: Binarization threshold (default: 175)
    
    Returns:
        Binarized image in same format as input (RGB if input was RGB)
    """
    is_rgb = len(image.shape) == 3
    
    # Convert to grayscale
    if is_rgb:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Convert back to RGB if input was RGB
    if is_rgb:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    return binary


def apply_line_normalization(image: np.ndarray, target_thickness: float = LINE_THICKNESS) -> np.ndarray:
    """
    Normalize line thickness to target value (default 2.0px).
    
    Uses Zhang-Suen skeletonization + dilation to achieve consistent line thickness.
    Requires RGB input (3 channels).
    
    Args:
        image: Input image (H×W×3 RGB)
        target_thickness: Target line thickness in pixels (default: 2.0)
    
    Returns:
        Image with normalized line thickness (H×W×3 RGB)
    """
    from line_normalizer import normalize_line_thickness
    return normalize_line_thickness(image, target_thickness=target_thickness)


def preprocess_for_training(
    image: np.ndarray,
    pre_shrink_enabled: bool = True,
    pre_shrink_factor: float = 0.90,
    apply_binarize: bool = True,
    apply_line_norm: bool = True,
    convert_to_grayscale: bool = True
) -> np.ndarray:
    """
    Apply full preprocessing pipeline for training data (non-augmented path).
    
    This ensures non-augmented images receive the same preprocessing as augmented:
    1. Pre-shrink (creates margins, introduces anti-aliasing)
    2. Binarization (removes anti-aliasing artifacts)
    3. Line thickness normalization (ensures consistent 2px lines)
    4. Grayscale conversion (for model input)
    
    Args:
        image: Input image (H×W×3 RGB, should be 568×274)
        pre_shrink_enabled: Apply pre-shrink (default: True)
        pre_shrink_factor: Pre-shrink factor (default: 0.90)
        apply_binarize: Apply binarization (default: True)
        apply_line_norm: Apply line normalization (default: True)
        convert_to_grayscale: Convert to grayscale at end (default: True)
    
    Returns:
        Preprocessed image (grayscale uint8 if convert_to_grayscale, else RGB uint8)
    """
    # Ensure RGB input
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Step 1: Pre-shrink
    if pre_shrink_enabled:
        image = apply_pre_shrink(image, factor=pre_shrink_factor)
    
    # Step 2: Binarization (especially important after shrink to fix anti-aliasing)
    if apply_binarize:
        image = apply_binarization(image)
    
    # Step 3: Line thickness normalization
    if apply_line_norm:
        image = apply_line_normalization(image)
    
    # Step 4: Convert to grayscale for model
    if convert_to_grayscale:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    return image


def _resize_to_target(image: np.ndarray) -> np.ndarray:
    """
    Resize to target dimensions with aspect ratio preserved and centered on white canvas.
    """
    h, w = image.shape[:2]
    if w == TARGET_WIDTH and h == TARGET_HEIGHT:
        return image

    scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.ones((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8) * 255
    offset_x = (TARGET_WIDTH - new_w) // 2
    offset_y = (TARGET_HEIGHT - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return canvas


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        rgb = image[:, :, :3].copy()
        alpha = image[:, :, 3:4] / 255.0
        white_bg = np.ones_like(rgb) * 255
        return (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
    return image


def _save_debug_image(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(image.shape) == 2:
        cv2.imwrite(path, image)
    else:
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _write_debug_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _timestamp_str() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


def preprocess_image_for_model(
    image: np.ndarray,
    apply_resize: bool = True,
    apply_binarization: bool = True,
    apply_line_normalization_flag: bool = True,
    pre_shrink_enabled: bool = False,
    pre_shrink_factor: float = 0.90,
    return_tensor: bool = True,
    model_input_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Full preprocessing pipeline for model input.
    
    This function ensures images are processed identically for training and inference.
    
    Args:
        image: Input image (any size, RGB or grayscale)
        apply_resize: Resize to 568×274 (should be True for external images)
        apply_binarization: Apply binary thresholding
        apply_line_normalization: Normalize line thickness to 2.0px
        pre_shrink_enabled: Apply pre-shrink (if model was trained with it)
        pre_shrink_factor: Pre-shrink factor (from model metadata)
        return_tensor: Return as normalized float32 for model input
    
    Returns:
        Preprocessed image ready for model input
    """
    # Ensure RGB
    image = _ensure_rgb(image)

    # Step 1: Resize to 568×274
    if apply_resize:
        image = _resize_to_target(image)

    # Steps 2-4 MUST match the training order in preprocess_for_training():
    # pre-shrink → binarize → line-norm. Line normalization comes LAST so the
    # final line thickness is invariant regardless of earlier resampling.
    # (The previous order - line-norm before pre-shrink - left ~0.4% of pixels
    # differing from the training transform.)

    # Step 2: Pre-shrink (if enabled)
    if pre_shrink_enabled:
        image = apply_pre_shrink(image, factor=pre_shrink_factor)

    # Step 3: Binarization (also removes anti-aliasing from resize/shrink)
    if apply_binarization:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, BINARIZATION_THRESHOLD, 255, cv2.THRESH_BINARY)
        image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    # Step 4: Line thickness normalization
    if apply_line_normalization_flag:
        image = apply_line_normalization(image, target_thickness=LINE_THICKNESS)

    # Step 5: Convert to grayscale for model
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 6: Resize to CNN input resolution (must match training)
    gray = resize_for_model_input(gray, model_input_size)

    if return_tensor:
        # Normalize to [0, 1] float32
        return gray.astype(np.float32) / 255.0

    return gray


def get_preprocessing_config_from_metadata(metadata: Dict) -> Dict:
    """
    Extract preprocessing configuration from model metadata.
    
    Args:
        metadata: Model metadata dictionary
    
    Returns:
        Preprocessing config dict with keys:
        - pre_shrink_enabled: bool
        - pre_shrink_factor: float
    """
    # Default config (no pre-shrink for backwards compatibility)
    config = {
        'pre_shrink_enabled': False,
        'pre_shrink_factor': 0.90,
        # None = full 568x274 (old models trained before model_input downscaling)
        'model_input_size': None
    }

    # Check augmentation config for pre-shrink settings
    augmentation = metadata.get('augmentation', {})
    if augmentation.get('enabled', False):
        aug_config = augmentation.get('config', {})
        pre_shrink = aug_config.get('pre_shrink', {})

        if pre_shrink:
            config['pre_shrink_enabled'] = pre_shrink.get('enabled', False)
            config['pre_shrink_factor'] = pre_shrink.get('factor', 0.90)
        else:
            # Metadata says augmentation was on but didn't record pre_shrink (older
            # models / incomplete metadata). Training applies pre_shrink by default, so
            # falling back to "off" would silently mismatch the model. Read the live
            # training config instead and log it.
            try:
                from config import get_config
                ps = get_config().get('augmentation.pre_shrink', {}) or {}
                config['pre_shrink_enabled'] = bool(ps.get('enabled', True))
                config['pre_shrink_factor'] = float(ps.get('factor', 0.90))
                logger.warning(
                    "Model metadata lacks augmentation.config.pre_shrink; falling back to "
                    f"live config (enabled={config['pre_shrink_enabled']}, "
                    f"factor={config['pre_shrink_factor']}). Re-train or patch metadata to make it self-describing.")
            except Exception as e:
                logger.warning(f"pre_shrink config missing in metadata and live-config fallback failed: {e}")

    # CNN input resolution used at training time (recorded in metadata)
    mi = metadata.get('model_input')
    if mi and mi.get('width') and mi.get('height'):
        config['model_input_size'] = (int(mi['width']), int(mi['height']))

    return config


def preprocess_pil_image_for_prediction(
    pil_image: Image.Image,
    metadata: Optional[Dict] = None,
    force_pre_shrink: bool = False,
    pre_shrink_factor: float = 0.90,
    debug: bool = False,
    debug_prefix: str = "optimized_drawing"
) -> np.ndarray:
    """
    Preprocess a PIL image for model prediction.
    
    Convenience function that handles PIL → numpy conversion and applies
    the full preprocessing pipeline based on model metadata.
    
    Args:
        pil_image: PIL Image object
        metadata: Model metadata (to determine if pre-shrink should be applied)
        force_pre_shrink: Force pre-shrink regardless of metadata
        pre_shrink_factor: Pre-shrink factor to use if forced
    
    Returns:
        Preprocessed image tensor (H, W) float32 normalized [0, 1]
    """
    # Convert PIL to RGB numpy
    if pil_image.mode != 'RGB':
        if pil_image.mode == 'RGBA':
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        else:
            pil_image = pil_image.convert('RGB')
    
    image = np.array(pil_image)
    
    # Determine pre-shrink settings
    pre_shrink_enabled = force_pre_shrink
    factor = pre_shrink_factor
    model_input_size = None

    if metadata and not force_pre_shrink:
        preproc_config = get_preprocessing_config_from_metadata(metadata)
        pre_shrink_enabled = preproc_config['pre_shrink_enabled']
        factor = preproc_config['pre_shrink_factor']
        model_input_size = preproc_config['model_input_size']

    # Log preprocessing config
    if pre_shrink_enabled:
        logger.info(f"Prediction preprocessing: pre-shrink enabled (factor={factor})")
    else:
        logger.debug("Prediction preprocessing: pre-shrink disabled")

    # Apply full preprocessing
    if debug:
        return _preprocess_with_debug(
            image=image,
            pre_shrink_enabled=pre_shrink_enabled,
            pre_shrink_factor=factor,
            debug_prefix=debug_prefix
        )

    return preprocess_image_for_model(
        image,
        apply_resize=True,
        apply_binarization=True,
        apply_line_normalization_flag=True,
        pre_shrink_enabled=pre_shrink_enabled,
        pre_shrink_factor=factor,
        return_tensor=True,
        model_input_size=model_input_size
    )


def preprocess_bytes_for_prediction(
    image_bytes: bytes,
    metadata: Optional[Dict] = None,
    force_pre_shrink: bool = False,
    pre_shrink_factor: float = 0.90,
    debug: bool = False,
    debug_prefix: str = "optimized_drawing"
) -> np.ndarray:
    """
    Preprocess image bytes for model prediction.
    
    Convenience function that handles bytes → PIL → numpy conversion.
    
    Args:
        image_bytes: Raw image bytes (PNG, JPG, etc.)
        metadata: Model metadata (to determine if pre-shrink should be applied)
        force_pre_shrink: Force pre-shrink regardless of metadata
        pre_shrink_factor: Pre-shrink factor to use if forced
    
    Returns:
        Preprocessed image tensor (H, W) float32 normalized [0, 1]
    """
    pil_image = Image.open(io.BytesIO(image_bytes))
    return preprocess_pil_image_for_prediction(
        pil_image, 
        metadata=metadata,
        force_pre_shrink=force_pre_shrink,
        pre_shrink_factor=pre_shrink_factor,
        debug=debug,
        debug_prefix=debug_prefix
    )


def _preprocess_with_debug(
    image: np.ndarray,
    pre_shrink_enabled: bool,
    pre_shrink_factor: float,
    debug_prefix: str
) -> np.ndarray:
    """
    Preprocess with debug artifacts written to /app/data/tmp.
    """
    timestamp = _timestamp_str()
    debug_dir = "/app/data/tmp"
    steps: List[Dict[str, Any]] = []

    def record_step(step_name: str, img: np.ndarray) -> None:
        filename = f"{debug_prefix}_{timestamp}_{step_name}.png"
        path = os.path.join(debug_dir, filename)
        _save_debug_image(path, img)
        steps.append({
            "step": step_name,
            "file": filename,
            "shape": list(img.shape),
            "dtype": str(img.dtype)
        })

    # Step 0: ensure RGB
    image = _ensure_rgb(image)
    record_step("input_rgb", image)

    # Step 1: resize to target
    image = _resize_to_target(image)
    record_step("resized", image)

    # Steps 2-4 in training order: pre-shrink → binarize → line-norm
    if pre_shrink_enabled:
        image = apply_pre_shrink(image, factor=pre_shrink_factor)
        record_step("pre_shrunk", image)

    image = apply_binarization(image)
    record_step("binarized", image)

    image = apply_line_normalization(image)
    record_step("line_normalized", image)

    # Step 5: final grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    record_step("final_gray", gray)

    # Write JSON manifest
    json_name = f"{debug_prefix}_{timestamp}_steps.json"
    json_path = os.path.join(debug_dir, json_name)
    _write_debug_json(json_path, {
        "timestamp_utc": timestamp,
        "debug_prefix": debug_prefix,
        "target_size": [TARGET_WIDTH, TARGET_HEIGHT],
        "binarization_threshold": BINARIZATION_THRESHOLD,
        "line_thickness": LINE_THICKNESS,
        "pre_shrink_enabled": pre_shrink_enabled,
        "pre_shrink_factor": pre_shrink_factor,
        "steps": steps
    })

    return gray.astype(np.float32) / 255.0
