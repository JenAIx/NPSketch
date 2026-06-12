"""
Upload Router - Image upload and processing endpoints for NPSketch API

Contains endpoints for:
- Image upload with duplicate detection
- Duplicate checking
- Image normalization (STEP 1)
- Image registration (STEP 3)
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db, TrainingDataImage
import io
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/check-duplicate")
async def check_duplicate(
    file: UploadFile = File(...),
    original_file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """
    Check if an image already exists in the database (by content hash of ORIGINAL).
    Returns duplicate info without storing the image.
    
    Args:
        file: Normalized image file to check
        original_file: Optional original file (before normalization) for hash calculation
        db: Database session
        
    Returns:
        Dict with 'is_duplicate' flag and optional 'existing_id'
    """
    import hashlib
    
    try:
        # Calculate hash from ORIGINAL file if provided
        if original_file:
            original_content = await original_file.read()
            image_hash = hashlib.sha256(original_content).hexdigest()
        else:
            # Fallback: hash the file we received
            content = await file.read()
            image_hash = hashlib.sha256(content).hexdigest()
        
        # Check if exists in the training-data table
        existing_training = db.query(TrainingDataImage).filter(
            TrainingDataImage.image_hash == image_hash
        ).first()

        if existing_training:
            return {
                "is_duplicate": True,
                "existing_id": existing_training.id,
                "existing_filename": existing_training.original_filename or f"{existing_training.patient_id}_{existing_training.task_type}",
                "uploaded_at": existing_training.uploaded_at.isoformat() if existing_training.uploaded_at else "N/A",
                "uploader": "Training Database",
                "source": "training_data",
                "patient_id": existing_training.patient_id,
                "task_type": existing_training.task_type,
                "source_format": existing_training.source_format
            }
        else:
            return {
                "is_duplicate": False
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Check failed: {str(e)}")


@router.post("/normalize-image")
async def normalize_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    STEP 1: Normalize image to 568×274 using the same proven routine as AI training data.
    
    Process (same as ai_training_data_upload):
    1. Auto-crop to content with 5px padding
    2. Scale to 568×274 (preserving aspect ratio)
    3. Center on canvas
    
    NOTE: Line thickness normalization is NOT done here - that's handled by register-image
    if Auto Match is enabled, or left as-is for display.
    """
    import cv2
    import numpy as np
    from PIL import Image
    
    try:
        logger.info("=" * 60)
        logger.info("NORMALIZE IMAGE TO 568×274 (same as AI training)")
        logger.info("=" * 60)
        
        # Read uploaded file
        content = await file.read()
        
        # Use PIL for consistent handling (same as training_data.py)
        pil_image = Image.open(io.BytesIO(content))
        
        # Convert to RGB if needed (canvas may send RGBA)
        if pil_image.mode == 'RGBA':
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        image_array = np.array(pil_image)
        logger.info(f"Uploaded: {image_array.shape[1]}×{image_array.shape[0]}")
        
        # Target dimensions
        TARGET_W, TARGET_H = 568, 274
        
        # Step 1: Auto-crop to content with padding (same as training_data.py)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        coords = cv2.findNonZero(binary)
        
        if coords is not None:
            # Calculate bounding box with padding
            x, y, w, h = cv2.boundingRect(coords)
            padding = 5  # Same as AI training: 5px padding
            
            # Add padding (with bounds checking)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image_array.shape[1] - x, w + 2 * padding)
            h = min(image_array.shape[0] - y, h + 2 * padding)
            
            # Crop to bounding box
            cropped_array = image_array[y:y+h, x:x+w]
            logger.info(f"Auto-cropped with {padding}px padding: {image_array.shape[1]}×{image_array.shape[0]} → {w}×{h}")
            
            # Step 2: Scale to 568×274 (preserving aspect ratio)
            scale = min(TARGET_W / w, TARGET_H / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize cropped content using PIL (LANCZOS for quality)
            cropped_img = Image.fromarray(cropped_array)
            resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.info(f"Scaled: {w}×{h} × {scale:.2f} = {new_w}×{new_h}")
            
            # Step 3: Center on 568×274 canvas
            final_canvas = np.ones((TARGET_H, TARGET_W, 3), dtype=np.uint8) * 255
            offset_x = (TARGET_W - new_w) // 2
            offset_y = (TARGET_H - new_h) // 2
            final_canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = np.array(resized_img)
            logger.info(f"Centered at: ({offset_x}, {offset_y}) on {TARGET_W}×{TARGET_H} canvas")
            
            result = final_canvas
        else:
            # No content detected - simple resize
            logger.warning("No content detected, simple resize")
            resized = pil_image.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            result = np.array(resized)
        
        logger.info(f"Normalized to: {TARGET_W}×{TARGET_H}")
        logger.info("=" * 60)
        
        # Convert RGB to BGR for cv2.imencode
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        # Encode as PNG
        success, buffer = cv2.imencode('.png', result_bgr)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Normalization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


