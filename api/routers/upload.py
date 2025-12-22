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
from database import get_db, UploadedImage
from models import UploadResponse, EvaluationResultResponse
from services import ReferenceService, EvaluationService
import io

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
        
        # Check if exists
        existing = db.query(UploadedImage).filter(
            UploadedImage.image_hash == image_hash
        ).first()
        
        if existing:
            return {
                "is_duplicate": True,
                "existing_id": existing.id,
                "existing_filename": existing.filename,
                "uploaded_at": existing.uploaded_at.isoformat(),
                "uploader": existing.uploader
            }
        else:
            return {
                "is_duplicate": False
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Check failed: {str(e)}")


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    original_file: UploadFile = File(None),  # Optional: original file before normalization
    uploader: str = Form(None),
    reference_name: str = Form("default_reference"),
    db: Session = Depends(get_db)
):
    """
    Upload and evaluate a hand-drawn image.
    
    Args:
        file: Processed/normalized image file to analyze
        original_file: Optional original file (before normalization) for hash calculation
        uploader: Optional uploader identifier
        reference_name: Name of reference to compare against
        db: Database session
        
    Returns:
        Upload result with evaluation metrics
    """
    try:
        # Read processed file content (for analysis)
        processed_content = await file.read()
        
        # Read original file content (for hash and storage)
        original_content = None
        if original_file:
            original_content = await original_file.read()
        
        # Process and evaluate
        eval_service = EvaluationService(db)
        uploaded_image, evaluation = eval_service.process_upload(
            processed_content,
            file.filename,
            reference_name,
            uploader,
            original_image_bytes=original_content  # Pass original for hash calculation
        )
        
        return UploadResponse(
            success=True,
            message="Image uploaded and evaluated successfully",
            image_id=uploaded_image.id,
            evaluation=EvaluationResultResponse.model_validate(evaluation)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


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
        print("=" * 60)
        print("🔄 NORMALIZE IMAGE TO 568×274 (same as AI training)")
        print("=" * 60)
        
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
        print(f"✓ Uploaded: {image_array.shape[1]}×{image_array.shape[0]}")
        
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
            print(f"  1️⃣  Auto-cropped with {padding}px padding: {image_array.shape[1]}×{image_array.shape[0]} → {w}×{h}")
            
            # Step 2: Scale to 568×274 (preserving aspect ratio)
            scale = min(TARGET_W / w, TARGET_H / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize cropped content using PIL (LANCZOS for quality)
            cropped_img = Image.fromarray(cropped_array)
            resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            print(f"  2️⃣  Scaled: {w}×{h} × {scale:.2f} = {new_w}×{new_h}")
            
            # Step 3: Center on 568×274 canvas
            final_canvas = np.ones((TARGET_H, TARGET_W, 3), dtype=np.uint8) * 255
            offset_x = (TARGET_W - new_w) // 2
            offset_y = (TARGET_H - new_h) // 2
            final_canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = np.array(resized_img)
            print(f"  3️⃣  Centered at: ({offset_x}, {offset_y}) on {TARGET_W}×{TARGET_H} canvas")
            
            result = final_canvas
        else:
            # No content detected - simple resize
            print("  ⚠️  No content detected, simple resize")
            resized = pil_image.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            result = np.array(resized)
        
        print(f"✅ Normalized to: {TARGET_W}×{TARGET_H}")
        print("=" * 60)
        
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
        print(f"❌ Normalization error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register-image")
async def register_image(
    file: UploadFile = File(...),
    enable_translation: bool = Form(True),
    enable_rotation: bool = Form(True),
    enable_scale: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    STEP 3: Auto Match - Apply registration + line detection + thin to 1px
    Input: Already normalized 568×274 image from frontend
    Output: Registered + thinned 1px lines
    """
    import cv2
    import numpy as np
    from image_processing.image_registration import ImageRegistration
    from image_processing.utils import load_image_from_bytes
    from skimage.morphology import skeletonize
    
    try:
        print("=" * 60)
        print("🔄 STEP 3: AUTO MATCH (Registration + Thinning)")
        print("=" * 60)
        print(f"  Options: translation={enable_translation}, rotation={enable_rotation}, scale={enable_scale}")
        
        # Read uploaded file (should already be 568×274 normalized from frontend)
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        print(f"  ✓ Input image: {img.shape} (should be 568×274)")
        
        # Get reference image
        reference_service = ReferenceService(db)
        ref_image = reference_service.get_reference_by_name("default_reference")
        if not ref_image:
            raise HTTPException(status_code=404, detail="Reference image not found")
        
        ref_img_data = load_image_from_bytes(ref_image.processed_image_data)
        print(f"  ✓ Reference: {ref_img_data.shape}")
        
        result_img = img.copy()
        
        # STEP 3a: Registration (if enabled)
        if enable_translation or enable_rotation or enable_scale:
            print("\n  🎯 STEP 3a: Registration...")
            registration = ImageRegistration()
            
            # Build motion type based on enabled options
            if enable_translation and enable_rotation and enable_scale:
                motion_type = 'similarity'  # All transformations
            elif enable_translation and enable_rotation:
                motion_type = 'euclidean'  # Translation + Rotation
            elif enable_translation:
                motion_type = 'translation'  # Translation only
            else:
                motion_type = 'translation'  # Default
            
            print(f"     Motion type: {motion_type}")
            
            try:
                registered_img, reg_info = registration.register_images(
                    result_img,
                    ref_img_data,
                    method='ecc',
                    motion_type=motion_type,
                    max_rotation_degrees=30.0
                )
                
                if reg_info.get('success', False):
                    result_img = registered_img
                    print(f"     ✅ Registration: tx={reg_info.get('translation_x', 0):.1f}, ty={reg_info.get('translation_y', 0):.1f}, rot={reg_info.get('rotation_degrees', 0):.1f}°")
                else:
                    print(f"     ⚠️  Registration skipped: {reg_info.get('reason', 'Unknown')}")
                    
            except Exception as reg_error:
                print(f"     ⚠️  Registration failed: {reg_error}")
        else:
            print("\n  ⏭️  STEP 3a: Registration skipped (all disabled)")
        
        # STEP 3b: Line Thinning to 1px
        print("\n  ✂️  STEP 3b: Thinning to 1px...")
        
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Skeletonize to 1px
        skeleton_bool = skeletonize(binary > 0)
        skeleton = (skeleton_bool * 255).astype(np.uint8)
        thinned = cv2.bitwise_not(skeleton)
        
        # Convert back to color
        result_img = cv2.cvtColor(thinned, cv2.COLOR_GRAY2BGR)
        print(f"     ✅ Thinned to 1px: {result_img.shape}")
        
        print("\n✅ AUTO MATCH COMPLETE!")
        print("=" * 60)
        
        # Encode as PNG
        success, buffer = cv2.imencode('.png', result_img)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Auto Match error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
