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
    STEP 1: Simple normalization - Auto-crop + Scale + Center to 568×274
    No registration, no thinning - just prepare the image
    """
    import cv2
    import numpy as np
    from image_processing.utils import load_image_from_bytes
    
    try:
        print("=" * 60)
        print("🔄 STEP 1: NORMALIZE IMAGE TO 568×274")
        print("=" * 60)
        
        # Read uploaded file
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        print(f"✓ Uploaded: {img.shape}")
        
        # Get reference for size matching
        reference_service = ReferenceService(db)
        ref_image = reference_service.get_reference_by_name("default_reference")
        if not ref_image:
            raise HTTPException(status_code=404, detail="Reference image not found")
        
        ref_img_data = load_image_from_bytes(ref_image.processed_image_data)
        ref_h, ref_w = ref_img_data.shape[:2]
        
        # Convert to grayscale for object detection
        gray_upload = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_upload = cv2.threshold(gray_upload, 127, 255, cv2.THRESH_BINARY_INV)
        coords_upload = np.column_stack(np.where(binary_upload > 0))
        
        gray_ref = cv2.cvtColor(ref_img_data, cv2.COLOR_BGR2GRAY)
        _, binary_ref = cv2.threshold(gray_ref, 127, 255, cv2.THRESH_BINARY_INV)
        coords_ref = np.column_stack(np.where(binary_ref > 0))
        
        if len(coords_upload) > 0 and len(coords_ref) > 0:
            # 1. AUTO-CROP Upload WITH PADDING to preserve edges
            padding = 10  # Add 10 pixels padding on each side to prevent losing lines at edges
            
            up_min_y, up_min_x = coords_upload.min(axis=0)
            up_max_y, up_max_x = coords_upload.max(axis=0)
            
            # Add padding (with bounds checking)
            img_h, img_w = img.shape[:2]
            up_min_y = max(0, up_min_y - padding)
            up_min_x = max(0, up_min_x - padding)
            up_max_y = min(img_h - 1, up_max_y + padding)
            up_max_x = min(img_w - 1, up_max_x + padding)
            
            cropped = img[up_min_y:up_max_y+1, up_min_x:up_max_x+1]
            print(f"  1️⃣  Cropped with {padding}px padding: {img.shape[:2]} → {cropped.shape[:2]}")
            
            # 2. SCALE to FIT 568×274 canvas (max width OR height)
            up_h, up_w = cropped.shape[:2]
            
            # Calculate scale to fit canvas (use MIN to ensure it fits!)
            scale_h = ref_h / up_h if up_h > 0 else 1.0  # 256 / object_height
            scale_w = ref_w / up_w if up_w > 0 else 1.0  # 256 / object_width
            scale = min(scale_h, scale_w)  # MIN = fits in both directions!
            scale = np.clip(scale, 0.1, 10.0)  # Safety limits
            
            # Use round() instead of int() to avoid truncation and pixel loss
            # Ensure we don't exceed canvas dimensions
            scaled_h = min(round(up_h * scale), ref_h)
            scaled_w = min(round(up_w * scale), ref_w)
            scaled = cv2.resize(cropped, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            print(f"  2️⃣  Scaled to fit canvas: {up_h}×{up_w} × {scale:.2f} = {scaled_h}×{scaled_w}")
            print(f"     (scale_h={scale_h:.2f}, scale_w={scale_w:.2f} → using min={scale:.2f}, rounded to preserve content)")
            
            # 3. CENTER on 568×274 canvas
            result = np.ones((ref_h, ref_w, 3), dtype=np.uint8) * 255
            
            # With min() scaling, object should always fit - but check anyway
            if scaled_h > ref_h or scaled_w > ref_w:
                # Shouldn't happen with min(), but handle it
                src_y = max(0, (scaled_h - ref_h) // 2)
                src_x = max(0, (scaled_w - ref_w) // 2)
                result = scaled[src_y:src_y+ref_h, src_x:src_x+ref_w].copy()
                print(f"  3️⃣  Centered (cropped - unexpected!): {scaled_h}×{scaled_w} → {ref_h}×{ref_w}")
            else:
                # Normal case: place centered
                y_offset = (ref_h - scaled_h) // 2
                x_offset = (ref_w - scaled_w) // 2
                result[y_offset:y_offset+scaled_h, x_offset:x_offset+scaled_w] = scaled
                print(f"  3️⃣  Centered at: ({x_offset}, {y_offset}) - object fills max width/height ✓")
            
        else:
            print("  ⚠️  No objects detected, simple resize")
            result = cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
        
        print(f"✅ Normalized to: {result.shape}")
        print("=" * 60)
        
        # Encode as PNG
        success, buffer = cv2.imencode('.png', result)
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
