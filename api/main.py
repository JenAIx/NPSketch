"""
NPSketch API - Automated Line Detection for Hand-Drawn Images

This FastAPI application provides endpoints for:
- Uploading and comparing hand-drawn images to reference templates
- Extracting line features using OpenCV
- Evaluating drawing accuracy
- Visualizing results

Author: ste
Date: October 2025
Version: 1.0
"""

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List
import os

from database import init_database, get_db, ReferenceImage, TestImage
from models import (
    UploadResponse,
    HealthResponse,
    EvaluationResultResponse,
    EvaluationUpdateRequest,
    ReferenceImageResponse,
    TestImageResponse
)
from services import ReferenceService, EvaluationService

# Initialize FastAPI app
app = FastAPI(
    title="NPSketch API",
    description="Automated line detection and comparison for hand-drawn images",
    version="1.0.0"
)

# Visualizations directory is created by evaluation service on first use
# and persisted via volume mount

# Mount static files for visualizations
app.mount("/api/visualizations", StaticFiles(directory="/app/data/visualizations"), name="visualizations")


@app.on_event("startup")
async def startup_event():
    """
    Initialize database and reference images on startup.
    """
    # Initialize database tables
    init_database()
    
    # Initialize default reference image
    db = next(get_db())
    try:
        ref_service = ReferenceService(db)
        ref_service.initialize_default_reference("default_reference")
        print("✓ Database initialized")
        print("✓ Default reference image loaded")
    finally:
        db.close()


@app.get("/api/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.
    
    Returns:
        System health status and database statistics
    """
    reference_count = db.query(ReferenceImage).count()
    
    return HealthResponse(
        status="healthy",
        database_initialized=True,
        reference_images_count=reference_count
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    uploader: str = None,
    reference_name: str = "default_reference",
    db: Session = Depends(get_db)
):
    """
    Upload and evaluate a hand-drawn image.
    
    Args:
        file: Image file to upload
        uploader: Optional uploader identifier
        reference_name: Name of reference to compare against
        db: Database session
        
    Returns:
        Upload result with evaluation metrics
    """
    try:
        # Read file content
        content = await file.read()
        
        # Process and evaluate
        eval_service = EvaluationService(db)
        uploaded_image, evaluation = eval_service.process_upload(
            content,
            file.filename,
            reference_name,
            uploader
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


@app.post("/api/normalize-image")
async def normalize_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    STEP 1: Simple normalization - Auto-crop + Scale + Center to 256×256
    No registration, no thinning - just prepare the image
    """
    import cv2
    import numpy as np
    import io
    from image_processing.utils import load_image_from_bytes
    
    try:
        print("=" * 60)
        print("🔄 STEP 1: NORMALIZE IMAGE TO 256×256")
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
            # 1. AUTO-CROP Upload
            up_min_y, up_min_x = coords_upload.min(axis=0)
            up_max_y, up_max_x = coords_upload.max(axis=0)
            cropped = img[up_min_y:up_max_y+1, up_min_x:up_max_x+1]
            print(f"  1️⃣  Cropped: {img.shape[:2]} → {cropped.shape[:2]}")
            
            # 2. SCALE to FIT 256×256 canvas (max width OR height)
            up_h, up_w = cropped.shape[:2]
            
            # Calculate scale to fit canvas (use MIN to ensure it fits!)
            scale_h = ref_h / up_h if up_h > 0 else 1.0  # 256 / object_height
            scale_w = ref_w / up_w if up_w > 0 else 1.0  # 256 / object_width
            scale = min(scale_h, scale_w)  # MIN = fits in both directions!
            scale = np.clip(scale, 0.1, 10.0)  # Safety limits
            
            scaled_h = int(up_h * scale)
            scaled_w = int(up_w * scale)
            scaled = cv2.resize(cropped, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            print(f"  2️⃣  Scaled to fit canvas: {up_h}×{up_w} × {scale:.2f} = {scaled_h}×{scaled_w}")
            print(f"     (scale_h={scale_h:.2f}, scale_w={scale_w:.2f} → using min={scale:.2f})")
            
            # 3. CENTER on 256×256 canvas
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


@app.post("/api/register-image")
async def register_image(
    file: UploadFile = File(...),
    enable_translation: bool = Form(True),
    enable_rotation: bool = Form(True),
    enable_scale: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    STEP 3: Auto Match - Apply registration + line detection + thin to 1px
    Input: Already normalized 256×256 image from frontend
    Output: Registered + thinned 1px lines
    """
    import cv2
    import numpy as np
    import io
    from image_processing.image_registration import ImageRegistration
    from image_processing.utils import load_image_from_bytes
    from skimage.morphology import skeletonize
    
    try:
        print("=" * 60)
        print("🔄 STEP 3: AUTO MATCH (Registration + Thinning)")
        print("=" * 60)
        print(f"  Options: translation={enable_translation}, rotation={enable_rotation}, scale={enable_scale}")
        
        # Read uploaded file (should already be 256×256 normalized from frontend)
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        print(f"  ✓ Input image: {img.shape} (should be 256×256)")
        
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


@app.get("/api/evaluations/recent", response_model=List[EvaluationResultResponse])
async def get_recent_evaluations(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get recent evaluation results.
    
    Args:
        limit: Maximum number of results to return
        db: Database session
        
    Returns:
        List of recent evaluations
    """
    eval_service = EvaluationService(db)
    evaluations = eval_service.list_recent_evaluations(limit)
    return [EvaluationResultResponse.model_validate(e) for e in evaluations]


@app.get("/api/evaluations/{eval_id}", response_model=EvaluationResultResponse)
async def get_evaluation(
    eval_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific evaluation result.
    
    Args:
        eval_id: Evaluation ID
        db: Database session
        
    Returns:
        Evaluation result
    """
    eval_service = EvaluationService(db)
    evaluation = eval_service.get_evaluation_by_id(eval_id)
    
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return EvaluationResultResponse.model_validate(evaluation)


@app.delete("/api/evaluations/{eval_id}")
async def delete_evaluation(
    eval_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete an evaluation result.
    
    Args:
        eval_id: Evaluation ID to delete
        db: Database session
        
    Returns:
        Success message
    """
    from database import EvaluationResult
    
    # Find evaluation
    evaluation = db.query(EvaluationResult).filter(EvaluationResult.id == eval_id).first()
    
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    # Delete visualization file if it exists
    if evaluation.visualization_path:
        viz_path = evaluation.visualization_path.replace('/api/visualizations/', '/app/data/visualizations/')
        if os.path.exists(viz_path):
            try:
                os.remove(viz_path)
            except Exception as e:
                print(f"Warning: Could not delete visualization file: {e}")
    
    # Delete from database
    db.delete(evaluation)
    db.commit()
    
    return {"message": "Evaluation deleted successfully", "id": eval_id}


@app.put("/api/evaluations/{eval_id}/evaluate", response_model=EvaluationResultResponse)
async def update_evaluation(
    eval_id: int,
    evaluation_data: EvaluationUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Update evaluation with user-provided correctness assessment.
    
    This endpoint allows users to manually evaluate the automated detection results,
    providing ground truth data for AI training purposes.
    
    Args:
        eval_id: Evaluation ID to update
        evaluation_data: User evaluation data (correct, missing, extra counts)
        db: Database session
        
    Returns:
        Updated evaluation result
    """
    from database import EvaluationResult
    from datetime import datetime
    
    # Find evaluation
    evaluation = db.query(EvaluationResult).filter(EvaluationResult.id == eval_id).first()
    
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    # Update user evaluation fields
    evaluation.user_evaluated = True
    evaluation.evaluated_correct = evaluation_data.evaluated_correct
    evaluation.evaluated_missing = evaluation_data.evaluated_missing
    evaluation.evaluated_extra = evaluation_data.evaluated_extra
    evaluation.evaluated_at_timestamp = datetime.utcnow()
    
    db.commit()
    db.refresh(evaluation)
    
    return EvaluationResultResponse.model_validate(evaluation)


@app.get("/api/references", response_model=List[ReferenceImageResponse])
async def list_references(db: Session = Depends(get_db)):
    """
    List all available reference images.
    
    Args:
        db: Database session
        
    Returns:
        List of reference images
    """
    ref_service = ReferenceService(db)
    references = ref_service.list_all_references()
    return [ReferenceImageResponse.model_validate(r) for r in references]


@app.get("/api/references/{ref_id}/image")
async def get_reference_image(
    ref_id: int,
    db: Session = Depends(get_db)
):
    """
    Get reference image data.
    
    Args:
        ref_id: Reference image ID
        db: Database session
        
    Returns:
        Image file
    """
    ref_service = ReferenceService(db)
    reference = ref_service.get_reference_by_id(ref_id)
    
    if not reference:
        raise HTTPException(status_code=404, detail="Reference not found")
    
    from fastapi.responses import Response
    return Response(content=reference.processed_image_data, media_type="image/png")


@app.get("/api/references/{ref_id}/features")
async def get_reference_features(
    ref_id: int,
    db: Session = Depends(get_db)
):
    """
    Get reference image features (detected lines).
    
    Args:
        ref_id: Reference image ID
        db: Database session
        
    Returns:
        Feature data including detected lines
    """
    ref_service = ReferenceService(db)
    reference = ref_service.get_reference_by_id(ref_id)
    
    if not reference:
        raise HTTPException(status_code=404, detail="Reference not found")
    
    import json
    features = json.loads(reference.feature_data)
    
    return {
        "reference_id": reference.id,
        "reference_name": reference.name,
        "num_lines": features.get("num_lines", 0),
        "lines": features.get("lines", []),
        "line_lengths": features.get("line_lengths", []),
        "line_angles": features.get("line_angles", []),
        "image_shape": features.get("image_shape", []),
        "num_contours": features.get("num_contours", 0)
    }


@app.post("/api/test-images", response_model=TestImageResponse)
async def create_test_image(
    file: UploadFile = File(...),
    test_name: str = Form(...),
    correct_lines: int = Form(...),
    missing_lines: int = Form(...),
    extra_lines: int = Form(...),
    db: Session = Depends(get_db)
):
    """
    Create a test image with manual scoring.
    
    Args:
        file: Image file
        test_name: Name/description of the test
        correct_lines: Expected number of correct lines
        missing_lines: Expected number of missing lines
        extra_lines: Expected number of extra lines
        db: Database session
        
    Returns:
        Created test image data
    """
    # Read image data
    image_data = await file.read()
    
    # Create test image record
    test_image = TestImage(
        test_name=test_name,
        image_data=image_data,
        expected_correct=correct_lines,
        expected_missing=missing_lines,
        expected_extra=extra_lines
    )
    
    db.add(test_image)
    db.commit()
    db.refresh(test_image)
    
    return test_image


@app.get("/api/test-images", response_model=List[TestImageResponse])
async def get_test_images(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get all test images.
    
    Args:
        limit: Maximum number of test images to return
        db: Database session
        
    Returns:
        List of test images
    """
    test_images = db.query(TestImage)\
        .order_by(TestImage.created_at.desc())\
        .limit(limit)\
        .all()
    
    return test_images


@app.get("/api/test-images/{test_id}/image")
async def get_test_image_file(
    test_id: int,
    db: Session = Depends(get_db)
):
    """
    Get test image file data.
    
    Args:
        test_id: Test image ID
        db: Database session
        
    Returns:
        Image file
    """
    test_image = db.query(TestImage).filter(TestImage.id == test_id).first()
    
    if not test_image:
        raise HTTPException(status_code=404, detail="Test image not found")
    
    from fastapi.responses import Response
    return Response(content=test_image.image_data, media_type="image/png")


@app.put("/api/test-images/{test_id}", response_model=TestImageResponse)
async def update_test_image(
    test_id: int,
    test_name: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Update test image name.
    
    Args:
        test_id: Test image ID
        test_name: New name for the test image
        db: Database session
        
    Returns:
        Updated test image data
    """
    test_image = db.query(TestImage).filter(TestImage.id == test_id).first()
    
    if not test_image:
        raise HTTPException(status_code=404, detail="Test image not found")
    
    test_image.test_name = test_name
    db.commit()
    db.refresh(test_image)
    
    return TestImageResponse.model_validate(test_image)


@app.delete("/api/test-images/{test_id}")
async def delete_test_image(
    test_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a test image.
    
    Args:
        test_id: Test image ID
        db: Database session
        
    Returns:
        Success message
    """
    test_image = db.query(TestImage).filter(TestImage.id == test_id).first()
    
    if not test_image:
        raise HTTPException(status_code=404, detail="Test image not found")
    
    db.delete(test_image)
    db.commit()
    
    return {"success": True, "message": "Test image deleted successfully"}


@app.post("/api/test-images/run-tests")
async def run_all_tests(
    use_registration: bool = True,
    registration_motion: str = "similarity",
    max_rotation_degrees: float = 30.0,
    position_tolerance: float = 120.0,  # Re-optimized default
    angle_tolerance: float = 50.0,      # Re-optimized default
    length_tolerance: float = 0.8,      # Re-optimized default
    db: Session = Depends(get_db)
):
    """
    Run automated tests on all test images.
    Evaluates each test image and compares expected vs actual results.
    
    Returns:
        Test results with statistics
    """
    import cv2
    import numpy as np
    from io import BytesIO
    
    # Get all test images
    test_images = db.query(TestImage).order_by(TestImage.id).all()
    
    if not test_images:
        return {
            "total_tests": 0,
            "results": [],
            "statistics": {}
        }
    
    # Get reference
    ref_service = ReferenceService(db)
    references = ref_service.list_all_references()
    if not references:
        raise HTTPException(status_code=404, detail="No reference image found")
    
    reference = references[0]
    
    # Evaluate each test image
    results = []
    eval_service = EvaluationService(db, use_registration=use_registration, registration_motion=registration_motion, max_rotation_degrees=max_rotation_degrees)
    
    # Create custom comparator with provided tolerances
    from image_processing import LineComparator
    eval_service.comparator = LineComparator(
        position_tolerance=position_tolerance,
        angle_tolerance=angle_tolerance,
        length_tolerance=length_tolerance
    )
    
    for test_img in test_images:
        try:
            # Load test image from blob
            nparr = np.frombuffer(test_img.image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Evaluate
            evaluation = eval_service.evaluate_test_image(image, reference.id, f"{test_img.id}")
            
            # Calculate differences
            correct_diff = evaluation.correct_lines - test_img.expected_correct
            missing_diff = evaluation.missing_lines - test_img.expected_missing
            extra_diff = evaluation.extra_lines - test_img.expected_extra
            
            # Calculate accuracy based on actual detection quality
            # Get total reference lines from reference
            from image_processing import LineDetector
            line_detector = LineDetector()
            ref_features = line_detector.features_from_json(reference.feature_data)
            total_ref_lines = len(ref_features['lines'])
            
            # Effective correct = actual correct - extra lines penalty (min 0)
            # Extra lines are false positives and should reduce the score
            effective_correct = max(0, evaluation.correct_lines - evaluation.extra_lines)
            
            # Reference Match: How well does the image match the reference?
            # 0% if nothing correct, 100% if all correct and no extras
            detection_score = effective_correct / total_ref_lines if total_ref_lines > 0 else 0.0
            detection_score = max(0.0, min(1.0, detection_score))  # Clamp between 0 and 1
            
            # Test Rating: How well did we predict the actual results? (Expected vs Actual)
            # Perfect test (Expected = Actual) = 100%
            # Maximum possible error per metric is total_ref_lines
            max_error_per_metric = total_ref_lines
            total_max_error = max_error_per_metric * 3  # 3 metrics: correct, missing, extra

            
            total_actual_error = abs(correct_diff) + abs(missing_diff) + abs(extra_diff)
            
            # Test rating: 100% if all diffs are 0, decreases with errors
            prediction_accuracy = 1.0 - (total_actual_error / total_max_error) if total_max_error > 0 else 1.0
            prediction_accuracy = max(0.0, min(1.0, prediction_accuracy))  # Clamp between 0 and 1
            
            results.append({
                "test_id": test_img.id,
                "test_name": test_img.test_name,
                "expected": {
                    "correct": test_img.expected_correct,
                    "missing": test_img.expected_missing,
                    "extra": test_img.expected_extra
                },
                "actual": {
                    "correct": evaluation.correct_lines,
                    "missing": evaluation.missing_lines,
                    "extra": evaluation.extra_lines,
                    "similarity_score": evaluation.similarity_score
                },
                "diff": {
                    "correct": correct_diff,
                    "missing": missing_diff,
                    "extra": extra_diff
                },
                "detection_score": detection_score,      # Reference Match: Image vs Reference
                "prediction_accuracy": prediction_accuracy,  # Test Rating: Expected vs Actual
                "accuracy": detection_score,  # Keep for backward compatibility
                "visualization_path": evaluation.visualization_path,
                "evaluation_id": evaluation.id
            })
            
        except Exception as e:
            results.append({
                "test_id": test_img.id,
                "test_name": test_img.test_name,
                "error": str(e),
                "accuracy": 0.0
            })
    
    # Calculate overall statistics
    successful_tests = [r for r in results if "error" not in r]
    
    if successful_tests:
        avg_detection_score = sum(r["detection_score"] for r in successful_tests) / len(successful_tests)
        avg_prediction_accuracy = sum(r["prediction_accuracy"] for r in successful_tests) / len(successful_tests)
        avg_accuracy = avg_detection_score  # For backward compatibility
        
        avg_correct_diff = sum(abs(r["diff"]["correct"]) for r in successful_tests) / len(successful_tests)
        avg_missing_diff = sum(abs(r["diff"]["missing"]) for r in successful_tests) / len(successful_tests)
        avg_extra_diff = sum(abs(r["diff"]["extra"]) for r in successful_tests) / len(successful_tests)
        
        perfect_detections = sum(1 for r in successful_tests if r["detection_score"] == 1.0)
        perfect_predictions = sum(1 for r in successful_tests if r["prediction_accuracy"] == 1.0)
        perfect_matches = perfect_detections  # For backward compatibility
        
        statistics = {
            "total_tests": len(test_images),
            "successful": len(successful_tests),
            "failed": len(test_images) - len(successful_tests),
            "average_detection_score": avg_detection_score,
            "average_prediction_accuracy": avg_prediction_accuracy,
            "average_accuracy": avg_accuracy,  # Backward compatibility
            "perfect_detections": perfect_detections,
            "perfect_predictions": perfect_predictions,
            "perfect_matches": perfect_matches,  # Backward compatibility
            "average_diff": {
                "correct": avg_correct_diff,
                "missing": avg_missing_diff,
                "extra": avg_extra_diff
            }
        }
    else:
        statistics = {
            "total_tests": len(test_images),
            "successful": 0,
            "failed": len(test_images),
            "average_accuracy": 0.0,
            "perfect_matches": 0
        }
    
    return {
        "total_tests": len(test_images),
        "results": results,
        "statistics": statistics
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.post("/api/reference/manual")
async def create_manual_reference(data: dict, db: Session = Depends(get_db)):
    """Create reference from manually drawn lines."""
    import json
    import numpy as np
    from database import ReferenceImage
    from image_processing.utils import image_to_bytes
    
    # Delete existing
    db.query(ReferenceImage).delete()
    db.commit()
    
    # Create features
    lines = []
    line_angles = []
    line_lengths = []
    
    for line_data in data['lines']:
        x1, y1 = line_data['start']['x'], line_data['start']['y']
        x2, y2 = line_data['end']['x'], line_data['end']['y']
        
        lines.append([x1, y1, x2, y2])
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        line_angles.append(angle)
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        line_lengths.append(length)
    
    features = {
        'num_lines': len(lines),
        'lines': lines,
        'image_shape': [256, 256],
        'line_lengths': line_lengths,
        'line_angles': line_angles,
        'line_counts': data['summary']
    }
    
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    
    ref = ReferenceImage(
        name="manual_reference",
        image_data=image_to_bytes(img),
        processed_image_data=image_to_bytes(img),
        feature_data=json.dumps(features),
        width=256,
        height=256
    )
    
    db.add(ref)
    db.commit()
    
    return {"success": True, "lines_count": len(lines), "summary": data['summary']}



@app.get("/api/reference/status")
async def get_reference_status(db: Session = Depends(get_db)):
    """Check if reference is properly initialized with features."""
    from database import ReferenceImage
    import json
    
    ref = db.query(ReferenceImage).first()
    
    if not ref:
        return {
            "initialized": False,
            "message": "No reference image found"
        }
    
    # Check if features exist and are valid
    if not ref.feature_data:
        return {
            "initialized": False,
            "message": "Reference exists but has no features"
        }
    
    try:
        features = json.loads(ref.feature_data)
        num_lines = features.get('num_lines', 0)
        
        if num_lines < 6:  # Minimum 6 lines for a valid reference
            return {
                "initialized": False,
                "message": f"Only {num_lines} lines defined (minimum 6 required)"
            }
        
        return {
            "initialized": True,
            "message": f"Reference properly initialized with {num_lines} lines",
            "num_lines": num_lines,
            "line_counts": features.get('line_counts', {})
        }
    except:
        return {
            "initialized": False,
            "message": "Invalid feature data"
        }


@app.post("/api/reference/features")
async def add_reference_feature(
    feature: dict,
    db: Session = Depends(get_db)
):
    """Add a new feature line to reference."""
    from database import ReferenceImage
    import json
    import numpy as np
    
    ref = db.query(ReferenceImage).first()
    if not ref:
        raise HTTPException(status_code=404, detail="No reference image found")
    
    # Load existing features
    if ref.feature_data:
        features = json.loads(ref.feature_data)
    else:
        features = {
            'num_lines': 0,
            'lines': [],
            'line_angles': [],
            'line_lengths': [],
            'image_shape': [256, 256],
            'line_counts': {'horizontal': 0, 'vertical': 0, 'diagonal': 0, 'total': 0}
        }
    
    # Add new line
    x1, y1 = feature['start']['x'], feature['start']['y']
    x2, y2 = feature['end']['x'], feature['end']['y']
    
    features['lines'].append([x1, y1, x2, y2])
    
    # Calculate angle
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    features['line_angles'].append(angle)
    
    # Calculate length
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    features['line_lengths'].append(length)
    
    # Update counts
    features['num_lines'] = len(features['lines'])
    
    # Categorize
    norm_angle = abs(angle) if abs(angle) <= 90 else 180 - abs(angle)
    if norm_angle < 15:
        features['line_counts']['horizontal'] += 1
    elif norm_angle > 75:
        features['line_counts']['vertical'] += 1
    else:
        features['line_counts']['diagonal'] += 1
    
    features['line_counts']['total'] = features['num_lines']
    
    # Save
    ref.feature_data = json.dumps(features)
    db.commit()
    
    return {
        "success": True,
        "feature_id": features['num_lines'] - 1,
        "total_features": features['num_lines'],
        "line_counts": features['line_counts']
    }


@app.delete("/api/reference/features/{feature_id}")
async def delete_reference_feature(
    feature_id: int,
    db: Session = Depends(get_db)
):
    """Delete a feature line from reference."""
    from database import ReferenceImage
    import json
    
    ref = db.query(ReferenceImage).first()
    if not ref:
        raise HTTPException(status_code=404, detail="No reference image found")
    
    if not ref.feature_data:
        raise HTTPException(status_code=404, detail="No features found")
    
    features = json.loads(ref.feature_data)
    
    if feature_id < 0 or feature_id >= len(features['lines']):
        raise HTTPException(status_code=404, detail="Feature not found")
    
    # Remove feature
    features['lines'].pop(feature_id)
    features['line_angles'].pop(feature_id)
    features['line_lengths'].pop(feature_id)
    features['num_lines'] = len(features['lines'])
    
    # Recalculate counts
    features['line_counts'] = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    
    for angle in features['line_angles']:
        norm_angle = abs(angle) if abs(angle) <= 90 else 180 - abs(angle)
        if norm_angle < 15:
            features['line_counts']['horizontal'] += 1
        elif norm_angle > 75:
            features['line_counts']['vertical'] += 1
        else:
            features['line_counts']['diagonal'] += 1
    
    features['line_counts']['total'] = features['num_lines']
    
    # Save
    ref.feature_data = json.dumps(features)
    db.commit()
    
    return {
        "success": True,
        "total_features": features['num_lines'],
        "line_counts": features['line_counts']
    }



@app.post("/api/reference/clear")
async def clear_reference_features(db: Session = Depends(get_db)):
    """Clear all features from reference (reset to empty state)."""
    from database import ReferenceImage
    import json
    import numpy as np
    from image_processing.utils import image_to_bytes
    
    ref = db.query(ReferenceImage).first()
    if not ref:
        raise HTTPException(status_code=404, detail="No reference image found")
    
    # Reset features to empty
    features = {
        'num_lines': 0,
        'lines': [],
        'line_angles': [],
        'line_lengths': [],
        'image_shape': [256, 256],
        'line_counts': {'horizontal': 0, 'vertical': 0, 'diagonal': 0, 'total': 0}
    }
    
    ref.feature_data = json.dumps(features)
    db.commit()
    
    return {
        "success": True,
        "message": "All features cleared",
        "features": 0
    }

