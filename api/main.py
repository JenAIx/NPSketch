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
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List
import os

from database import init_database, get_db, ReferenceImage, TestImage
from models import (
    UploadResponse,
    HealthResponse,
    EvaluationResultResponse,
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
