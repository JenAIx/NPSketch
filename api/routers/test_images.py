"""
Test Images Router - Test image management and testing endpoints for NPSketch API

Contains endpoints for:
- Creating test images with manual scoring
- Getting test images and test image files
- Updating and deleting test images
- Running automated tests on all test images
"""

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import List
from database import get_db, TestImage
from models import TestImageResponse
from services import ReferenceService, EvaluationService
import cv2
import numpy as np

router = APIRouter(prefix="/api/test-images", tags=["test_images"])


@router.post("", response_model=TestImageResponse)
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


@router.get("", response_model=List[TestImageResponse])
async def get_test_images(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get all test images."""
    test_images = db.query(TestImage)\
        .order_by(TestImage.created_at.desc())\
        .limit(limit)\
        .all()
    
    return test_images


@router.get("/{test_id}/image")
async def get_test_image_file(
    test_id: int,
    db: Session = Depends(get_db)
):
    """Get test image file data."""
    test_image = db.query(TestImage).filter(TestImage.id == test_id).first()
    
    if not test_image:
        raise HTTPException(status_code=404, detail="Test image not found")
    
    return Response(content=test_image.image_data, media_type="image/png")


@router.put("/{test_id}", response_model=TestImageResponse)
async def update_test_image(
    test_id: int,
    test_name: str = Form(...),
    db: Session = Depends(get_db)
):
    """Update test image name."""
    test_image = db.query(TestImage).filter(TestImage.id == test_id).first()
    
    if not test_image:
        raise HTTPException(status_code=404, detail="Test image not found")
    
    test_image.test_name = test_name
    db.commit()
    db.refresh(test_image)
    
    return TestImageResponse.model_validate(test_image)


@router.delete("/{test_id}")
async def delete_test_image(
    test_id: int,
    db: Session = Depends(get_db)
):
    """Delete a test image."""
    test_image = db.query(TestImage).filter(TestImage.id == test_id).first()
    
    if not test_image:
        raise HTTPException(status_code=404, detail="Test image not found")
    
    db.delete(test_image)
    db.commit()
    
    return {"success": True, "message": "Test image deleted successfully"}


@router.post("/run-tests")
async def run_all_tests(
    use_registration: bool = True,
    registration_motion: str = "similarity",
    max_rotation_degrees: float = 30.0,
    position_tolerance: float = 120.0,
    angle_tolerance: float = 50.0,
    length_tolerance: float = 0.8,
    db: Session = Depends(get_db)
):
    """Run automated tests on all test images."""
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
            effective_correct = max(0, evaluation.correct_lines - evaluation.extra_lines)
            
            # Reference Match: How well does the image match the reference?
            detection_score = effective_correct / total_ref_lines if total_ref_lines > 0 else 0.0
            detection_score = max(0.0, min(1.0, detection_score))
            
            # Test Rating: How well did we predict the actual results?
            max_error_per_metric = total_ref_lines
            total_max_error = max_error_per_metric * 3  # 3 metrics
            
            total_actual_error = abs(correct_diff) + abs(missing_diff) + abs(extra_diff)
            
            # Test rating: 100% if all diffs are 0, decreases with errors
            prediction_accuracy = 1.0 - (total_actual_error / total_max_error) if total_max_error > 0 else 1.0
            prediction_accuracy = max(0.0, min(1.0, prediction_accuracy))
            
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
                "detection_score": detection_score,
                "prediction_accuracy": prediction_accuracy,
                "accuracy": detection_score,  # Backward compatibility
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
        avg_accuracy = avg_detection_score
        
        avg_correct_diff = sum(abs(r["diff"]["correct"]) for r in successful_tests) / len(successful_tests)
        avg_missing_diff = sum(abs(r["diff"]["missing"]) for r in successful_tests) / len(successful_tests)
        avg_extra_diff = sum(abs(r["diff"]["extra"]) for r in successful_tests) / len(successful_tests)
        
        perfect_detections = sum(1 for r in successful_tests if r["detection_score"] == 1.0)
        perfect_predictions = sum(1 for r in successful_tests if r["prediction_accuracy"] == 1.0)
        
        statistics = {
            "total_tests": len(test_images),
            "successful": len(successful_tests),
            "failed": len(test_images) - len(successful_tests),
            "average_detection_score": avg_detection_score,
            "average_prediction_accuracy": avg_prediction_accuracy,
            "average_accuracy": avg_accuracy,
            "perfect_detections": perfect_detections,
            "perfect_predictions": perfect_predictions,
            "perfect_matches": perfect_detections,
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
