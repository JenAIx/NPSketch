"""
Evaluations Router - Evaluation management endpoints for NPSketch API

Contains endpoints for:
- Getting recent evaluations
- Getting specific evaluation details
- Deleting evaluations (with cascading cleanup)
- Updating evaluations with user feedback
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from database import get_db, EvaluationResult, ExtractedFeature, UploadedImage
from models import EvaluationResultResponse, EvaluationUpdateRequest
from services import EvaluationService
import os

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


@router.get("/recent", response_model=List[EvaluationResultResponse])
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


@router.get("/{eval_id}", response_model=EvaluationResultResponse)
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


@router.delete("/{eval_id}")
async def delete_evaluation(
    eval_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete an evaluation result and associated uploaded image.
    
    Args:
        eval_id: Evaluation ID to delete
        db: Database session
        
    Returns:
        Success message
    """
    # Find evaluation
    evaluation = db.query(EvaluationResult).filter(EvaluationResult.id == eval_id).first()
    
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    image_id = evaluation.image_id
    
    # Delete visualization file if it exists
    if evaluation.visualization_path:
        viz_path = evaluation.visualization_path.replace('/api/visualizations/', '/app/data/visualizations/')
        if os.path.exists(viz_path):
            try:
                os.remove(viz_path)
            except Exception as e:
                print(f"Warning: Could not delete visualization file: {e}")
    
    # Delete evaluation
    db.delete(evaluation)
    
    # Check if this image has any other evaluations
    other_evaluations = db.query(EvaluationResult).filter(
        EvaluationResult.image_id == image_id,
        EvaluationResult.id != eval_id
    ).count()
    
    # If no other evaluations exist for this image, delete the image and features
    if other_evaluations == 0:
        # Delete extracted features
        db.query(ExtractedFeature).filter(
            ExtractedFeature.image_id == image_id
        ).delete()
        
        # Delete uploaded image
        uploaded_image = db.query(UploadedImage).filter(
            UploadedImage.id == image_id
        ).first()
        
        if uploaded_image:
            db.delete(uploaded_image)
    
    db.commit()
    
    return {
        "message": "Evaluation deleted successfully", 
        "id": eval_id,
        "image_deleted": other_evaluations == 0
    }


@router.put("/{eval_id}/evaluate", response_model=EvaluationResultResponse)
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
