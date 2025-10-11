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

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List
import os

from database import init_database, get_db, ReferenceImage
from models import (
    UploadResponse,
    HealthResponse,
    EvaluationResultResponse,
    ReferenceImageResponse
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
