"""
NPSketch API - Automated Line Detection for Hand-Drawn Images

This FastAPI application provides endpoints for:
- Uploading and comparing hand-drawn images to reference templates
- Extracting line features using OpenCV
- Evaluating drawing accuracy
- Visualizing results

Author: Stefan Brodoehl
Date: October 2025
Version: 1.0
"""

from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import init_database, get_db, ReferenceImage
from models import HealthResponse
from services import ReferenceService

# Import routers
from routers import (
    admin_router,
    upload_router,
    evaluations_router,
    references_router,
    test_images_router,
    training_data_router,
    ai_training_router
)

# Initialize FastAPI app
app = FastAPI(
    title="NPSketch API",
    description="Automated line detection and comparison for hand-drawn images",
    version="1.0.0"
)

# Mount static files for visualizations
app.mount("/api/visualizations", StaticFiles(directory="/app/data/visualizations"), name="visualizations")

# Include routers
app.include_router(admin_router)
app.include_router(upload_router)
app.include_router(evaluations_router)
app.include_router(references_router)
app.include_router(test_images_router)
app.include_router(training_data_router)
app.include_router(ai_training_router)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)