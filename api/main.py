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

from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os

from database import init_database, get_db
from models import HealthResponse

# Import routers
from routers import (
    admin_router,
    upload_router,
    training_data_router,
    ai_training_base_router,
    ai_training_classification_router,
    ai_training_models_router,
    evaluator_router
)

# Single source of truth for the app version (also returned by /api/health)
APP_VERSION = "2.3.0"

# Initialize FastAPI app
app = FastAPI(
    title="NPSketch API",
    description="CNN-based scoring of hand-drawn neuropsychological figures",
    version=APP_VERSION
)

# Enable CORS for external access (e.g., from mars.biomag.uni-jena.de)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development/internal use
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for visualizations (ensure directory exists)
VIS_DIR = "/app/data/visualizations"
os.makedirs(VIS_DIR, exist_ok=True)
app.mount("/api/visualizations", StaticFiles(directory=VIS_DIR), name="visualizations")

# Include routers
app.include_router(admin_router)
app.include_router(upload_router)
app.include_router(training_data_router)

# AI Training routers (split for better organization)
app.include_router(ai_training_base_router)
app.include_router(ai_training_classification_router)
app.include_router(ai_training_models_router)
app.include_router(evaluator_router)


@app.on_event("startup")
async def startup_event():
    """Initialize the database on startup."""
    init_database()
    print("✓ Database initialized")


@app.get("/api/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        database_initialized=True,
        reference_images_count=0,
        version=APP_VERSION
    )


@app.get("/api/reference-image")
async def reference_image():
    """Serve the canonical OCS-Plus reference figure (templates/ is not on the nginx root)."""
    from fastapi.responses import FileResponse
    path = "/app/templates/reference_image.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Reference image not found")
    return FileResponse(path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)