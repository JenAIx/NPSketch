"""
NPSketch API Routers

This package contains modular FastAPI routers for different functional areas:
- admin: Administrative endpoints (migrations, etc.)
- upload: Image upload and processing endpoints  
- evaluations: Evaluation management endpoints
- references: Reference image management endpoints
- test_images: Test image management and testing endpoints
- training_data: AI training data extraction endpoints
"""

from .admin import router as admin_router
from .upload import router as upload_router  
from .evaluations import router as evaluations_router
from .references import router as references_router
from .test_images import router as test_images_router
from .training_data import router as training_data_router

__all__ = [
    "admin_router",
    "upload_router", 
    "evaluations_router",
    "references_router",
    "test_images_router",
    "training_data_router"
]
