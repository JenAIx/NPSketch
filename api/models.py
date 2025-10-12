"""
Pydantic models for request/response validation.

These models define the API contracts for NPSketch endpoints.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class EvaluationResultResponse(BaseModel):
    """Response model for evaluation results."""
    
    id: int
    image_id: int
    reference_id: int
    correct_lines: int
    missing_lines: int
    extra_lines: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    visualization_path: Optional[str] = None
    evaluated_at: datetime
    
    # User evaluation fields for AI training
    user_evaluated: bool = False
    evaluated_correct: Optional[int] = None
    evaluated_missing: Optional[int] = None
    evaluated_extra: Optional[int] = None
    evaluated_at_timestamp: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class EvaluationUpdateRequest(BaseModel):
    """Request model for updating user evaluation."""
    
    evaluated_correct: int = Field(..., ge=0, description="User-evaluated correct lines count")
    evaluated_missing: int = Field(..., ge=0, description="User-evaluated missing lines count")
    evaluated_extra: int = Field(..., ge=0, description="User-evaluated extra lines count")


class UploadedImageResponse(BaseModel):
    """Response model for uploaded images."""
    
    id: int
    filename: str
    uploader: Optional[str] = None
    uploaded_at: datetime
    
    class Config:
        from_attributes = True


class ReferenceImageResponse(BaseModel):
    """Response model for reference images."""
    
    id: int
    name: str
    width: int
    height: int
    created_at: datetime
    feature_data: Optional[str] = None  # Include feature_data JSON string
    
    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Response model for image upload."""
    
    success: bool
    message: str
    image_id: Optional[int] = None
    evaluation: Optional[EvaluationResultResponse] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    database_initialized: bool
    reference_images_count: int


class FeatureStats(BaseModel):
    """Statistics about extracted features."""
    
    num_lines: int
    num_contours: int
    image_resolution: tuple[int, int]


class TestImageResponse(BaseModel):
    """Response model for test images."""
    
    id: int
    test_name: str
    expected_correct: int
    expected_missing: int
    expected_extra: int
    created_at: datetime
    
    class Config:
        from_attributes = True


