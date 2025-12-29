"""
Pydantic Models for NPSketch AI Training

Provides type-safe configuration and request/response models.
"""

from pydantic import BaseModel, Field, validator, field_validator, model_validator
from typing import Optional, Literal, Tuple, List, Dict, Any
from datetime import datetime


# ============================================================
# Training Configuration Models
# ============================================================

class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    target_feature: str = Field(..., min_length=1, description="Target feature to predict")
    train_split: float = Field(0.8, ge=0.1, le=0.9, description="Train/validation split ratio")
    num_epochs: int = Field(50, ge=1, le=1000, description="Number of training epochs")
    learning_rate: float = Field(0.001, gt=0, le=0.1, description="Learning rate")
    batch_size: int = Field(8, ge=1, le=128, description="Batch size")
    use_augmentation: bool = Field(True, description="Enable data augmentation")
    use_normalization: bool = Field(True, description="Enable target normalization")
    add_synthetic_bad_images: bool = Field(False, description="Add synthetic bad images")
    synthetic_n_samples: int = Field(50, ge=0, le=500, description="Number of synthetic images")
    
    @field_validator('synthetic_n_samples')
    @classmethod
    def validate_synthetic(cls, v, info):
        """Validate synthetic samples count."""
        # Access other fields via info.data
        if info.data.get('add_synthetic_bad_images') and v == 0:
            raise ValueError('synthetic_n_samples must be > 0 if add_synthetic_bad_images is enabled')
        return v
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate overall configuration consistency."""
        # If not using augmentation, synthetic images should also be disabled
        # (because they go through augmentation pipeline)
        if not self.use_augmentation and self.add_synthetic_bad_images:
            raise ValueError('Cannot add synthetic images without augmentation enabled')
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "target_feature": "Total_Score",
                "train_split": 0.8,
                "num_epochs": 50,
                "learning_rate": 0.001,
                "batch_size": 8,
                "use_augmentation": True,
                "use_normalization": True,
                "add_synthetic_bad_images": True,
                "synthetic_n_samples": 50
            }
        }


class AugmentationConfig(BaseModel):
    """Configuration for data augmentation."""
    
    rotation_range: Tuple[float, float] = Field((-3, 3), description="Rotation range in degrees")
    translation_range: Tuple[int, int] = Field((-10, 10), description="Translation range in pixels")
    scale_range: Tuple[float, float] = Field((0.95, 1.05), description="Scale range")
    num_augmentations: int = Field(5, ge=0, le=20, description="Number of augmentations per image")
    use_warping: bool = Field(True, description="Enable local warping")
    warping_displacement: int = Field(15, ge=5, le=30, description="Warping displacement in pixels")
    warping_control_points: int = Field(9, ge=4, le=16, description="Number of warping control points")
    warping_ratio: float = Field(0.4, ge=0, le=1, description="Ratio of augmentations using warping")
    binarization_threshold: int = Field(175, ge=0, le=255, description="Binarization threshold")
    line_thickness: int = Field(2, ge=1, le=5, description="Target line thickness in pixels")
    
    @field_validator('rotation_range', 'translation_range', 'scale_range')
    @classmethod
    def validate_range(cls, v):
        """Validate that range is (min, max) with min < max."""
        if len(v) != 2:
            raise ValueError('Range must be a tuple of (min, max)')
        if v[0] >= v[1]:
            raise ValueError('Range minimum must be less than maximum')
        return v


class SyntheticImageConfig(BaseModel):
    """Configuration for synthetic bad image generation."""
    
    n_samples: int = Field(50, ge=0, le=500, description="Number of synthetic images to generate")
    complexity_levels: int = Field(5, ge=1, le=10, description="Number of complexity levels")
    score_threshold: float = Field(20.0, ge=0, le=60, description="Maximum score for 'bad' images")
    require_real_bad_lines: bool = Field(False, description="Fail if no real bad lines found")
    image_size: Tuple[int, int] = Field((568, 274), description="Image size (width, height)")
    min_lines: int = Field(5, ge=1, le=50, description="Minimum lines per image")
    max_lines: int = Field(20, ge=1, le=50, description="Maximum lines per image")
    random_seed: int = Field(42, description="Random seed for reproducibility")
    cache_enabled: bool = Field(True, description="Enable caching of line pools")
    
    @field_validator('max_lines')
    @classmethod
    def validate_line_count(cls, v, info):
        """Validate that max_lines > min_lines."""
        min_lines = info.data.get('min_lines')
        if min_lines is not None and v <= min_lines:
            raise ValueError('max_lines must be greater than min_lines')
        return v


# ============================================================
# Model Metadata Models
# ============================================================

class ModelArchitectureInfo(BaseModel):
    """Model architecture information."""
    
    architecture: str = "resnet18"
    backbone: str = "ImageNet"
    input_size: str = "568×274×1"
    total_parameters: int
    trainable_parameters: int
    output_neurons: int
    head_layers: Optional[List[Dict[str, Any]]] = None


class SyntheticImageMetadata(BaseModel):
    """Metadata about synthetic images used in training."""
    
    enabled: bool
    n_samples: int = 0
    n_generated: int = 0
    complexity_levels: int = 0
    score_threshold: float = 20.0


class DatasetMetadata(BaseModel):
    """Metadata about training dataset."""
    
    total_samples: int
    train_samples: int
    val_samples: int
    train_batches: int
    val_batches: int
    split_strategy: str
    n_bins: Optional[int] = None
    train_target_range: List[float] = []
    val_target_range: List[float] = []


class ModelMetadata(BaseModel):
    """Complete model metadata."""
    
    version: str = "1.1.0"
    target_feature: str
    training_mode: Literal["regression", "classification"]
    num_classes: Optional[int] = None
    training_config: Dict[str, Any]
    normalization: Dict[str, Any]
    augmentation: Dict[str, Any]
    synthetic_bad_images: SyntheticImageMetadata
    dataset: DatasetMetadata
    split_quality: Dict[str, Any]
    model: ModelArchitectureInfo
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    train_image_ids: List[Any]
    val_image_ids: List[Any]
    all_image_ids: List[Any]
    trained_at: datetime
    model_filename: Optional[str] = None
    saved_at: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================
# API Request/Response Models
# ============================================================

class TrainingStartRequest(TrainingConfig):
    """Request model for starting training."""
    pass


class TrainingStartResponse(BaseModel):
    """Response model for training start."""
    
    success: bool
    message: str
    config: Dict[str, Any]


class TrainingStatus(BaseModel):
    """Training status information."""
    
    status: Literal["idle", "training", "completed", "error"]
    progress: Dict[str, Any] = {}
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about a saved model."""
    
    filename: str
    feature: str
    size_mb: float
    created_at: float
    has_metadata: bool


class ModelListResponse(BaseModel):
    """Response model for model listing."""
    
    models: List[ModelInfo]
    total_count: int


# ============================================================
# Validation Models
# ============================================================

class FeatureValidation(BaseModel):
    """Validation result for a feature."""
    
    feature_name: str
    exists: bool
    count: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    is_classification: bool
    num_classes: Optional[int] = None


class DatasetValidation(BaseModel):
    """Validation result for dataset."""
    
    ready: bool
    status: Literal["ready", "not_ready", "warning"]
    issues: List[str] = []
    warnings: List[str] = []
    recommendations: List[str] = []
    total_images: int
    labeled_images: int
    features: List[FeatureValidation] = []


# ============================================================
# Helper Functions
# ============================================================

def validate_training_config(config: Dict[str, Any]) -> TrainingConfig:
    """
    Validate training configuration dictionary.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Validated TrainingConfig
    
    Raises:
        ValidationError: If validation fails
    """
    return TrainingConfig(**config)


def create_synthetic_config_from_dict(config: Dict[str, Any]) -> SyntheticImageConfig:
    """
    Create SyntheticImageConfig from dictionary.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        SyntheticImageConfig instance
    """
    return SyntheticImageConfig(**config)

