"""
AI Training Models Router

Endpoints for model management (list, metadata, test, delete).
"""

from fastapi import APIRouter, Depends, HTTPException, Body, UploadFile, File, Form
from sqlalchemy.orm import Session
from database import get_db, TrainingDataImage
import json
import sys

sys.path.insert(0, '/app')
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/ai-training", tags=["ai_training_models"])


def prepare_test_dataloaders(
    feature: str,
    metadata: dict,
    normalizer,
    is_classification: bool,
    num_classes: int,
    db: Session
):
    """
    Prepare test dataloaders using validation data from metadata.
    
    Uses val_image_ids from model metadata (original images, no augmentation/synthetic).
    Fast and avoids timeout issues.
    
    Args:
        feature: Target feature name
        metadata: Model metadata
        normalizer: Normalizer instance (if used)
        is_classification: True for classification mode
        num_classes: Number of classes (for classification)
        db: Database session
    
    Returns:
        (test_loader, stats)
    """
    # Get validation IDs from metadata
    val_image_ids = metadata.get('val_image_ids', [])
    
    # Filter to only integer IDs (exclude synthetic images like "synthetic_bad_0")
    val_image_ids_db = [id for id in val_image_ids if isinstance(id, int)]
    
    n_synthetic_val = len(val_image_ids) - len(val_image_ids_db)
    
    if n_synthetic_val > 0:
        logger.info(f"Filtered out {n_synthetic_val} synthetic validation images")
    
    logger.info(f"Using {len(val_image_ids_db)} validation images from metadata (original, no augmentation)")
    
    if len(val_image_ids_db) == 0:
        raise ValueError(f"No validation images in metadata")
    
    # Load validation images from DB
    val_images = db.query(TrainingDataImage).filter(
        TrainingDataImage.id.in_(val_image_ids_db),
        TrainingDataImage.features_data.isnot(None)
    ).all()
    
    if len(val_images) == 0:
        raise ValueError(f"Validation images not found in database (IDs may have been deleted)")
    
    test_images = []
    for img in val_images:
        test_images.append({
            'id': img.id,
            'processed_image_data': img.processed_image_data,
            'features_data': img.features_data
        })
    
    logger.info(f"Loaded {len(test_images)} validation images for testing")
    
    # Create test dataloader (no train/val split, just test)
    from ai_training.dataset import DrawingDataset
    from torch.utils.data import DataLoader
    
    test_dataset = DrawingDataset(
        test_images,
        feature,
        transform=None,
        normalizer=normalizer,
        is_classification=is_classification,
        num_classes=num_classes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False
    )
    
    stats = {
        'test_samples': len(test_dataset),
        'test_batches': len(test_loader)
    }
    
    return test_loader, stats


@router.get("/models")
async def list_models():
    """List all saved models with metadata."""
    import os
    from pathlib import Path
    
    models_dir = Path("/app/data/models")
    models_dir.mkdir(parents=True, exist_ok=True)  # Create lazily when endpoint is called
    
    models = []
    for model_file in models_dir.glob("*.pth"):
        stat = model_file.stat()
        
        # Extract info from filename: model_{feature}_{timestamp}.pth
        parts = model_file.stem.split('_')
        if len(parts) >= 3:
            feature = '_'.join(parts[1:-2]) if len(parts) > 3 else parts[1]
            timestamp = '_'.join(parts[-2:])
        else:
            feature = "unknown"
            timestamp = "unknown"
        
        # Check if metadata file exists
        metadata_file = models_dir / f"{model_file.stem}_metadata.json"
        has_metadata = metadata_file.exists()
        
        model_info = {
            'filename': model_file.name,
            'path': str(model_file),
            'feature': feature,
            'timestamp': timestamp,
            'size_mb': stat.st_size / (1024 * 1024),
            'created_at': stat.st_mtime,
            'has_metadata': has_metadata
        }
        
        models.append(model_info)
    
    # Sort by creation time (newest first)
    models.sort(key=lambda x: x['created_at'], reverse=True)
    
    return {
        'models': models,
        'total': len(models)
    }


@router.get("/models/{model_filename}/metadata")
async def get_model_metadata(model_filename: str):
    """Get metadata for a specific model."""
    from pathlib import Path
    import json
    
    # Remove .pth extension if present
    model_stem = model_filename.replace('.pth', '')
    metadata_file = Path("/app/data/models") / f"{model_stem}_metadata.json"
    
    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Metadata not found")
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/test")
async def test_model(
    request: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Test a saved model on current test data.
    Supports both regression and classification models.
    """
    import os
    from pathlib import Path
    import torch
    from ai_training.trainer import CNNTrainer
    from ai_training.dataset import create_dataloaders
    
    try:
        model_filename = request.get('model_filename')
        if not model_filename:
            raise HTTPException(status_code=400, detail="model_filename is required")
        
        model_path = Path("/app/data/models") / model_filename
        metadata_path = Path("/app/data/models") / f"{model_path.stem}_metadata.json"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Initialize variables at the start
        training_mode = "regression"  # Default
        num_outputs = 1
        normalizer = None
        is_classification = False
        num_classes = None
        feature = None
        metadata = None
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract configuration from metadata
            feature = metadata['target_feature']
            num_outputs = metadata['model']['output_neurons']
            
            # Prefer training_mode from metadata, fallback to detection
            training_mode = metadata.get('training_mode')
            if training_mode:
                # Use training_mode from metadata
                is_classification = (training_mode == "classification")
                if is_classification:
                    num_classes = metadata.get('num_classes')
                    if not num_classes:
                        # Fallback: extract from feature name
                        if feature.startswith('Custom_Class_'):
                            num_classes = int(feature.replace('Custom_Class_', ''))
                        else:
                            # Use output_neurons as fallback
                            num_classes = num_outputs
                    logger.info(f"Testing CLASSIFICATION model: {num_classes} classes (from metadata)")
                else:
                    # Regression mode
                    # Get normalizer if used
                    if metadata.get('normalization', {}).get('enabled', False):
                        from ai_training.normalization import TargetNormalizer
                        normalizer = TargetNormalizer.from_config(metadata['normalization'])
                    logger.info(f"Testing REGRESSION model (from metadata)")
            else:
                # Fallback: detect from feature name
                is_classification = feature.startswith('Custom_Class_')
                if is_classification:
                    training_mode = "classification"
                    num_classes = int(feature.replace('Custom_Class_', ''))
                    logger.info(f"Testing CLASSIFICATION model: {num_classes} classes (detected from feature)")
                else:
                    training_mode = "regression"
                    # Get normalizer if used
                    if metadata.get('normalization', {}).get('enabled', False):
                        from ai_training.normalization import TargetNormalizer
                        normalizer = TargetNormalizer.from_config(metadata['normalization'])
                    logger.info(f"Testing REGRESSION model (detected from feature)")
        else:
            # Fallback: extract from filename
            parts = model_filename.split('_')
            
            # Validate filename format: expect at least model_{feature}_...
            if len(parts) < 2:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid model filename format: '{model_filename}'. "
                           f"Expected format: model_{{feature}}_{{timestamp}}.pth"
                )
            
            # Extract feature name (everything between 'model_' and last 2 parts which are timestamp)
            feature = '_'.join(parts[1:-2]) if len(parts) > 3 else parts[1]
            
            is_classification = feature.startswith('Custom_Class_')
            if is_classification:
                training_mode = "classification"
                num_classes = int(feature.replace('Custom_Class_', ''))
                num_outputs = num_classes
                logger.info(f"Testing CLASSIFICATION model: {num_classes} classes (from filename)")
            else:
                training_mode = "regression"
                logger.info(f"Testing REGRESSION model (from filename)")
        
        if not feature:
            raise HTTPException(status_code=400, detail="Could not determine target feature from model")
        
        # Prepare test dataloaders using holdout data
        try:
            test_loader, test_stats = prepare_test_dataloaders(
                feature=feature,
                metadata=metadata if metadata else {},
                normalizer=normalizer,
                is_classification=is_classification,
                num_classes=num_classes,
                db=db
            )
        except Exception as e:
            logger.error(f"Failed to prepare test data: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to prepare test data: {str(e)}")
        
        # Validate test dataloader
        if len(test_loader) == 0:
            raise HTTPException(status_code=400, detail=f"No test samples found for feature '{feature}'")
        
        # Create trainer and load model
        trainer = CNNTrainer(
            num_outputs=num_outputs,
            normalizer=normalizer,
            training_mode=training_mode
        )
        trainer.load_model(str(model_path))
        
        # Evaluate on test set
        try:
            test_metrics = trainer.evaluate_metrics(test_loader)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to evaluate model: {str(e)}")
        
        return {
            'success': True,
            'model': model_filename,
            'target_feature': feature,
            'training_mode': training_mode,
            'num_outputs': num_outputs,
            'val_metrics': test_metrics,  # Use 'val_metrics' key for frontend compatibility
            'test_samples': test_stats['test_samples'],
            'test_type': 'validation_split'  # Using validation split from training
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@router.post("/models/predict-single")
async def predict_single_image(
    file: UploadFile = File(...),
    model_filename: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Predict score/class for a single uploaded image using a trained model.
    
    Returns:
        - For regression: predicted score
        - For classification: predicted class and probabilities with custom names
    """
    import torch
    import numpy as np
    from pathlib import Path
    from PIL import Image
    import io
    
    from ai_training.trainer import CNNTrainer
    from ai_training.model import DrawingClassifier
    
    try:
        model_path = Path("/app/data/models") / model_filename
        metadata_path = Path("/app/data/models") / f"{model_path.stem}_metadata.json"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Load metadata
        training_mode = "regression"
        num_outputs = 1
        class_names = {}
        class_boundaries = []
        target_feature = "Total_Score"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            target_feature = metadata.get('target_feature', 'Total_Score')
            num_outputs = metadata.get('model', {}).get('output_neurons', 1)
            training_mode = metadata.get('training_mode', 'regression')
            
            # For classification, extract class names from database
            if target_feature.startswith('Custom_Class_'):
                training_mode = "classification"
                num_classes_str = target_feature.replace('Custom_Class_', '')
                num_classes = int(num_classes_str)
                num_outputs = num_classes
                
                # Query DB to get class names and boundaries
                for img in db.query(TrainingDataImage).filter(
                    TrainingDataImage.features_data.isnot(None)
                ).limit(500).all():
                    try:
                        features = json.loads(img.features_data)
                        if 'Custom_Class' in features and num_classes_str in features.get('Custom_Class', {}):
                            cc = features['Custom_Class'][num_classes_str]
                            label = cc['label']
                            class_names[label] = cc.get('name_custom', f'Class_{label}')
                            if not class_boundaries and 'boundaries' in cc:
                                class_boundaries = cc['boundaries']
                            # Stop when we have all classes
                            if len(class_names) >= num_classes:
                                break
                    except:
                        continue
        
        # Load and preprocess image
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB first (needed for line normalization)
        if pil_img.mode != 'RGB':
            if pil_img.mode == 'RGBA':
                background = Image.new('RGB', pil_img.size, (255, 255, 255))
                background.paste(pil_img, mask=pil_img.split()[3])
                pil_img = background
            else:
                pil_img = pil_img.convert('RGB')
        
        # Resize to expected dimensions (568x274)
        pil_img = pil_img.resize((568, 274), Image.Resampling.LANCZOS)
        
        # CRITICAL: Normalize line thickness to 2.00px (same as training data)
        # This ensures consistency with how training images were processed
        from line_normalizer import normalize_line_thickness
        img_array_rgb = np.array(pil_img)
        img_array_rgb = normalize_line_thickness(img_array_rgb, target_thickness=2.0)
        
        # Convert to grayscale after normalization
        pil_img = Image.fromarray(img_array_rgb).convert('L')
        
        # Convert to tensor
        img_array = np.array(pil_img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Load model
        model = DrawingClassifier(num_outputs=num_outputs, pretrained=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            
            if training_mode == "classification":
                # Get probabilities and predicted class
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                
                # Get class probabilities with custom names
                class_probs = {}
                class_info = []
                for i, prob in enumerate(probabilities.tolist()):
                    custom_name = class_names.get(i, f"Class_{i}")
                    class_probs[custom_name] = round(prob * 100, 1)
                    
                    # Build range info from boundaries
                    range_str = ""
                    if class_boundaries and len(class_boundaries) > i + 1:
                        range_str = f"[{class_boundaries[i]}-{class_boundaries[i+1]}]"
                    
                    class_info.append({
                        'label': i,
                        'name': custom_name,
                        'probability': round(prob * 100, 1),
                        'range': range_str
                    })
                
                predicted_name = class_names.get(predicted_class, f"Class_{predicted_class}")
                
                return {
                    'success': True,
                    'model': model_filename,
                    'target_feature': target_feature,
                    'training_mode': 'classification',
                    'num_classes': num_outputs,
                    'class_names': class_names,
                    'boundaries': class_boundaries,
                    'prediction': {
                        'class': predicted_class,
                        'class_name': predicted_name,
                        'confidence': round(probabilities[predicted_class].item() * 100, 1),
                        'probabilities': class_probs,
                        'class_details': class_info
                    }
                }
            else:
                # Regression - get predicted score
                raw_value = output[0][0].item()
                predicted_value = raw_value
                
                # Denormalize if normalization was used
                norm_config = metadata.get('normalization', {}) if metadata_path.exists() else {}
                
                # Check if normalization was applied (has method or min/max values)
                if norm_config.get('method') == 'min_max' or (norm_config.get('min_value') is not None and norm_config.get('max_value') is not None):
                    min_val = norm_config.get('min_value', 0)
                    max_val = norm_config.get('max_value', 60)
                    
                    # Denormalize: value * (max - min) + min
                    predicted_value = raw_value * (max_val - min_val) + min_val
                    
                    # Clamp to valid range
                    predicted_value = max(min_val, min(max_val, predicted_value))
                
                return {
                    'success': True,
                    'model': model_filename,
                    'target_feature': target_feature,
                    'training_mode': 'regression',
                    'normalization': {
                        'applied': bool(norm_config.get('method')),
                        'min': norm_config.get('min_value'),
                        'max': norm_config.get('max_value')
                    },
                    'prediction': {
                        'value': round(predicted_value, 2),
                        'raw_output': round(raw_value, 4)
                    }
                }
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_filename}")
async def delete_model(model_filename: str):
    """Delete a saved model and its metadata."""
    import os
    from pathlib import Path
    
    model_path = Path("/app/data/models") / model_filename
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Delete model file
        os.unlink(model_path)
        
        # Delete metadata file if exists
        metadata_path = Path("/app/data/models") / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            os.unlink(metadata_path)
            return {"success": True, "message": f"Model and metadata deleted: {model_filename}"}
        
        return {"success": True, "message": f"Model deleted: {model_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/models/{model_filename}/rename")
async def rename_model(model_filename: str, request: dict = Body(...)):
    """Rename a model by updating its display label in metadata."""
    from pathlib import Path
    import json
    
    new_label = request.get('new_label', '').strip()
    if not new_label:
        raise HTTPException(status_code=400, detail="new_label is required")
    
    model_path = Path("/app/data/models") / model_filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    metadata_path = Path("/app/data/models") / f"{model_path.stem}_metadata.json"
    
    try:
        # Load existing metadata or create new one
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Create minimal metadata if it doesn't exist
            metadata = {}
        
        # Update the display label
        metadata['display_label'] = new_label
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "success": True,
            "message": f"Model renamed to: {new_label}",
            "new_label": new_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-orphaned-metadata")
async def cleanup_orphaned_metadata():
    """Delete orphaned metadata files (metadata without corresponding .pth file)."""
    from pathlib import Path
    import os
    
    models_dir = Path("/app/data/models")
    cleaned = 0
    
    try:
        # Find all metadata files
        for metadata_file in models_dir.glob("*_metadata.json"):
            # Check if corresponding .pth file exists
            model_stem = metadata_file.stem.replace('_metadata', '')
            model_file = models_dir / f"{model_stem}.pth"
            
            if not model_file.exists():
                # Orphaned metadata - delete it
                os.unlink(metadata_file)
                cleaned += 1
        
        return {
            "success": True,
            "cleaned": cleaned,
            "message": f"Deleted {cleaned} orphaned metadata file(s)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

