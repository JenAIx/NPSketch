"""
AI Training Models Router

Endpoints for model management (list, metadata, test, delete).
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import get_db, TrainingDataImage
import json
import sys

sys.path.insert(0, '/app')

router = APIRouter(prefix="/api/ai-training", tags=["ai_training_models"])


@router.get("/models")
async def list_models():
    """List all saved models with metadata."""
    import os
    from pathlib import Path
    
    models_dir = Path("/app/data/models")
    models_dir.mkdir(exist_ok=True)
    
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
        
        # Load metadata to get training configuration
        training_mode = "regression"  # Default
        num_outputs = 1
        normalizer = None
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract configuration from metadata
            feature = metadata['target_feature']
            num_outputs = metadata['model']['output_neurons']
            
            # Detect if classification
            is_classification = feature.startswith('Custom_Class_')
            if is_classification:
                training_mode = "classification"
                num_classes = int(feature.replace('Custom_Class_', ''))
                print(f"Testing CLASSIFICATION model: {num_classes} classes")
            else:
                training_mode = "regression"
                # Get normalizer if used
                if metadata.get('normalization', {}).get('enabled', False):
                    from ai_training.normalization import TargetNormalizer
                    normalizer = TargetNormalizer.from_config(metadata['normalization'])
                print(f"Testing REGRESSION model")
        else:
            # Fallback: extract from filename
            parts = model_filename.split('_')
            feature = '_'.join(parts[1:-2]) if len(parts) > 3 else parts[1]
            is_classification = feature.startswith('Custom_Class_')
            if is_classification:
                training_mode = "classification"
                num_classes = int(feature.replace('Custom_Class_', ''))
                num_outputs = num_classes
        
        # Load training data
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        images_data = []
        for img in images:
            images_data.append({
                'id': img.id,
                'processed_image_data': img.processed_image_data,
                'features_data': img.features_data
            })
        
        # Create dataloaders with correct configuration
        train_loader, val_loader, stats = create_dataloaders(
            images_data,
            feature,
            train_split=0.8,
            batch_size=8,
            normalizer=normalizer,
            is_classification=is_classification if 'is_classification' in locals() else False,
            num_classes=num_classes if 'num_classes' in locals() else None
        )
        
        # Create trainer with correct configuration
        trainer = CNNTrainer(
            num_outputs=num_outputs,
            normalizer=normalizer,
            training_mode=training_mode
        )
        trainer.load_model(str(model_path))
        
        # Evaluate on both sets
        train_metrics = trainer.evaluate_metrics(train_loader)
        val_metrics = trainer.evaluate_metrics(val_loader)
        
        return {
            'success': True,
            'model': model_filename,
            'target_feature': feature,
            'training_mode': training_mode,
            'num_outputs': num_outputs,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'dataset_stats': stats
        }
        
    except Exception as e:
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

