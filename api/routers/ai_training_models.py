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
            
            # Extract configuration from metadata (with defensive fallbacks)
            feature = metadata.get('target_feature')
            model_info = metadata.get('model', {})
            num_outputs = model_info.get('output_neurons')
            
            # If critical fields are missing, fall back to filename-based detection
            if not feature or num_outputs is None:
                logger.warning(f"Metadata missing critical fields (target_feature or model.output_neurons), falling back to filename detection")
                metadata = None  # Force fallback to filename detection
                # Reset variables to ensure clean fallback
                feature = None
                num_outputs = None
            else:
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
                        # Get normalizer if used (respect enabled flag, check method for backward compatibility)
                        norm_config = metadata.get('normalization', {})
                        enabled = norm_config.get('enabled', True)  # Default to True for backward compatibility
                        if enabled and norm_config.get('method'):
                            from ai_training.normalization import TargetNormalizer
                            try:
                                normalizer = TargetNormalizer.from_config(norm_config)
                                logger.info(f"Normalizer loaded: method={norm_config.get('method')}, enabled={enabled}")
                            except Exception as e:
                                logger.warning(f"Failed to load normalizer: {e}")
                                normalizer = None
                        logger.info(f"Testing REGRESSION model (from metadata)")
                else:
                    # Fallback: detect from feature name (feature is guaranteed to exist here)
                    if feature:
                        is_classification = feature.startswith('Custom_Class_')
                    else:
                        # Should not reach here if defensive checks above work, but safety net
                        is_classification = False
                    if is_classification:
                        training_mode = "classification"
                        num_classes = int(feature.replace('Custom_Class_', ''))
                        logger.info(f"Testing CLASSIFICATION model: {num_classes} classes (detected from feature)")
                    else:
                        training_mode = "regression"
                        # Get normalizer if used (respect enabled flag, check method for backward compatibility)
                        norm_config = metadata.get('normalization', {})
                        enabled = norm_config.get('enabled', True)  # Default to True for backward compatibility
                        if enabled and norm_config.get('method'):
                            from ai_training.normalization import TargetNormalizer
                            try:
                                normalizer = TargetNormalizer.from_config(norm_config)
                                logger.info(f"Normalizer loaded: method={norm_config.get('method')}, enabled={enabled}")
                            except Exception as e:
                                logger.warning(f"Failed to load normalizer: {e}")
                                normalizer = None
                        logger.info(f"Testing REGRESSION model (detected from feature)")
        
        # Execute filename-based fallback if metadata is None (either file doesn't exist or was set to None due to missing fields)
        if metadata is None:
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
                num_outputs = 1  # Regression models have 1 output neuron
                logger.info(f"Testing REGRESSION model (from filename)")
        
        if not feature:
            raise HTTPException(status_code=400, detail="Could not determine target feature from model")
        
        # Prepare test dataloaders using holdout data
        # Only use prepare_test_dataloaders if metadata exists and has val_image_ids
        if metadata and metadata.get('val_image_ids'):
            try:
                test_loader, test_stats = prepare_test_dataloaders(
                    feature=feature,
                    metadata=metadata,
                    normalizer=normalizer,
                    is_classification=is_classification,
                    num_classes=num_classes,
                    db=db
                )
            except Exception as e:
                logger.error(f"Failed to prepare test data: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to prepare test data: {str(e)}")
        else:
            # Fallback: Load all available images with the feature (metadata missing or incomplete)
            logger.warning(f"Metadata missing or incomplete (no val_image_ids). Loading all available images with feature '{feature}' for testing. "
                          f"Note: This may include images used during training (data leakage possible).")
            
            # Load all images with the target feature from database
            from ai_training.dataset import DrawingDataset
            from torch.utils.data import DataLoader
            
            all_images = db.query(TrainingDataImage).filter(
                TrainingDataImage.features_data.isnot(None)
            ).all()
            
            # Filter images that have the target feature
            test_images = []
            for img in all_images:
                try:
                    features = json.loads(img.features_data)
                    if feature in features or (is_classification and 'Custom_Class' in features):
                        test_images.append({
                            'id': img.id,
                            'processed_image_data': img.processed_image_data,
                            'features_data': img.features_data
                        })
                except (json.JSONDecodeError, KeyError):
                    continue
            
            if len(test_images) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No images found with feature '{feature}' in database"
                )
            
            logger.info(f"Loaded {len(test_images)} images for testing (fallback mode - may include training data)")
            
            # Create test dataloader
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
            
            test_stats = {
                'test_samples': len(test_dataset),
                'test_batches': len(test_loader)
            }
        
        # Validate test dataloader
        if len(test_loader) == 0:
            raise HTTPException(status_code=400, detail=f"No test samples found for feature '{feature}'")
        
        # Get use_sigmoid from metadata (if available)
        use_sigmoid = None
        if metadata:
            use_sigmoid = metadata.get('use_sigmoid')
        
        # Create trainer and load model
        trainer = CNNTrainer(
            num_outputs=num_outputs,
            normalizer=normalizer,
            training_mode=training_mode,
            use_sigmoid=use_sigmoid  # Use explicit value from metadata if available
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
    
    PREPROCESSING ALIGNMENT (2026-01-20):
    - Uses shared preprocessing module to match training pipeline
    - Applies pre-shrink if model was trained with augmentation that used pre-shrink
    - Reads pre-shrink config from model metadata
    
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
    from ai_training.preprocessing import (
        preprocess_bytes_for_prediction,
        get_preprocessing_config_from_metadata
    )
    
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
        metadata = {}  # Store full metadata for preprocessing config
        
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
        
        # Load and preprocess image using shared preprocessing pipeline
        # This ensures consistency with training: resize, binarize, line normalize, pre-shrink
        image_bytes = await file.read()
        
        # Get preprocessing config from model metadata
        preproc_config = get_preprocessing_config_from_metadata(metadata)
        logger.info(f"Prediction preprocessing: pre_shrink_enabled={preproc_config['pre_shrink_enabled']}, "
                   f"factor={preproc_config['pre_shrink_factor']}")
        
        # Apply full preprocessing pipeline (matches training)
        img_array = preprocess_bytes_for_prediction(
            image_bytes,
            metadata=metadata,
            debug=False
        )
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Resolve use_sigmoid from metadata so the rebuilt model matches the
        # trained architecture. Sigmoid has no parameters, so loading a sigmoid
        # checkpoint into a linear head succeeds silently but skips the
        # activation - predictions would be wrong.
        use_sigmoid = metadata.get('use_sigmoid')
        if use_sigmoid is None:
            use_sigmoid = metadata.get('model', {}).get('use_sigmoid')
        if use_sigmoid is None:
            # Legacy fallback: all old regression models with min-max
            # normalization were trained with a sigmoid head.
            use_sigmoid = (training_mode == "regression"
                           and bool(metadata.get('normalization', {}).get('method')))

        # Load model
        model = DrawingClassifier(num_outputs=num_outputs, pretrained=False, use_sigmoid=bool(use_sigmoid))
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

            if training_mode == "components":
                # 60 sub-labels (20 elements x PRES/ACC/POS); Total_Score = sum.
                # Apply post-hoc calibration if present in metadata: per-label decision
                # thresholds + a non-negative linear readout of the 60 probabilities.
                probs = torch.sigmoid(output)[0].tolist()
                thr = metadata.get('thresholds')
                if not (isinstance(thr, list) and len(thr) == 60):
                    thr = [0.5] * 60
                hard = [1 if probs[j] >= thr[j] else 0 for j in range(60)]
                elements = []
                for e in range(20):
                    pr, ac, po = probs[3*e], probs[3*e+1], probs[3*e+2]
                    elements.append({
                        'element': e + 1,
                        'presence': round(pr, 3), 'accuracy': round(ac, 3), 'position': round(po, 3),
                        'thr_presence': round(thr[3*e], 3), 'thr_accuracy': round(thr[3*e+1], 3),
                        'thr_position': round(thr[3*e+2], 3),
                        'subscore_hard': int(hard[3*e] + hard[3*e+1] + hard[3*e+2]),
                    })
                sc = metadata.get('score_calibration') or {}
                w = sc.get('weights'); bcal = sc.get('bias')
                calibrated = None
                if isinstance(w, list) and len(w) == 60 and bcal is not None:
                    calibrated = round(float(sum(w[j] * probs[j] for j in range(60)) + bcal), 1)
                    calibrated = max(0.0, min(60.0, calibrated))
                total_score = calibrated if calibrated is not None else int(sum(hard))
                return {
                    'success': True,
                    'model': model_filename,
                    'target_feature': 'Components',
                    'training_mode': 'components',
                    'prediction': {
                        'total_score': total_score,
                        'calibrated': calibrated is not None,
                        'total_score_hard': int(sum(hard)),
                        'total_score_soft': round(float(sum(probs)), 2),
                        'elements': elements,
                    }
                }
            elif training_mode == "classification":
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

                    # Clamp to valid range (report whether clamping occurred,
                    # frequent clamping indicates a model/config problem)
                    unclamped_value = predicted_value
                    predicted_value = max(min_val, min(max_val, predicted_value))
                    was_clamped = (unclamped_value != predicted_value)
                else:
                    was_clamped = False

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
                        'raw_output': round(raw_value, 4),
                        'was_clamped': was_clamped
                    }
                }
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/run-on-test-images")
async def run_on_test_images(request: dict = Body(...), db: Session = Depends(get_db)):
    """
    Run a trained model over the DRAWN test images and compare predicted vs
    expected (the "Run Tests" batch). Each DRAWN image with features is scored
    using the same preprocessing as predict-single.
    """
    import os, io, torch
    import numpy as np
    from pathlib import Path
    from ai_training.model import DrawingClassifier
    from ai_training.preprocessing import preprocess_bytes_for_prediction

    model_filename = request.get("model_filename")
    if not model_filename:
        raise HTTPException(status_code=400, detail="model_filename required")
    # Which images to evaluate on, and how many. DRAWN/UPLOAD are genuine test images
    # (not in the TELEFRED training set); the training sources are optimistic.
    source_format = request.get("source_format", "DRAWN")
    try:
        limit = max(1, min(int(request.get("limit", 300)), 2000))
    except (TypeError, ValueError):
        limit = 300
    model_path = Path("/app/data/models") / model_filename
    meta_path = Path("/app/data/models") / f"{model_path.stem}_metadata.json"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    metadata = json.load(open(meta_path)) if meta_path.exists() else {}
    mode = metadata.get("training_mode", "regression")
    num_outputs = metadata.get("model", {}).get("output_neurons", 1)
    use_sigmoid = bool(metadata.get("use_sigmoid", False))
    norm = metadata.get("normalization", {})

    model = DrawingClassifier(num_outputs=num_outputs, pretrained=False, use_sigmoid=use_sigmoid)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    def expected_total(feats):
        c = feats.get("components")
        if c:
            return sum(c["presence"]) + sum(c["accuracy"]) + sum(c["position"])
        return feats.get("Total_Score")

    rows = db.query(TrainingDataImage).filter(
        TrainingDataImage.source_format == source_format,
        TrainingDataImage.features_data.isnot(None),
    ).limit(limit).all()

    results, diffs = [], []
    for r in rows:
        feats = json.loads(r.features_data)
        exp = expected_total(feats)
        # use the normalized stored image (consistent with predict-single / calibration;
        # original_file_data is raw red ink for TELEFRED and would be mis-read)
        arr = preprocess_bytes_for_prediction(r.processed_image_data, metadata=metadata)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = model(t)
        if mode == "components":
            # Mirror predict-single: calibrated readout if available, else
            # per-label-threshold hard sum, else naive soft sum.
            probs = torch.sigmoid(out)[0].numpy()
            sc = metadata.get("score_calibration") or {}
            w = sc.get("weights"); bcal = sc.get("bias")
            thr = metadata.get("thresholds")
            if isinstance(w, list) and len(w) == 60 and bcal is not None:
                pred = round(max(0.0, min(60.0, float(np.dot(probs, w) + bcal))), 2)
            elif isinstance(thr, list) and len(thr) == 60:
                pred = int(sum(1 for j in range(60) if probs[j] >= thr[j]))
            else:
                pred = int(round(float(probs.sum())))
        elif mode == "classification":
            pred = int(torch.argmax(out, dim=1).item())
        else:
            v = out[0][0].item()
            if norm.get("method") == "min_max":
                mn, mx = norm.get("min_value", 0), norm.get("max_value", 60)
                v = max(mn, min(mx, v * (mx - mn) + mn))
            pred = round(v, 2)
        row = {"id": r.id, "name": r.test_name or r.patient_id, "expected": exp, "predicted": pred}
        if exp is not None and mode != "classification":
            row["abs_error"] = round(abs(pred - exp), 2)
            diffs.append(abs(pred - exp))
        results.append(row)

    # sort worst-error first so problem cases are visible at the top
    results.sort(key=lambda r: r.get("abs_error", -1), reverse=True)

    summary = {"count": len(results), "training_mode": mode, "source_format": source_format}
    if diffs:
        d = np.array(diffs, dtype=float)
        preds = np.array([r["predicted"] for r in results if r.get("abs_error") is not None], dtype=float)
        exps = np.array([r["expected"] for r in results if r.get("abs_error") is not None], dtype=float)
        summary["mae"] = round(float(d.mean()), 2)
        summary["rmse"] = round(float(np.sqrt((d ** 2).mean())), 2)
        summary["bias"] = round(float((preds - exps).mean()), 2)   # +→ over-predicts
        summary["within_3"] = round(float((d <= 3).mean()) * 100, 1)
        summary["within_5"] = round(float((d <= 5).mean()) * 100, 1)
    return {"success": True, "model": model_filename, "summary": summary, "results": results}


@router.get("/models/{model_filename}/predict/{image_id}")
async def predict_by_id(model_filename: str, image_id: int, db: Session = Depends(get_db)):
    """Run a model on a stored DB image and return the prediction (same shape as
    predict-single) — used to show the per-element breakdown alongside the Grad-CAM."""
    import torch
    from pathlib import Path
    from ai_training.model import DrawingClassifier
    from ai_training.preprocessing import preprocess_bytes_for_prediction

    model_path = Path("/app/data/models") / model_filename
    meta_path = Path("/app/data/models") / f"{model_path.stem}_metadata.json"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    metadata = json.load(open(meta_path)) if meta_path.exists() else {}
    mode = metadata.get("training_mode", "regression")
    num_outputs = metadata.get("model", {}).get("output_neurons", 1)
    use_sigmoid = bool(metadata.get("use_sigmoid", False))

    row = db.query(TrainingDataImage).filter(TrainingDataImage.id == image_id).first()
    if not row or not row.processed_image_data:
        raise HTTPException(status_code=404, detail="Image not found")

    model = DrawingClassifier(num_outputs=num_outputs, pretrained=False, use_sigmoid=use_sigmoid)
    ck = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck)
    model.eval()
    arr = preprocess_bytes_for_prediction(row.processed_image_data, metadata=metadata, debug=False)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        out = model(t)

    if mode == "components":
        probs = torch.sigmoid(out)[0].tolist()
        thr = metadata.get("thresholds")
        if not (isinstance(thr, list) and len(thr) == 60):
            thr = [0.5] * 60
        hard = [1 if probs[j] >= thr[j] else 0 for j in range(60)]
        elements = [{
            "element": e + 1,
            "presence": round(probs[3*e], 3), "accuracy": round(probs[3*e+1], 3), "position": round(probs[3*e+2], 3),
            "thr_presence": round(thr[3*e], 3), "thr_accuracy": round(thr[3*e+1], 3), "thr_position": round(thr[3*e+2], 3),
            "subscore_hard": int(hard[3*e] + hard[3*e+1] + hard[3*e+2]),
        } for e in range(20)]
        sc = metadata.get("score_calibration") or {}
        w = sc.get("weights"); bcal = sc.get("bias")
        calibrated = None
        if isinstance(w, list) and len(w) == 60 and bcal is not None:
            calibrated = round(max(0.0, min(60.0, float(sum(w[j]*probs[j] for j in range(60)) + bcal))), 1)
        return {"success": True, "training_mode": "components", "prediction": {
            "total_score": calibrated if calibrated is not None else int(sum(hard)),
            "calibrated": calibrated is not None,
            "total_score_hard": int(sum(hard)), "total_score_soft": round(float(sum(probs)), 2),
            "elements": elements}}
    elif mode == "classification":
        probs = torch.softmax(out, dim=1)[0]
        return {"success": True, "training_mode": "classification",
                "prediction": {"class": int(torch.argmax(probs).item()),
                               "confidence": round(float(probs.max().item()) * 100, 1)}}
    else:
        v = out[0, 0].item()
        norm = metadata.get("normalization", {})
        if norm.get("method") == "min_max":
            mn, mx = norm.get("min_value", 0), norm.get("max_value", 60)
            v = max(mn, min(mx, v * (mx - mn) + mn))
        return {"success": True, "training_mode": "regression",
                "prediction": {"value": round(v, 2)}}


@router.get("/models/{model_filename}/gradcam/{image_id}")
async def gradcam_overlay(model_filename: str, image_id: int, element: int = None,
                          db: Session = Depends(get_db)):
    """Grad-CAM (XAI) overlay for one image + model: where the model looks to produce
    its score (default), or for a single element E1..E20 if `element` is given.
    Drawn over the figure with the reference faded behind. Returns a PNG."""
    import io
    import torch
    import numpy as np
    import cv2
    from pathlib import Path
    from fastapi.responses import StreamingResponse
    from ai_training.model import DrawingClassifier
    from ai_training.preprocessing import preprocess_bytes_for_prediction

    W, H = 568, 274
    model_path = Path("/app/data/models") / model_filename
    meta_path = Path("/app/data/models") / f"{model_path.stem}_metadata.json"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    metadata = json.load(open(meta_path)) if meta_path.exists() else {}
    mode = metadata.get("training_mode", "regression")
    num_outputs = metadata.get("model", {}).get("output_neurons", 1)
    use_sigmoid = bool(metadata.get("use_sigmoid", False))

    row = db.query(TrainingDataImage).filter(TrainingDataImage.id == image_id).first()
    if not row or not row.processed_image_data:
        raise HTTPException(status_code=404, detail="Image not found")

    model = DrawingClassifier(num_outputs=num_outputs, pretrained=False, use_sigmoid=use_sigmoid)
    ck = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck)
    model.eval()

    arr = preprocess_bytes_for_prediction(row.processed_image_data, metadata=metadata, debug=False)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    acts = {}
    handle = model.backbone.layer3.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    out = model(x)
    A = acts["a"]
    # scalar target that drives the reported score
    if mode == "components":
        if element and 1 <= int(element) <= 20:
            e = int(element) - 1                       # this element's 3 sub-label logits
            target = out[0, 3 * e] + out[0, 3 * e + 1] + out[0, 3 * e + 2]
        else:
            sc = metadata.get("score_calibration") or {}
            w = sc.get("weights")
            probs = torch.sigmoid(out[0])
            target = (torch.tensor(w, dtype=torch.float32) * probs).sum() if (isinstance(w, list) and len(w) == 60) else probs.sum()
    elif mode == "classification":
        target = out[0, int(torch.argmax(out[0]).item())]
    else:
        target = out[0, 0]
    g = torch.autograd.grad(target, A)[0]
    cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * A).sum(dim=1)).squeeze(0).detach().numpy()
    handle.remove()
    cam = cv2.resize(cam, (W, H))
    m = cam.max()
    cam = (cam / m) ** 1.3 if m > 0 else cam

    draw = cv2.resize((arr * 255).clip(0, 255).astype(np.uint8), (W, H)).astype(np.float32)
    ref_arr = preprocess_bytes_for_prediction(open("/app/templates/reference_image.png", "rb").read(),
                                              metadata=metadata, debug=False)
    ref = cv2.resize((ref_arr * 255).clip(0, 255).astype(np.uint8), (W, H)).astype(np.float32)
    base = 255 - (255 - ref) * 0.20            # reference faded to ~20% darkness
    base = np.minimum(base, draw)              # the actual drawing on top, full strength
    bg = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    a = (cam[..., None] * 0.6)
    out_img = (bg * (1 - a) + heat * a).clip(0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".png", out_img)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


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



# --------------------------------------------------------------------------
# Component Map (explainability) — rebuild the per-element heatmaps
# --------------------------------------------------------------------------
import threading as _threading
from datetime import datetime as _dt

_component_map_job = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def _run_component_map_build():
    global _component_map_job
    _component_map_job.update(running=True, error=None,
                              started_at=_dt.now().isoformat(), finished_at=None)
    try:
        from ai_training.component_heatmaps import main as _gen
        _gen()
        _component_map_job["finished_at"] = _dt.now().isoformat()
        logger.info("Component map rebuild complete")
    except Exception as e:
        logger.error(f"Component map rebuild failed: {e}", exc_info=True)
        _component_map_job["error"] = str(e)
    finally:
        _component_map_job["running"] = False


@router.post("/component-map/rebuild")
async def rebuild_component_map():
    """Regenerate the 20 per-element heatmaps (data-driven + Grad-CAM) in the background."""
    if _component_map_job["running"]:
        return {"success": False, "message": "Rebuild already running", "status": _component_map_job}
    _threading.Thread(target=_run_component_map_build, daemon=True).start()
    return {"success": True, "message": "Component map rebuild started"}


@router.get("/component-map/status")
async def component_map_status():
    """Current status of the component-map rebuild job."""
    return _component_map_job
