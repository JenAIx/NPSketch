"""
AI Training Classification Router

Endpoints for feature distribution analysis and class generation.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import get_db, TrainingDataImage
import json
import numpy as np
import re
import sys

sys.path.insert(0, '/app')
from ai_training.classification_generator import generate_balanced_classes
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/ai-training", tags=["ai_training_classification"])


@router.get("/feature-distribution/{feature_name}")
async def get_feature_distribution(
    feature_name: str,
    db: Session = Depends(get_db)
):
    """
    Get distribution data for a feature (histogram, stats, auto-classifications).
    Returns only aggregated data, not individual values.
    
    Args:
        feature_name: Name of the feature (e.g., 'Total_Score')
    
    Returns:
        Distribution data with histogram bins, statistics, and pre-calculated class splits
    """
    try:
        # 1. Fetch all scores from DB
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        scores = []
        for img in images:
            try:
                features = json.loads(img.features_data)
                if feature_name in features:
                    score = float(features[feature_name])
                    scores.append(score)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
        
        if len(scores) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No samples found with feature '{feature_name}'"
            )
        
        scores = np.array(scores)
        
        # 2. Calculate statistics
        stats = {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "range": float(np.max(scores) - np.min(scores)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75))
        }
        
        # 3. Generate histogram (25 bins)
        hist_counts, hist_edges = np.histogram(scores, bins=25)
        histogram_data = []
        total_samples = len(scores)
        
        for i in range(len(hist_counts)):
            bin_min = float(hist_edges[i])
            bin_max = float(hist_edges[i + 1])
            count = int(hist_counts[i])
            percentage = (count / total_samples) * 100.0
            
            histogram_data.append({
                "min": bin_min,
                "max": bin_max,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        histogram = {
            "bins": 25,
            "data": histogram_data
        }
        
        # 4. Pre-calculate auto-classifications (2-5 classes)
        auto_classifications = {}
        
        for num_classes in [2, 3, 4, 5]:
            try:
                result = generate_balanced_classes(scores, num_classes, method="quantile")
                auto_classifications[f"{num_classes}_classes"] = result
            except Exception as e:
                # If classification fails, skip this number
                logger.warning(f"Could not generate {num_classes} classes: {e}")
                continue
        
        return {
            "feature_name": feature_name,
            "total_samples": total_samples,
            "statistics": stats,
            "histogram": histogram,
            "auto_classifications": auto_classifications
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating distribution: {str(e)}")


@router.post("/generate-classes")
async def generate_and_save_classes(
    config: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Generate class boundaries and save class labels to database.
    
    Request Body:
    {
        "feature_name": "Total_Score",
        "num_classes": 4,
        "method": "quantile" | "custom",
        "custom_classes": [...]  // Optional, for custom boundaries/names
    }
    
    Returns:
        Success status and class distribution
    """
    try:
        feature_name = config.get("feature_name")
        num_classes = config.get("num_classes")
        method = config.get("method", "quantile")
        custom_classes = config.get("custom_classes")  # Custom boundaries and names
        
        if not feature_name:
            raise HTTPException(status_code=400, detail="feature_name is required")
        
        if not num_classes or num_classes < 2 or num_classes > 10:
            raise HTTPException(
                status_code=400,
                detail="num_classes must be between 2 and 10"
            )
        
        # 1. Get all scores
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        scores = []
        valid_images = []
        
        for img in images:
            try:
                features = json.loads(img.features_data)
                if feature_name in features:
                    score = float(features[feature_name])
                    scores.append(score)
                    valid_images.append(img)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
        
        if len(scores) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No samples found with feature '{feature_name}'"
            )
        
        scores = np.array(scores)
        
        # 2. Determine class structure (custom or auto-generate)
        if method == "custom" and custom_classes:
            # Use custom classes from frontend
            class_definitions = custom_classes
            logger.info(f"Using custom classes: {len(custom_classes)} classes")
        else:
            # Auto-generate with classification_generator
            result = generate_balanced_classes(scores, num_classes, method="quantile")
            class_definitions = result["classes"]
            logger.info(f"Auto-generated: {len(class_definitions)} classes")
        
        # Build boundaries list from class definitions
        all_boundaries = [class_definitions[0]["min"]]
        for cls in class_definitions:
            all_boundaries.append(cls["max"])
        
        # 3. Assign classes and update DB (REPLACE Custom_Class completely!)
        updated_count = 0
        class_distribution = {i: 0 for i in range(len(class_definitions))}
        
        for img in valid_images:
            try:
                features = json.loads(img.features_data)
                score = float(features[feature_name])
                
                # Determine class based on score ranges
                class_id = None
                for cls in class_definitions:
                    if score >= cls["min"] and score <= cls["max"]:
                        class_id = cls["id"]
                        break
                
                if class_id is None:
                    # Score outside all ranges - skip
                    logger.warning(f"Score {score} not in any class range")
                    continue
                
                # Get class info
                cls_info = class_definitions[class_id]
                
                # Update features_data
                # Total_Score bleibt unverändert!
                
                # REPLACE Custom_Class completely (only one active classification!)
                num_classes_str = str(len(class_definitions))
                features["Custom_Class"] = {
                    num_classes_str: {
                        "label": int(class_id),
                        "name_custom": cls_info.get("custom_name"),
                        "name_generic": cls_info.get("generic_name") or f"Class_{class_id} [{cls_info['min']}-{cls_info['max']}]",
                        "boundaries": all_boundaries
                    }
                }
                
                img.features_data = json.dumps(features)
                class_distribution[class_id] += 1
                updated_count += 1
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Error processing image {img.id}: {e}", exc_info=True)
                continue
        
        db.commit()
        
        # Calculate class percentages
        class_info = []
        total = updated_count
        for cls_def in class_definitions:
            class_id = cls_def["id"]
            count = class_distribution.get(class_id, 0)
            percentage = (count / total * 100.0) if total > 0 else 0.0
            class_info.append({
                "class_id": class_id,
                "count": count,
                "percentage": round(percentage, 2),
                "range": f"{cls_def['min']}-{cls_def['max']}",
                "custom_name": cls_def.get("custom_name"),
                "generic_name": cls_def.get("generic_name")
            })
        
        return {
            "success": True,
            "updated_count": updated_count,
            "num_classes": len(class_definitions),
            "actual_num_classes": len(class_definitions),
            "method": method,
            "boundaries": all_boundaries,
            "class_distribution": class_distribution,
            "class_info": class_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error generating classes: {str(e)}")


@router.get("/custom-class-distribution/{feature_name}")
async def get_custom_class_distribution(
    feature_name: str,
    db: Session = Depends(get_db)
):
    """
    Get distribution for an existing Custom_Class feature.
    
    Args:
        feature_name: e.g., "Custom_Class_5"
    
    Returns:
        Class distribution with counts and percentages
    """
    try:
        # Extract num_classes from feature name
        if not feature_name.startswith("Custom_Class_"):
            raise HTTPException(status_code=400, detail="Invalid feature name")
        
        num_classes_str = feature_name.replace("Custom_Class_", "")
        
        # Get all images with this classification
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        class_counts = {}
        class_info = {}
        total_samples = 0
        
        for img in images:
            try:
                features = json.loads(img.features_data)
                
                # Check if Custom_Class exists
                if "Custom_Class" not in features:
                    continue
                
                # Check if this num_classes exists
                if num_classes_str not in features["Custom_Class"]:
                    continue
                
                class_data = features["Custom_Class"][num_classes_str]
                class_id = class_data["label"]
                
                # Count this class
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                    class_info[class_id] = {
                        "name_custom": class_data.get("name_custom"),
                        "name_generic": class_data.get("name_generic"),
                        "boundaries": class_data.get("boundaries")
                    }
                
                class_counts[class_id] += 1
                total_samples += 1
                
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        
        if total_samples == 0:
            raise HTTPException(status_code=404, detail="No samples found with this classification")
        
        # Build class list with percentages
        classes = []
        boundaries = class_info[0]["boundaries"] if 0 in class_info else []
        
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / total_samples) * 100.0
            info = class_info[class_id]
            
            # Extract range from boundaries or generic name
            if boundaries and len(boundaries) > class_id + 1:
                # Calculate non-overlapping range for this class
                range_min = boundaries[class_id]
                range_max = boundaries[class_id + 1] - 1 if class_id < len(boundaries) - 2 else boundaries[class_id + 1]
                
                range_str = f"[{range_min}, {range_max}]" if range_min != range_max else f"= {range_min}"
            else:
                # Parse from generic name as fallback
                match = re.search(r'\[([^\]]+)\]', info["name_generic"] or "")
                range_str = match.group(0) if match else f"Class {class_id}"
            
            classes.append({
                "id": class_id,
                "name_custom": info["name_custom"],
                "name_generic": info["name_generic"],
                "range": range_str,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        return {
            "feature_name": feature_name,
            "num_classes": num_classes_str,
            "total_samples": total_samples,
            "classes": classes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading custom class distribution: {str(e)}")


@router.post("/recalculate-classes")
async def recalculate_class_counts(
    config: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Recalculate class counts based on custom boundaries.
    
    Request Body:
    {
        "feature_name": "Total_Score",
        "num_classes": 5,
        "boundaries": [[2, 42], [43, 50], [51, 58], [59, 59], [60, 60]]
    }
    """
    try:
        feature_name = config.get("feature_name")
        num_classes = config.get("num_classes")
        boundaries = config.get("boundaries")  # List of [min, max] pairs
        
        if not feature_name or not boundaries:
            raise HTTPException(status_code=400, detail="feature_name and boundaries required")
        
        # Get all scores
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        scores = []
        for img in images:
            try:
                features = json.loads(img.features_data)
                if feature_name in features:
                    scores.append(float(features[feature_name]))
            except:
                continue
        
        if len(scores) == 0:
            raise HTTPException(status_code=404, detail="No scores found")
        
        scores = np.array(scores)
        total = len(scores)
        
        # Calculate counts for each class
        result_classes = []
        for class_id, (class_min, class_max) in enumerate(boundaries):
            # Count scores in this range
            mask = (scores >= class_min) & (scores <= class_max)
            count = int(np.sum(mask))
            percentage = (count / total) * 100.0
            
            result_classes.append({
                "id": class_id,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        return {
            "success": True,
            "classes": result_classes,
            "total": total
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recalculating: {str(e)}")

