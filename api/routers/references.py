"""
References Router - Reference image management endpoints for NPSketch API

Contains endpoints for:
- Listing reference images
- Getting reference image data and features
- Manual reference creation and editing
- Reference feature management (add/delete/clear)
- Reference status checking
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, FileResponse
from sqlalchemy.orm import Session
from typing import List
from database import get_db, ReferenceImage
from models import ReferenceImageResponse
from services import ReferenceService
import json
import os

router = APIRouter(prefix="/api", tags=["references"])

# Path to default reference image
DEFAULT_REFERENCE_IMAGE = "/app/templates/reference_image.png"


@router.get("/references", response_model=List[ReferenceImageResponse])
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


@router.get("/references/{ref_id}/image")
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
    
    return Response(content=reference.processed_image_data, media_type="image/png")


@router.get("/references/{ref_id}/features")
async def get_reference_features(
    ref_id: int,
    db: Session = Depends(get_db)
):
    """
    Get reference image features (detected lines).
    
    Args:
        ref_id: Reference image ID
        db: Database session
        
    Returns:
        Feature data including detected lines
    """
    ref_service = ReferenceService(db)
    reference = ref_service.get_reference_by_id(ref_id)
    
    if not reference:
        raise HTTPException(status_code=404, detail="Reference not found")
    
    features = json.loads(reference.feature_data)
    
    return {
        "reference_id": reference.id,
        "reference_name": reference.name,
        "num_lines": features.get("num_lines", 0),
        "lines": features.get("lines", []),
        "line_lengths": features.get("line_lengths", []),
        "line_angles": features.get("line_angles", []),
        "image_shape": features.get("image_shape", []),
        "num_contours": features.get("num_contours", 0)
    }


@router.post("/reference/manual")
async def create_manual_reference(data: dict, db: Session = Depends(get_db)):
    """Create reference from manually drawn lines."""
    import numpy as np
    from image_processing.utils import image_to_bytes
    
    # Delete existing
    db.query(ReferenceImage).delete()
    db.commit()
    
    # Create features
    lines = []
    line_angles = []
    line_lengths = []
    
    for line_data in data['lines']:
        x1, y1 = line_data['start']['x'], line_data['start']['y']
        x2, y2 = line_data['end']['x'], line_data['end']['y']
        
        lines.append([x1, y1, x2, y2])
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        line_angles.append(angle)
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        line_lengths.append(length)
    
    features = {
        'num_lines': len(lines),
        'lines': lines,
        'image_shape': [568, 274],  # [width, height]
        'line_lengths': line_lengths,
        'line_angles': line_angles,
        'line_counts': data['summary']
    }
    
    # Create white canvas at 568×274 (note: numpy is height, width)
    img = np.ones((274, 568, 3), dtype=np.uint8) * 255
    
    ref = ReferenceImage(
        name="manual_reference",
        image_data=image_to_bytes(img),
        processed_image_data=image_to_bytes(img),
        feature_data=json.dumps(features),
        width=568,
        height=274
    )
    
    db.add(ref)
    db.commit()
    
    return {"success": True, "lines_count": len(lines), "summary": data['summary']}


@router.get("/reference/status")
async def get_reference_status(db: Session = Depends(get_db)):
    """Check if reference is properly initialized with features."""
    ref = db.query(ReferenceImage).first()
    
    if not ref:
        return {
            "initialized": False,
            "message": "No reference image found"
        }
    
    # Check if features exist and are valid
    if not ref.feature_data:
        return {
            "initialized": False,
            "message": "Reference exists but has no features"
        }
    
    try:
        features = json.loads(ref.feature_data)
        num_lines = features.get('num_lines', 0)
        
        # Allow any number of lines (no minimum required)
        if num_lines < 1:
            return {
                "initialized": False,
                "message": "No features defined yet"
            }
        
        return {
            "initialized": True,
            "message": f"Reference properly initialized with {num_lines} lines",
            "num_lines": num_lines,
            "line_counts": features.get('line_counts', {})
        }
    except:
        return {
            "initialized": False,
            "message": "Invalid feature data"
        }


@router.post("/reference/features")
async def add_reference_feature(
    feature: dict,
    db: Session = Depends(get_db)
):
    """Add a new feature line to reference."""
    import numpy as np
    
    ref = db.query(ReferenceImage).first()
    if not ref:
        raise HTTPException(status_code=404, detail="No reference image found")
    
    # Load existing features
    if ref.feature_data:
        features = json.loads(ref.feature_data)
    else:
        features = {
            'num_lines': 0,
            'lines': [],
            'line_angles': [],
            'line_lengths': [],
            'image_shape': [256, 256],
            'line_counts': {'horizontal': 0, 'vertical': 0, 'diagonal': 0, 'total': 0}
        }
    
    # Add new line
    x1, y1 = feature['start']['x'], feature['start']['y']
    x2, y2 = feature['end']['x'], feature['end']['y']
    
    features['lines'].append([x1, y1, x2, y2])
    
    # Calculate angle
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    features['line_angles'].append(angle)
    
    # Calculate length
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    features['line_lengths'].append(length)
    
    # Update counts
    features['num_lines'] = len(features['lines'])
    
    # Categorize
    norm_angle = abs(angle) if abs(angle) <= 90 else 180 - abs(angle)
    if norm_angle < 15:
        features['line_counts']['horizontal'] += 1
    elif norm_angle > 75:
        features['line_counts']['vertical'] += 1
    else:
        features['line_counts']['diagonal'] += 1
    
    features['line_counts']['total'] = features['num_lines']
    
    # Save
    ref.feature_data = json.dumps(features)
    db.commit()
    
    return {
        "success": True,
        "feature_id": features['num_lines'] - 1,
        "total_features": features['num_lines'],
        "line_counts": features['line_counts']
    }


@router.delete("/reference/features/{feature_id}")
async def delete_reference_feature(
    feature_id: int,
    db: Session = Depends(get_db)
):
    """Delete a feature line from reference."""
    ref = db.query(ReferenceImage).first()
    if not ref:
        raise HTTPException(status_code=404, detail="No reference image found")
    
    if not ref.feature_data:
        raise HTTPException(status_code=404, detail="No features found")
    
    features = json.loads(ref.feature_data)
    
    if feature_id < 0 or feature_id >= len(features['lines']):
        raise HTTPException(status_code=404, detail="Feature not found")
    
    # Remove feature
    features['lines'].pop(feature_id)
    features['line_angles'].pop(feature_id)
    features['line_lengths'].pop(feature_id)
    features['num_lines'] = len(features['lines'])
    
    # Recalculate counts
    features['line_counts'] = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
    
    for angle in features['line_angles']:
        norm_angle = abs(angle) if abs(angle) <= 90 else 180 - abs(angle)
        if norm_angle < 15:
            features['line_counts']['horizontal'] += 1
        elif norm_angle > 75:
            features['line_counts']['vertical'] += 1
        else:
            features['line_counts']['diagonal'] += 1
    
    features['line_counts']['total'] = features['num_lines']
    
    # Save
    ref.feature_data = json.dumps(features)
    db.commit()
    
    return {
        "success": True,
        "total_features": features['num_lines'],
        "line_counts": features['line_counts']
    }


@router.post("/reference/clear")
async def clear_reference_features(db: Session = Depends(get_db)):
    """Clear all features from reference (reset to empty state)."""
    import numpy as np
    from image_processing.utils import image_to_bytes
    
    ref = db.query(ReferenceImage).first()
    if not ref:
        raise HTTPException(status_code=404, detail="No reference image found")
    
    # Reset features to empty
    features = {
        'num_lines': 0,
        'lines': [],
        'line_angles': [],
        'line_lengths': [],
        'image_shape': [256, 256],
        'line_counts': {'horizontal': 0, 'vertical': 0, 'diagonal': 0, 'total': 0}
    }
    
    ref.feature_data = json.dumps(features)
    db.commit()
    
    return {
        "success": True,
        "message": "All features cleared",
        "features": 0
    }
