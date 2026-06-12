"""
Training Data Extraction Router

Handles file uploads and processing for AI training data:
- MATLAB .mat files → MAT Extractor
- OCS PNG/JPG images → OCS Extractor
- Oxford PNG images → Direct normalization (filename-based)

Stores original + processed data in database with duplicate detection.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import subprocess
import shutil
import tempfile
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)
import json
import hashlib
import re
import cv2
import numpy as np
import io
import scipy.io
from PIL import Image
import csv

from database import get_db, TrainingDataImage, ReferenceImage
from image_quality_check.contour_quality import run_quality_check
from line_normalizer import normalize_line_thickness
from services import EvaluationService
from image_processing import LineDetector

router = APIRouter(prefix="/api", tags=["training_data"])

# Temporary upload directory
UPLOAD_DIR = "/app/data/tmp/uploads"
OUTPUT_DIR = "/app/data/tmp/extracted"

# Note: Directories are created lazily when needed, not at import time
# This prevents creating ./api/data/ when modules are imported during Docker startup


def extract_patient_id(filename: str) -> str:
    """Extract patient ID from filename."""
    # Try PC pattern first (e.g., PC56, PC0460)
    pc_match = re.search(r'(PC\d+)', filename, re.IGNORECASE)
    if pc_match:
        return pc_match.group(1).upper()
    
    # Try Park pattern
    park_match = re.search(r'(Park_\d+)', filename, re.IGNORECASE)
    if park_match:
        return park_match.group(1)
    
    # Try TEAM pattern
    team_match = re.search(r'(TEAM[KD]\d+)', filename, re.IGNORECASE)
    if team_match:
        return team_match.group(1).upper()
    
    # Try Pro pattern as fallback
    pro_match = re.search(r'(Pro\d+)', filename, re.IGNORECASE)
    if pro_match:
        return pro_match.group(1)
    
    return "UNKNOWN"


def extract_task_type(filename: str) -> str:
    """Extract task type from filename."""
    filename_upper = filename.upper()
    
    if 'REFERENCE' in filename_upper:
        return 'REFERENCE'
    elif 'RECALL' in filename_upper:
        return 'RECALL'
    elif 'COPY' in filename_upper:
        return 'COPY'
    
    return 'UNKNOWN'


def parse_oxford_filename(filename: str) -> tuple:
    """
    Parse Oxford-style filename to extract patient_id and task_type.
    
    Expected format: {ID}_{COND}.png
    Examples:
        - C0078_COPY.png → ("C0078", "COPY")
        - C0078_RECALL.png → ("C0078", "RECALL")
        - Park_16_COPY.png → ("Park_16", "COPY")
    
    Args:
        filename: The image filename
    
    Returns:
        tuple: (patient_id, task_type) or (None, None) if parsing fails
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Try to split by underscore
    parts = name.split('_')
    
    if len(parts) < 2:
        return (None, None)
    
    # Last part should be COPY or RECALL
    task_type = parts[-1].upper()
    if task_type not in ['COPY', 'RECALL']:
        return (None, None)
    
    # Everything before last underscore is patient_id
    patient_id = '_'.join(parts[:-1])
    
    return (patient_id, task_type)


def normalize_oxford_image_data(image_data: np.ndarray, target_size=(568, 274)) -> tuple:
    """
    Normalize Oxford-style PNG image data.
    
    Process:
    1. Auto-crop to content (5px padding)
    2. Resize to target size (568×274)
    3. Normalize line thickness to 2.00px
    
    Args:
        image_data: RGB numpy array (H×W×3)
        target_size: Target resolution (width, height)
    
    Returns:
        tuple: (normalized_rgb_array, success_bool)
    """
    try:
        # Step 1: Calculate bounding box and crop
        threshold = 250
        padding = 5
        
        # Convert to grayscale for bbox calculation
        if len(image_data.shape) == 3:
            gray = np.mean(image_data, axis=2)
        else:
            gray = image_data
        
        # Find non-white pixels
        content_mask = gray < threshold
        rows = np.any(content_mask, axis=1)
        cols = np.any(content_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return (None, False)
        
        min_y, max_y = np.where(rows)[0][[0, -1]]
        min_x, max_x = np.where(cols)[0][[0, -1]]
        
        # Add padding
        min_x = max(0, min_x - padding)
        max_x = min(image_data.shape[1] - 1, max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(image_data.shape[0] - 1, max_y + padding)
        
        # Crop
        cropped = image_data[min_y:max_y+1, min_x:max_x+1]
        
        # Step 2: Resize to target size
        pil_image = Image.fromarray(cropped)
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        resized = np.array(pil_image)
        
        # Step 3: Normalize line thickness to 2.00px
        normalized = normalize_line_thickness(resized, target_thickness=2.0)
        
        return (normalized, True)
        
    except Exception as e:
        logger.error(f"Error normalizing image: {e}", exc_info=True)
        return (None, False)


@router.post("/extract-training-data")
async def extract_training_data(
    format: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    raise HTTPException(
        status_code=410,
        detail=("Bulk training-data import was retired. The single import path is "
                "api/data_consolidation/import_unified.py (reads templates/labels.csv + img/). "
                "See templates/README.md."),
    )
    # Legacy implementation below is unreachable (retired 2026-06-12).
    """
    Extract training data from uploaded files and save to database.
    
    Args:
        format: 'mat' or 'ocs'
        files: List of uploaded files
        db: Database session
    
    Returns:
        {
            "success": True,
            "session_id": "...",
            "results": [
                {
                    "original_filename": "...",
                    "status": "success|duplicate|error",
                    "message": "...",
                    "extracted_images": [
                        {
                            "id": 123,
                            "patient_id": "PC56",
                            "task_type": "COPY",
                            "filename": "PC56_COPY_drawn_20251111.png"
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    """
    if format not in ['mat', 'ocs']:
        raise HTTPException(status_code=400, detail="Invalid format. Must be 'mat' or 'ocs'")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Create unique session (directories created lazily here, not at module import)
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    
    # Create base directories and session directories
    os.makedirs(session_upload_dir, exist_ok=True)
    os.makedirs(session_output_dir, exist_ok=True)
    
    results = []
    
    try:
        # Process each file
        for uploaded_file in files:
            result = {
                "original_filename": uploaded_file.filename,
                "status": "processing",
                "message": "",
                "extracted_images": []
            }
            
            try:
                # Read original file
                original_content = await uploaded_file.read()
                
                # Calculate original file hash
                original_file_hash = hashlib.sha256(original_content).hexdigest()
                
                # For MAT files: Check if this exact .mat file was already uploaded
                # For OCS files: Check duplicate now (since it's one image per file)
                if format == 'mat':
                    # Check if any image from this MAT file already exists
                    existing = db.query(TrainingDataImage).filter(
                        TrainingDataImage.source_format == 'MAT',
                        TrainingDataImage.original_filename == uploaded_file.filename
                    ).first()
                    
                    if existing:
                        # Check if it's really the same file (not just same name)
                        existing_hash = hashlib.sha256(existing.original_file_data).hexdigest()
                        if existing_hash == original_file_hash:
                            result["status"] = "duplicate"
                            result["message"] = f"MAT file already processed (ID #{existing.id})"
                            result["existing_id"] = existing.id
                            results.append(result)
                            continue
                
                elif format == 'ocs':
                    # Check for duplicates by file hash
                    existing = db.query(TrainingDataImage).filter(
                        TrainingDataImage.image_hash == original_file_hash
                    ).first()
                    
                    if existing:
                        result["status"] = "duplicate"
                        result["message"] = f"Duplicate of image #{existing.id} uploaded at {existing.uploaded_at}"
                        result["existing_id"] = existing.id
                        results.append(result)
                        continue
                
                # Save file temporarily
                file_path = os.path.join(session_upload_dir, uploaded_file.filename)
                with open(file_path, 'wb') as f:
                    f.write(original_content)
                
                # Run appropriate extractor on this single file
                file_output_dir = os.path.join(session_output_dir, f"file_{len(results)}")
                os.makedirs(file_output_dir, exist_ok=True)
                
                # Create temp dir with just this file
                single_file_input_dir = os.path.join(session_upload_dir, f"single_{len(results)}")
                os.makedirs(single_file_input_dir, exist_ok=True)
                shutil.copy(file_path, single_file_input_dir)
                
                if format == 'mat':
                    success = await run_mat_extractor(single_file_input_dir, file_output_dir)
                else:
                    success = await run_ocs_extractor(single_file_input_dir, file_output_dir)
                
                if not success:
                    result["status"] = "error"
                    result["message"] = "Extraction failed"
                    results.append(result)
                    continue
                
                # Process extracted images and save to DB
                extracted_files = [f for f in os.listdir(file_output_dir) if f.endswith('.png')]
                
                if not extracted_files:
                    result["status"] = "error"
                    result["message"] = "No images extracted"
                    results.append(result)
                    continue
                
                for extracted_filename in extracted_files:
                    extracted_path = os.path.join(file_output_dir, extracted_filename)
                    
                    # Extract metadata
                    patient_id = extract_patient_id(extracted_filename)
                    task_type = extract_task_type(extracted_filename)
                    
                    # Skip REFERENCE images from MAT files (we only want COPY and RECALL)
                    if format == 'mat' and task_type == 'REFERENCE':
                        continue
                    
                    # Read extracted image
                    with open(extracted_path, 'rb') as f:
                        processed_content = f.read()
                    
                    # For MAT files: Calculate hash from processed image (each drawing is unique)
                    # For OCS files: Use original file hash
                    if format == 'mat':
                        # Each COPY/RECALL is a separate dataset with unique hash
                        image_hash = hashlib.sha256(processed_content).hexdigest()
                    else:
                        # OCS: Use original file hash
                        image_hash = original_file_hash
                    
                    # Check for duplicates of this specific image
                    existing = db.query(TrainingDataImage).filter(
                        TrainingDataImage.image_hash == image_hash
                    ).first()
                    
                    if existing:
                        # Skip this specific image (but continue with others)
                        logger.info(f"Skipping duplicate: {extracted_filename} (ID {existing.id})")
                        continue
                    
                    # Get image dimensions
                    img_array = cv2.imdecode(np.frombuffer(processed_content, np.uint8), cv2.IMREAD_COLOR)
                    height, width = img_array.shape[:2]
                    
                    # Create metadata
                    metadata = {
                        "width": width,
                        "height": height,
                        "line_thickness": 2.0,
                        "auto_crop": True,
                        "padding_px": 5,
                        "extracted_filename": extracted_filename
                    }
                    
                    # Save to database
                    training_image = TrainingDataImage(
                        patient_id=patient_id,
                        task_type=task_type,
                        source_format=format.upper(),
                        original_filename=uploaded_file.filename,
                        original_file_data=original_content,
                        processed_image_data=processed_content,
                        image_hash=image_hash,
                        extraction_metadata=json.dumps(metadata),
                        session_id=session_id
                    )
                    
                    db.add(training_image)
                    db.commit()
                    db.refresh(training_image)
                    
                    result["extracted_images"].append({
                        "id": training_image.id,
                        "patient_id": patient_id,
                        "task_type": task_type,
                        "filename": extracted_filename,
                        "width": width,
                        "height": height
                    })
                
                result["status"] = "success"
                num_saved = len(result["extracted_images"])
                result["message"] = f"Extracted and saved {num_saved} image(s)"
                if format == 'mat':
                    result["message"] += " (COPY + RECALL only)"
                
            except Exception as e:
                result["status"] = "error"
                result["message"] = str(e)
            
            results.append(result)
        
        # Count statistics
        success_count = sum(1 for r in results if r["status"] == "success")
        duplicate_count = sum(1 for r in results if r["status"] == "duplicate")
        error_count = sum(1 for r in results if r["status"] == "error")
        total_extracted = sum(len(r.get("extracted_images", [])) for r in results)
        
        return {
            "success": True,
            "session_id": session_id,
            "statistics": {
                "total_files": len(files),
                "success": success_count,
                "duplicates": duplicate_count,
                "errors": error_count,
                "total_images_extracted": total_extracted
            },
            "results": results
        }
        
    except Exception as e:
        # Clean up on error
        cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Clean up temporary files (keep session_output_dir for now, will be cleaned by cleanup job)
        try:
            if os.path.exists(session_upload_dir):
                shutil.rmtree(session_upload_dir)
        except:
            pass
        
        # Also clean up any loose PNG files in tmp root (from direct extractor runs)
        try:
            tmp_root = "/app/data/tmp"
            for file in os.listdir(tmp_root):
                file_path = os.path.join(tmp_root, file)
                # Only delete PNG files directly in tmp (not in subdirectories)
                if os.path.isfile(file_path) and file.endswith('.png'):
                    os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Could not clean tmp root: {e}")


async def run_mat_extractor(input_dir: str, output_dir: str) -> bool:
    """Run MAT extractor on uploaded files."""
    try:
        cmd = [
            'python3',
            '/app/mat_extraction/mat_extractor.py',
            '--input', input_dir,
            '--output', output_dir,
            '--config', '/app/mat_extraction/mat_extractor.conf'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running MAT extractor: {e}", exc_info=True)
        return False


async def run_ocs_extractor(input_dir: str, output_dir: str) -> bool:
    """Run OCS extractor on uploaded files."""
    try:
        cmd = [
            'python3',
            '/app/ocs_extraction/ocs_extractor.py',
            '--input', input_dir,
            '--output', output_dir,
            '--config', '/app/ocs_extraction/ocs_extractor.conf'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running OCS extractor: {e}", exc_info=True)
        return False


@router.post("/extract-training-data-oxford")
async def extract_oxford_data(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    raise HTTPException(
        status_code=410,
        detail=("Bulk training-data import was retired. The single import path is "
                "api/data_consolidation/import_unified.py (reads templates/labels.csv + img/). "
                "See templates/README.md."),
    )
    # Legacy implementation below is unreachable (retired 2026-06-12).
    """
    Extract Oxford-style PNG images and save to database.
    
    Expected filename format: {ID}_{COND}.png
    Examples: C0078_COPY.png, C0078_RECALL.png, Park_16_COPY.png
    
    Process:
    1. Parse patient_id and task_type from filename
    2. Normalize image (auto-crop, resize to 568×274, line thickness 2px)
    3. Check for duplicates
    4. Save to database with source_format='OXFORD'
    
    Args:
        files: List of uploaded PNG files
        db: Database session
    
    Returns:
        {
            "success": True,
            "session_id": "...",
            "results": [
                {
                    "original_filename": "...",
                    "status": "success|duplicate|error",
                    "message": "...",
                    "extracted_images": [...]
                },
                ...
            ]
        }
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Create unique session
    session_id = f"oxford_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    results = []
    
    try:
        # Process each file
        for uploaded_file in files:
            result = {
                "original_filename": uploaded_file.filename,
                "status": "pending",
                "message": "",
                "extracted_images": []
            }
            
            try:
                # Read file
                original_content = await uploaded_file.read()
                
                # Calculate hash of ORIGINAL file for duplicate detection (before normalization)
                original_file_hash = hashlib.sha256(original_content).hexdigest()
                
                # Check for duplicates by original file hash
                existing = db.query(TrainingDataImage).filter(
                    TrainingDataImage.image_hash == original_file_hash
                ).first()
                
                if existing:
                    result["status"] = "duplicate"
                    result["message"] = f"Duplicate of image #{existing.id} (patient_id={existing.patient_id}, uploaded at {existing.uploaded_at})"
                    result["existing_id"] = existing.id
                    results.append(result)
                    continue
                
                # Parse filename
                patient_id, task_type = parse_oxford_filename(uploaded_file.filename)
                
                if not patient_id or not task_type:
                    result["status"] = "error"
                    result["message"] = f"Invalid filename format. Expected: {{ID}}_{{COPY|RECALL}}.png (e.g., C0078_COPY.png)"
                    results.append(result)
                    continue
                
                # Load image
                image = Image.open(io.BytesIO(original_content))
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_data = np.array(image)
                
                # Normalize image
                normalized_data, success = normalize_oxford_image_data(image_data, target_size=(568, 274))
                
                if not success or normalized_data is None:
                    result["status"] = "error"
                    result["message"] = "Failed to normalize image (empty or invalid content)"
                    results.append(result)
                    continue
                
                # Convert normalized data to PNG bytes
                normalized_pil = Image.fromarray(normalized_data)
                processed_buffer = io.BytesIO()
                normalized_pil.save(processed_buffer, format='PNG')
                processed_data = processed_buffer.getvalue()
                
                # Create extraction metadata
                extraction_metadata = {
                    "width": 568,
                    "height": 274,
                    "line_thickness": 2.0,
                    "auto_crop": True,
                    "padding_px": 5,
                    "normalization_method": "Zhang-Suen + dilation",
                    "source": "Oxford-style PNG (UI upload)",
                    "original_resolution": f"{image.width}×{image.height}",
                    "original_file_size": len(original_content),
                    "processed_file_size": len(processed_data)
                }
                
                # Create database entry
                training_image = TrainingDataImage(
                    patient_id=patient_id,
                    task_type=task_type,
                    source_format="OXFORD",
                    original_filename=uploaded_file.filename,
                    original_file_data=original_content,
                    processed_image_data=processed_data,
                    image_hash=original_file_hash,  # Use hash of ORIGINAL file for duplicate detection
                    extraction_metadata=json.dumps(extraction_metadata),
                    features_data=json.dumps({}),  # Empty for now, can be filled later
                    session_id=session_id
                )
                
                db.add(training_image)
                db.commit()
                db.refresh(training_image)
                
                result["status"] = "success"
                result["message"] = "Successfully extracted and saved to database"
                result["extracted_images"] = [{
                    "id": training_image.id,
                    "patient_id": patient_id,
                    "task_type": task_type,
                    "filename": uploaded_file.filename
                }]
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.filename}: {e}", exc_info=True)
                result["status"] = "error"
                result["message"] = f"Error processing file: {str(e)}"
                results.append(result)
        
        # Count statistics
        success_count = sum(1 for r in results if r["status"] == "success")
        duplicate_count = sum(1 for r in results if r["status"] == "duplicate")
        error_count = sum(1 for r in results if r["status"] == "error")
        total_extracted = sum(len(r.get("extracted_images", [])) for r in results)
        
        return {
            "success": True,
            "session_id": session_id,
            "statistics": {
                "total_files": len(files),
                "success": success_count,
                "duplicates": duplicate_count,
                "errors": error_count,
                "total_images_extracted": total_extracted
            },
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


@router.get("/training-data-images")
async def get_training_data_images(
    limit: int = 100,
    offset: int = 0,
    patient_id: str = None,
    task_type: str = None,
    source_format: str = None,
    search: str = None,
    only_missing: bool = False,
    quality_check_failed: bool = False,
    ids: str = None,
    db: Session = Depends(get_db)
):
    """
    Get list of training data images from database.
    
    Args:
        limit: Maximum number of results
        offset: Offset for pagination
        patient_id: Filter by patient ID
        task_type: Filter by task type
        source_format: Filter by source format
        search: Search term (filters ID, patient_id, task_type, source_format, filename)
        only_missing: If true, only return images without features
        quality_check_failed: If true, only return images that failed quality check
        ids: Comma-separated list of image IDs to filter by (e.g., "122,123,456")
        db: Database session
    
    Returns:
        List of training data images with metadata
    """
    from sqlalchemy import or_, cast, String, func
    from sqlalchemy.orm import load_only
    
    # Build filter conditions (shared between count and fetch queries)
    filters = []
    
    # Filter by specific IDs (takes priority)
    if ids and ids.strip():
        try:
            id_list = [int(id.strip()) for id in ids.split(',') if id.strip()]
            if id_list:
                filters.append(TrainingDataImage.id.in_(id_list))
        except ValueError:
            pass  # Invalid IDs, ignore filter
    
    if patient_id:
        filters.append(TrainingDataImage.patient_id == patient_id)
    if task_type:
        filters.append(TrainingDataImage.task_type == task_type)
    if source_format:
        filters.append(TrainingDataImage.source_format == source_format)
    
    # Search filter - match across multiple fields
    if search and search.strip():
        search_term = f"%{search.strip()}%"
        filters.append(
            or_(
                cast(TrainingDataImage.id, String).ilike(search_term),
                TrainingDataImage.patient_id.ilike(search_term),
                TrainingDataImage.task_type.ilike(search_term),
                TrainingDataImage.source_format.ilike(search_term),
                TrainingDataImage.original_filename.ilike(search_term),
            )
        )
    
    # Only missing features filter
    if only_missing:
        filters.append(
            or_(
                TrainingDataImage.features_data.is_(None),
                TrainingDataImage.features_data == '{}',
                TrainingDataImage.features_data == 'null',
                TrainingDataImage.features_data == '',
            )
        )
    
    # Quality check failed filter
    if quality_check_failed:
        filters.append(TrainingDataImage.quality_check_status == "invalid")
    
    # Get total count using func.count (fast - doesn't load data)
    count_query = db.query(func.count(TrainingDataImage.id))
    for f in filters:
        count_query = count_query.filter(f)
    total = count_query.scalar()
    
    # Fetch data using load_only to exclude BLOB columns (critical for performance)
    # Without this, SQLite loads ALL image data (~316MB) even for metadata-only queries
    query = db.query(TrainingDataImage).options(
        load_only(
            TrainingDataImage.id,
            TrainingDataImage.patient_id,
            TrainingDataImage.task_type,
            TrainingDataImage.source_format,
            TrainingDataImage.original_filename,
            TrainingDataImage.test_name,
            TrainingDataImage.extraction_metadata,
            TrainingDataImage.features_data,
            TrainingDataImage.uploaded_at,
            TrainingDataImage.session_id,
            TrainingDataImage.ground_truth_correct,
            TrainingDataImage.ground_truth_extra,
            TrainingDataImage.quality_check_status,
            TrainingDataImage.quality_check_date
        )
    )
    for f in filters:
        query = query.filter(f)
    query = query.order_by(TrainingDataImage.uploaded_at.desc())
    images = query.offset(offset).limit(limit).all()
    
    results = []
    for img in images:
        metadata = json.loads(img.extraction_metadata) if img.extraction_metadata else {}
        has_features = bool(img.features_data and img.features_data != '{}' and img.features_data != 'null')
        
        results.append({
            "id": img.id,
            "patient_id": img.patient_id,
            "task_type": img.task_type,
            "source_format": img.source_format,
            "original_filename": img.original_filename,
            "test_name": img.test_name,
            "width": metadata.get("width", 568),   # processed images are always 568×274
            "height": metadata.get("height", 274),
            "uploaded_at": img.uploaded_at.isoformat(),
            "session_id": img.session_id,
            "has_features": has_features,
            "quality_check_status": img.quality_check_status,
            "quality_check_date": img.quality_check_date.isoformat() if img.quality_check_date else None
        })
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "images": results
    }


@router.get("/training-data-stats")
def get_training_data_stats(db: Session = Depends(get_db)):
    """Get aggregated statistics using fast ORM count query."""
    from sqlalchemy import func
    
    # Single fast count query (avoid multiple queries that cause blocking)
    total = db.query(func.count(TrainingDataImage.id)).scalar() or 0
    
    return {
        "total": total,
        "by_source": {"MAT": 0, "OCS": 0, "OXFORD": 0, "DRAWN": 0},
        "patients": 0,
        "with_features": 0,
        "without_features": total,
        "quality": {"valid": 0, "invalid": 0, "unchecked": total}
    }


# Global state for quality check background job
quality_check_job = {
    "running": False,
    "progress": 0,
    "total": 0,
    "checked": 0,
    "invalid": 0,
    "valid": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def _run_quality_check_background(recheck_invalid_only: bool = False):
    """Background task to run quality check and update DB."""
    import threading
    from database import SessionLocal
    from image_quality_check.contour_quality import analyze_image, preprocess_original_image
    
    global quality_check_job
    
    quality_check_job["running"] = True
    quality_check_job["started_at"] = datetime.now().isoformat()
    quality_check_job["finished_at"] = None
    quality_check_job["error"] = None
    quality_check_job["checked"] = 0
    quality_check_job["invalid"] = 0
    quality_check_job["valid"] = 0
    
    # Quality check parameters
    red_threshold = {"r_min": 150, "g_max": 100, "b_max": 100}
    
    try:
        db = SessionLocal()
        
        # Get entries to check
        query = db.query(TrainingDataImage)
        if recheck_invalid_only:
            # Only check entries that are NULL or invalid
            from sqlalchemy import or_
            query = query.filter(
                or_(
                    TrainingDataImage.quality_check_status.is_(None),
                    TrainingDataImage.quality_check_status == "invalid"
                )
            )
        
        entries = query.all()
        quality_check_job["total"] = len(entries)
        
        logger.info(f"Quality check background job started: {len(entries)} entries to check")
        
        for i, entry in enumerate(entries):
            try:
                # Use original image data
                image_bytes = entry.original_file_data
                if not image_bytes:
                    image_bytes = entry.processed_image_data
                
                if not image_bytes:
                    continue
                
                # Analyze image with all parameters
                result = analyze_image(
                    image_bytes=image_bytes,
                    use_original=True,
                    red_threshold=red_threshold,
                    min_component_area=100,
                    min_component_ratio=0.01,
                    min_gap_px=80,
                    merge_kernel=5,
                    merge_iterations=2,
                    peak_threshold_ratio=0.1,
                    min_peak_separation=9999,  # Disable projection-based check
                    outside_margin_ratio=0.05,
                    outside_ink_ratio=0.08,
                    min_contour_area=1,
                    contour_only=False,
                    require_gap_and_outside=True,
                    min_component_height_ratio=0.1,
                    require_contours=True,
                    min_contour_count=2,
                )
                
                # Update entry
                is_flagged = result.get("flagged", False)
                entry.quality_check_status = "invalid" if is_flagged else "valid"
                entry.quality_check_date = datetime.now()
                
                if is_flagged:
                    quality_check_job["invalid"] += 1
                else:
                    quality_check_job["valid"] += 1
                
                quality_check_job["checked"] = i + 1
                quality_check_job["progress"] = int((i + 1) / len(entries) * 100)
                
                # Commit in batches so DB is updated while running
                if (i + 1) % 50 == 0:
                    db.commit()
                
            except Exception as e:
                logger.warning(f"Error checking image {entry.id}: {e}")
                continue
        
        db.commit()
        db.close()
        
        quality_check_job["finished_at"] = datetime.now().isoformat()
        logger.info(f"Quality check complete: {quality_check_job['invalid']} invalid, {quality_check_job['valid']} valid")
        
    except Exception as e:
        logger.error(f"Quality check background job error: {e}", exc_info=True)
        quality_check_job["error"] = str(e)
    finally:
        quality_check_job["running"] = False


@router.post("/training-data-image-quality-check/start")
async def start_quality_check(recheck_invalid_only: bool = True):
    """
    Start background quality check job.
    
    Args:
        recheck_invalid_only: If true, only check entries with status=NULL or status="invalid"
    """
    import threading
    
    global quality_check_job
    
    if quality_check_job["running"]:
        return {
            "success": False,
            "message": "Quality check already running",
            "status": quality_check_job,
        }
    
    # Start background thread
    thread = threading.Thread(
        target=_run_quality_check_background,
        args=(recheck_invalid_only,),
        daemon=True
    )
    thread.start()
    
    return {
        "success": True,
        "message": "Quality check started",
        "recheck_invalid_only": recheck_invalid_only,
    }


@router.get("/training-data-image-quality-check/status")
async def get_quality_check_status():
    """Get current status of quality check background job."""
    return quality_check_job


@router.get("/training-data-image/{image_id}/original")
async def get_training_data_original(image_id: int, db: Session = Depends(get_db)):
    """Serve original uploaded file or preview image for MAT files."""
    img = db.query(TrainingDataImage).filter(TrainingDataImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # For MAT files: Extract unoptimized drawing from .mat file
    if img.source_format == 'MAT':
        try:
            import scipy.io
            from PIL import Image, ImageDraw
            import tempfile
            
            # Save MAT file to temp location
            with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
                tmp.write(img.original_file_data)
                tmp_path = tmp.name
            
            # Load MAT data
            mat_data = scipy.io.loadmat(tmp_path)
            
            # Determine which key to use based on task_type
            if img.task_type == 'COPY':
                key_prefix = 'data_complex_copy'
            elif img.task_type == 'RECALL':
                key_prefix = 'data_complex_memory_copy'
            else:
                raise Exception(f"Unknown task type: {img.task_type}")
            
            # Extract drawing lines (BEFORE optimization)
            if key_prefix in mat_data:
                data = mat_data[key_prefix][0, 0]
                
                # Get drawing area rect
                draw_area = data['draw_area'][0, 0]
                rect_data = draw_area['rect']
                if isinstance(rect_data, np.ndarray) and rect_data.shape == (1, 4):
                    rect = rect_data[0]
                else:
                    rect = np.array([0, 0, 568, 568])
                
                x1, y1, x2, y2 = rect
                original_width = int(x2 - x1)
                original_height = int(y2 - y1)
                
                # Extract trails
                trails = data['trails'][0, 0]
                cont_lines_array = trails['cont_lines']
                
                # Create canvas at ORIGINAL size (no optimization)
                canvas = Image.new('RGB', (original_width, original_height), (255, 255, 255))
                draw = ImageDraw.Draw(canvas)
                
                # Draw all lines at ORIGINAL thickness
                num_lines = cont_lines_array.shape[1]
                for i in range(num_lines):
                    line = cont_lines_array[0, i]
                    if isinstance(line, np.ndarray) and line.shape[0] >= 2:
                        # Convert points to canvas coordinates
                        points = []
                        for j in range(line.shape[0]):
                            x = int(line[j, 0] - x1)
                            y = int(line[j, 1] - y1)
                            points.append((x, y))
                        
                        if len(points) >= 2:
                            draw.line(points, fill=(0, 0, 0), width=2)
                
                # Resize to 568×274 with padding (no stretch)
                scale = min(568 / original_width, 274 / original_height)
                new_w = int(original_width * scale)
                new_h = int(original_height * scale)
                
                canvas = canvas.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Center on 568×274 canvas
                final_canvas = Image.new('RGB', (568, 274), (255, 255, 255))
                offset_x = (568 - new_w) // 2
                offset_y = (274 - new_h) // 2
                final_canvas.paste(canvas, (offset_x, offset_y))
                
                # Return as PNG
                img_io = io.BytesIO()
                final_canvas.save(img_io, 'PNG')
                img_io.seek(0)
                
                os.unlink(tmp_path)
                
                return StreamingResponse(
                    img_io,
                    media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"}
                )
            
            os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Error extracting MAT original drawing: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            # Fallback: Return processed image
            return StreamingResponse(
                io.BytesIO(img.processed_image_data),
                media_type="image/png"
            )
    
    # For OCS files: Return actual original
    content_type = "application/octet-stream"
    if img.original_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        content_type = f"image/{img.original_filename.split('.')[-1].lower()}"
    
    return StreamingResponse(
        io.BytesIO(img.original_file_data),
        media_type=content_type,
        headers={"Content-Disposition": f"inline; filename={img.original_filename}"}
    )


@router.get("/training-data-image/{image_id}/processed")
async def get_training_data_processed(image_id: int, db: Session = Depends(get_db)):
    """Serve processed/extracted image."""
    img = db.query(TrainingDataImage).filter(TrainingDataImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return StreamingResponse(
        io.BytesIO(img.processed_image_data),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"}
    )


@router.delete("/training-data-image/{image_id}")
async def delete_training_data_image(image_id: int, db: Session = Depends(get_db)):
    """Delete a training data image."""
    img = db.query(TrainingDataImage).filter(TrainingDataImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    db.delete(img)
    db.commit()
    
    return {"success": True}


@router.post("/training-data-image/{image_id}/quality-status")
async def update_quality_status(image_id: int, data: dict, db: Session = Depends(get_db)):
    """
    Manually update the quality check status of an image.
    
    Args:
        image_id: Image ID
        data: Dictionary with 'status' ("valid" or "invalid")
    """
    img = db.query(TrainingDataImage).filter(TrainingDataImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    status = data.get('status')
    if status not in ('valid', 'invalid'):
        raise HTTPException(status_code=400, detail="Status must be 'valid' or 'invalid'")
    
    img.quality_check_status = status
    img.quality_check_date = datetime.now()
    
    db.commit()
    
    logger.info(f"Quality status for image {image_id} manually set to '{status}'")
    
    return {
        "success": True,
        "image_id": img.id,
        "quality_check_status": img.quality_check_status,
        "quality_check_date": img.quality_check_date.isoformat()
    }


@router.get("/training-data-image/{image_id}/features")
async def get_training_data_features(image_id: int, db: Session = Depends(get_db)):
    """Get features/labels for a training data image."""
    from sqlalchemy.orm import load_only
    
    # Only load the columns we need (exclude BLOBs for performance)
    img = db.query(TrainingDataImage).options(
        load_only(
            TrainingDataImage.id,
            TrainingDataImage.patient_id,
            TrainingDataImage.task_type,
            TrainingDataImage.features_data
        )
    ).filter(TrainingDataImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if img.features_data:
        features = json.loads(img.features_data)
    else:
        features = {}
    
    return {
        "image_id": img.id,
        "patient_id": img.patient_id,
        "task_type": img.task_type,
        "features": features,
        "has_features": bool(img.features_data and img.features_data != '{}')
    }


@router.post("/training-data-image/{image_id}/features")
async def update_training_data_features(
    image_id: int,
    features: dict,
    db: Session = Depends(get_db)
):
    """Update features/labels for a training data image."""
    img = db.query(TrainingDataImage).filter(TrainingDataImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Store features as JSON
    img.features_data = json.dumps(features)
    db.commit()
    
    return {
        "success": True,
        "image_id": img.id,
        "features": features
    }


@router.delete("/training-data-image/{image_id}/features")
async def delete_training_data_features(image_id: int, db: Session = Depends(get_db)):
    """Delete all features/labels for a training data image."""
    img = db.query(TrainingDataImage).filter(TrainingDataImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    img.features_data = None
    db.commit()
    
    return {"success": True}


@router.post("/training-data-image/{image_id}/ground-truth")
async def update_ground_truth(
    image_id: int,
    data: dict,
    db: Session = Depends(get_db)
):
    """
    Update ground truth values for a training data image.
    
    Args:
        image_id: Image ID
        data: Dictionary with ground_truth_correct and ground_truth_extra
        
    Returns:
        Success status and updated values
    """
    img = db.query(TrainingDataImage).filter(TrainingDataImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Update ground truth values
    img.ground_truth_correct = data.get('ground_truth_correct')
    img.ground_truth_extra = data.get('ground_truth_extra')
    
    db.commit()
    db.refresh(img)
    
    return {
        "success": True,
        "image_id": img.id,
        "ground_truth_correct": img.ground_truth_correct,
        "ground_truth_extra": img.ground_truth_extra
    }


@router.post("/training-data-image/{image_id}/crop-and-reprocess")
async def crop_and_reprocess_image(
    image_id: int,
    data: dict,
    db: Session = Depends(get_db)
):
    """
    Crop the original image and re-run the optimization pipeline.
    
    Args:
        image_id: Image ID
        data: Dictionary with crop coordinates {x1, y1, x2, y2} (pixels in original image)
        
    Returns:
        Success status and new processed image dimensions
    """
    from ocs_extraction.ocs_extractor import normalize_line_thickness as ocs_normalize
    
    img = db.query(TrainingDataImage).filter(TrainingDataImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not img.original_file_data:
        raise HTTPException(status_code=400, detail="No original image data available")
    
    # Get crop coordinates
    x1 = int(data.get('x1', 0))
    y1 = int(data.get('y1', 0))
    x2 = int(data.get('x2', 0))
    y2 = int(data.get('y2', 0))
    
    if x1 >= x2 or y1 >= y2:
        raise HTTPException(status_code=400, detail="Invalid crop coordinates")
    
    try:
        # Load original image
        original_img = Image.open(io.BytesIO(img.original_file_data))
        if original_img.mode == 'RGBA':
            background = Image.new('RGB', original_img.size, (255, 255, 255))
            background.paste(original_img, mask=original_img.split()[3])
            original_img = background
        elif original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        # Crop to specified region
        cropped = original_img.crop((x1, y1, x2, y2))
        cropped_array = np.array(cropped)
        
        # Check for red pixels
        r = cropped_array[:, :, 0]
        g = cropped_array[:, :, 1]
        b = cropped_array[:, :, 2]
        
        red_threshold = {'r_min': 150, 'g_max': 100, 'b_max': 100}
        red_mask = (r >= red_threshold['r_min']) & \
                   (g <= red_threshold['g_max']) & \
                   (b <= red_threshold['b_max'])
        
        has_red = np.any(red_mask)
        
        if has_red:
            # Extract only red pixels → render as black on white
            height, width = red_mask.shape
            content_mask = red_mask
            processed_array = np.ones((height, width, 3), dtype=np.uint8) * 255
            processed_array[red_mask] = [0, 0, 0]
        else:
            # Use grayscale content (black/dark pixels)
            gray = cv2.cvtColor(cropped_array, cv2.COLOR_RGB2GRAY)
            # Threshold: pixels darker than 180 become black (lines)
            _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            content_mask = binary < 128  # Where the lines are
            processed_array = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # Auto-crop to content bounding box with padding (like new images)
        padding = 5
        if np.any(content_mask):
            rows = np.any(content_mask, axis=1)
            cols = np.any(content_mask, axis=0)
            min_y, max_y = np.where(rows)[0][[0, -1]]
            min_x, max_x = np.where(cols)[0][[0, -1]]
            
            # Add padding
            h, w = content_mask.shape
            min_x = max(0, min_x - padding)
            max_x = min(w - 1, max_x + padding)
            min_y = max(0, min_y - padding)
            max_y = min(h - 1, max_y + padding)
            
            # Crop to bounding box
            processed_array = processed_array[min_y:max_y+1, min_x:max_x+1]
        
        # Resize to standard canvas size (568x274) - stretch to fill
        canvas_size = (568, 274)
        processed_img = Image.fromarray(processed_array, mode='RGB')
        processed_img = processed_img.resize(canvas_size, Image.Resampling.LANCZOS)
        
        # Binarize after resize to ensure clean black/white
        processed_array = np.array(processed_img)
        gray = cv2.cvtColor(processed_array, cv2.COLOR_RGB2GRAY)
        _, binary_clean = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        processed_array = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2RGB)
        
        # Normalize line thickness to exactly 2px
        normalized = ocs_normalize(processed_array, target_thickness=2, threshold=200)
        processed_img = Image.fromarray(normalized, mode='RGB')
        
        # Save to bytes
        buffer = io.BytesIO()
        processed_img.save(buffer, format='PNG')
        processed_bytes = buffer.getvalue()
        
        # Update database
        img.processed_image_data = processed_bytes
        
        # Update metadata
        metadata = json.loads(img.extraction_metadata) if img.extraction_metadata else {}
        metadata['width'] = canvas_size[0]
        metadata['height'] = canvas_size[1]
        metadata['crop_applied'] = {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'timestamp': datetime.now().isoformat()
        }
        img.extraction_metadata = json.dumps(metadata)
        
        db.commit()
        
        logger.info(f"Crop and reprocess successful for image {image_id}: crop=({x1},{y1})-({x2},{y2})")
        
        return {
            "success": True,
            "image_id": img.id,
            "width": canvas_size[0],
            "height": canvas_size[1],
            "red_pixels_extracted": bool(has_red),  # Convert numpy.bool_ to Python bool
            "crop": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }
        
    except Exception as e:
        logger.error(f"Error cropping image {image_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.get("/training-data-features-template")
async def download_features_template(db: Session = Depends(get_db)):
    """
    Generate CSV template with all training data entries for bulk feature upload.
    
    Returns:
        CSV file with columns: Patient, Task, Total_Score, Data_Quality
    """
    from sqlalchemy.orm import load_only
    
    # Only load columns needed for CSV (exclude BLOBs for performance)
    images = db.query(TrainingDataImage).options(
        load_only(
            TrainingDataImage.patient_id,
            TrainingDataImage.task_type,
            TrainingDataImage.features_data
        )
    ).order_by(
        TrainingDataImage.patient_id, 
        TrainingDataImage.task_type
    ).all()
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['Patient', 'Task', 'Total_Score', 'Data_Quality'])
    
    # Rows - add existing features if present
    for img in images:
        features = {}
        if img.features_data:
            try:
                features = json.loads(img.features_data)
            except:
                pass
        
        total_score = features.get('Total_Score', '')
        data_quality = features.get('Data_Quality', '')
        
        writer.writerow([
            img.patient_id,
            img.task_type,
            total_score,
            data_quality
        ])
    
    # Return as downloadable CSV
    csv_content = output.getvalue()
    output.close()
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=training_data_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )


@router.post("/training-data-features-upload")
async def upload_features_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload CSV file with features and update database.
    
    CSV Format: Patient, Task, Total_Score, Data_Quality
    Matching: Case-insensitive Patient + Task
    
    Returns:
        Update statistics
    """
    try:
        # Read CSV
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Auto-detect delimiter using csv.Sniffer (handles quoted fields correctly)
        try:
            # Sample first few lines for detection
            sample = '\n'.join(content_str.split('\n')[:5])
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample, delimiters=',;').delimiter
        except Exception:
            # Fallback to comma if detection fails
            delimiter = ','
        
        csv_reader = csv.DictReader(io.StringIO(content_str), delimiter=delimiter)
        
        updated = 0
        skipped = 0
        errors = []
        
        for row in csv_reader:
            try:
                patient = row.get('Patient', '').strip()
                task = row.get('Task', '').strip()
                total_score = row.get('Total_Score', '').strip()
                data_quality = row.get('Data_Quality', '').strip()
                
                if not patient or not task:
                    skipped += 1
                    continue
                
                # Find matching entry in DB (case-insensitive)
                img = db.query(TrainingDataImage).filter(
                    TrainingDataImage.patient_id.ilike(patient),
                    TrainingDataImage.task_type.ilike(task)
                ).first()
                
                if img:
                    # Load existing features or create new
                    features = {}
                    if img.features_data:
                        try:
                            features = json.loads(img.features_data)
                        except:
                            pass
                    
                    # Track if any feature was successfully added
                    any_success = False
                    
                    # Update features from CSV
                    if total_score:
                        try:
                            features['Total_Score'] = float(total_score)
                            any_success = True
                        except ValueError:
                            # Sanitize error message to prevent XSS
                            safe_patient = patient.replace('<', '&lt;').replace('>', '&gt;')
                            safe_task = task.replace('<', '&lt;').replace('>', '&gt;')
                            safe_score = total_score.replace('<', '&lt;').replace('>', '&gt;')
                            errors.append(f"{safe_patient}/{safe_task}: Invalid Total_Score '{safe_score}'")
                    
                    if data_quality:
                        try:
                            features['Data_Quality'] = float(data_quality)
                            any_success = True
                        except ValueError:
                            # Sanitize error message to prevent XSS
                            safe_patient = patient.replace('<', '&lt;').replace('>', '&gt;')
                            safe_task = task.replace('<', '&lt;').replace('>', '&gt;')
                            safe_quality = data_quality.replace('<', '&lt;').replace('>', '&gt;')
                            errors.append(f"{safe_patient}/{safe_task}: Invalid Data_Quality '{safe_quality}'")
                    
                    # Only save and count as updated if at least one feature was successfully parsed
                    if any_success:
                        img.features_data = json.dumps(features)
                        db.commit()
                        updated += 1
                    else:
                        # All features failed to parse - skip this row
                        skipped += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                # Sanitize exception message to prevent XSS
                safe_error = str(e).replace('<', '&lt;').replace('>', '&gt;')
                errors.append(f"Row error: {safe_error}")
        
        return {
            "success": True,
            "updated": updated,
            "skipped": skipped,
            "errors": errors,
            "total_rows": updated + skipped
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")


@router.post("/save-drawn-image")
async def save_drawn_image(
    file: UploadFile = File(...),
    name: str = Form(...),
    total_score: Optional[int] = Form(None),
    components: Optional[str] = Form(None),  # JSON {presence:[20], accuracy:[20], position:[20]}
    source_format: str = Form('DRAWN'),
    task_type: str = Form('DRAWN'),
    db: Session = Depends(get_db)
):
    """
    Save a manually drawn / uploaded image as training data.

    Optional evaluation (training format):
    - total_score: Total_Score (sum of the component sub-labels)
    - components: JSON with the 20 elements × Presence/Accuracy/Position (0/1)
    
    Source format: DRAWN (from draw tool), UPLOAD (from upload page), MAT, OCS
    Task type: DRAWN, UPLOAD, undefined, COPY, RECALL
    """
    import hashlib
    
    try:
        # Read image (original)
        raw_content = await file.read()
        image_hash = hashlib.sha256(raw_content).hexdigest()
        
        # Convert to RGB and re-save as original (for consistency)
        temp_img = Image.open(io.BytesIO(raw_content))
        if temp_img.mode == 'RGBA':
            background = Image.new('RGB', temp_img.size, (255, 255, 255))
            background.paste(temp_img, mask=temp_img.split()[3])
            temp_img = background
        elif temp_img.mode != 'RGB':
            temp_img = temp_img.convert('RGB')
        
        # Save as PNG bytes
        original_buffer = io.BytesIO()
        temp_img.save(original_buffer, format='PNG')
        content = original_buffer.getvalue()
        
        # Check for duplicate image by hash (prevent same image being saved multiple times)
        existing = db.query(TrainingDataImage).filter(
            TrainingDataImage.image_hash == image_hash
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400, 
                detail=f"Duplicate image detected! This image already exists in database (ID: {existing.id}, uploaded: {existing.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')})"
            )
        
        # Also check for duplicate name (secondary check)
        existing_name = db.query(TrainingDataImage).filter(
            TrainingDataImage.test_name == name
        ).first()
        
        if existing_name:
            raise HTTPException(status_code=400, detail=f"Name '{name}' already exists (ID: {existing_name.id})")
        
        # Load image for processing
        image = Image.open(io.BytesIO(content))
        
        # Convert to RGB if needed (canvas may send RGBA)
        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        # Step 1: Auto-crop to content with 5px padding (like MAT/OCS)
        # Find bounding box of drawn content
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Find non-white pixels
        coords = cv2.findNonZero(binary)
        
        if coords is not None:
            # Calculate bounding box with padding
            x, y, w, h = cv2.boundingRect(coords)
            padding = 5
            
            # Add padding (with bounds checking)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image_array.shape[1] - x, w + 2 * padding)
            h = min(image_array.shape[0] - y, h + 2 * padding)
            
            # Crop to bounding box
            cropped_array = image_array[y:y+h, x:x+w]
            
            # Step 2: Scale to 568×274 (preserving aspect ratio, centered)
            scale = min(568 / w, 274 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize cropped content
            cropped_img = Image.fromarray(cropped_array)
            resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center on 568×274 canvas
            final_canvas = np.ones((274, 568, 3), dtype=np.uint8) * 255
            offset_x = (568 - new_w) // 2
            offset_y = (274 - new_h) // 2
            final_canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = np.array(resized_img)
            
            image_array = final_canvas
        
        # Step 3: Normalize line thickness to 2px (CNN-ready)
        # This ensures consistency with MAT/OCS images
        normalized_array = normalize_line_thickness(image_array, target_thickness=2)
        
        # Convert normalized array back to bytes
        normalized_image = Image.fromarray(normalized_array, mode='RGB')
        normalized_buffer = io.BytesIO()
        normalized_image.save(normalized_buffer, format='PNG')
        normalized_content = normalized_buffer.getvalue()
        
        # Get final dimensions
        height, width = normalized_array.shape[:2]
        
        feats = {}
        if total_score is not None:
            feats["Total_Score"] = total_score
        if components:
            try:
                comp = json.loads(components)
                # keep only the canonical sub-label arrays
                feats["components"] = {
                    "presence": [int(x) for x in comp.get("presence", [])],
                    "accuracy": [int(x) for x in comp.get("accuracy", [])],
                    "position": [int(x) for x in comp.get("position", [])],
                }
            except (ValueError, TypeError):
                pass
        features_data = json.dumps(feats) if feats else None

        # Create entry
        training_image = TrainingDataImage(
            patient_id=name,  # Use name as patient_id
            task_type=task_type,  # Use passed task_type (DRAWN, UPLOAD, undefined, etc.)
            source_format=source_format,  # Use passed source_format (DRAWN, UPLOAD, MAT, OCS)
            original_filename=f"{name}.png",
            original_file_data=content,  # Original drawing (raw)
            processed_image_data=normalized_content,  # CNN-ready (normalized 2px lines)
            image_hash=image_hash,
            features_data=features_data,
            test_name=name,
            session_id=f'{source_format.lower()}_upload',  # e.g., 'upload_upload' or 'drawn_upload'
            extraction_metadata=json.dumps({
                "width": width,
                "height": height,
                "manually_drawn": True,
                "auto_cropped": True,
                "padding_px": 5,
                "line_thickness_normalized": True,
                "target_thickness_px": 2
            })
        )
        
        db.add(training_image)
        db.commit()
        db.refresh(training_image)
        
        return {
            "success": True,
            "id": training_image.id,
            "name": name,
            "message": "Drawing saved as training data"
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-old-sessions")
async def cleanup_old_sessions(max_age_hours: int = 24):
    """Clean up old temporary extraction directories."""
    import time
    
    # Ensure directories exist before trying to clean them
    if not os.path.exists(OUTPUT_DIR):
        return {
            "success": True,
            "sessions_cleaned": 0,
            "max_age_hours": max_age_hours,
            "message": "No temporary directories to clean"
        }
    
    cutoff_time = time.time() - (max_age_hours * 3600)
    sessions_cleaned = 0
    
    for session_id in os.listdir(OUTPUT_DIR):
        session_dir = os.path.join(OUTPUT_DIR, session_id)
        if os.path.isdir(session_dir):
            dir_mtime = os.path.getmtime(session_dir)
            if dir_mtime < cutoff_time:
                cleanup_session(session_id)
                sessions_cleaned += 1
    
    return {
        "success": True,
        "sessions_cleaned": sessions_cleaned,
        "max_age_hours": max_age_hours
    }


def cleanup_session(session_id: str):
    """Clean up all files from a session."""
    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    
    for directory in [session_upload_dir, session_output_dir]:
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
        except Exception as e:
            logger.warning(f"Error cleaning up {directory}: {e}")


@router.post("/training-data-image/{image_id}/evaluate")
async def evaluate_training_data_image(
    image_id: int,
    db: Session = Depends(get_db)
):
    """
    Evaluate a training data image by running line detection and comparing to ground truth.
    
    This endpoint:
    1. Loads the processed image from database
    2. Runs automated line detection
    3. Compares detected lines to reference
    4. Compares detected metrics to ground truth values
    5. Returns both automated and ground truth values for comparison
    
    Args:
        image_id: Training data image ID
        db: Database session
    
    Returns:
        Evaluation results with automated detection vs ground truth comparison
    """
    # Get training image
    training_img = db.query(TrainingDataImage).filter(
        TrainingDataImage.id == image_id
    ).first()
    
    if not training_img:
        raise HTTPException(status_code=404, detail="Training image not found")
    
    # Get reference image (assume default reference for now)
    reference = db.query(ReferenceImage).first()
    if not reference:
        raise HTTPException(status_code=404, detail="No reference image found")
    
    # Load processed image
    nparr = np.frombuffer(training_img.processed_image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run evaluation using EvaluationService
    eval_service = EvaluationService(db)
    
    # Use evaluate_test_image (doesn't store to DB, just returns result)
    evaluation_result = eval_service.evaluate_test_image(
        image,
        reference.id,
        f"training_{training_img.id}"
    )
    
    # Get reference line count for calculations
    line_detector = LineDetector()
    ref_features = line_detector.features_from_json(reference.feature_data)
    total_ref_lines = len(ref_features['lines'])
    
    # Calculate missing lines (reference - correct)
    # Note: evaluation_result already has missing_lines calculated correctly
    detected_correct = evaluation_result.correct_lines
    detected_missing = evaluation_result.missing_lines
    detected_extra = evaluation_result.extra_lines
    
    # Calculate automated similarity score
    automated_similarity = evaluation_result.similarity_score
    
    # If ground truth exists, calculate accuracy of detection
    ground_truth_accuracy = None
    accuracy_details = None
    
    if training_img.ground_truth_correct is not None:
        # Compare automated detection to ground truth
        gt_correct = training_img.ground_truth_correct
        gt_extra = training_img.ground_truth_extra or 0
        gt_missing = total_ref_lines - gt_correct  # Calculate expected missing
        
        # Calculate differences
        correct_diff = abs(detected_correct - gt_correct)
        missing_diff = abs(detected_missing - gt_missing)
        extra_diff = abs(detected_extra - gt_extra)
        
        # Total error (sum of absolute differences)
        total_error = correct_diff + missing_diff + extra_diff
        max_error = total_ref_lines * 3  # 3 metrics, max error = ref_lines each
        
        # Accuracy: 1.0 if perfect match, decreases with error
        ground_truth_accuracy = max(0.0, 1.0 - (total_error / max_error)) if max_error > 0 else 1.0
        
        accuracy_details = {
            "correct_diff": correct_diff,
            "missing_diff": missing_diff,
            "extra_diff": extra_diff,
            "total_error": total_error,
            "max_error": max_error
        }
    
    # Prepare response
    response = {
        "image_id": training_img.id,
        "patient_id": training_img.patient_id,
        "task_type": training_img.task_type,
        "source_format": training_img.source_format,
        "total_reference_lines": total_ref_lines,
        
        # Automated detection results
        "automated": {
            "correct_lines": detected_correct,
            "missing_lines": detected_missing,
            "extra_lines": detected_extra,
            "similarity_score": automated_similarity
        },
        
        # Ground truth (if available)
        "ground_truth": {
            "correct_lines": training_img.ground_truth_correct,
            "extra_lines": training_img.ground_truth_extra,
            "missing_lines": total_ref_lines - training_img.ground_truth_correct if training_img.ground_truth_correct is not None else None,
            "has_ground_truth": training_img.ground_truth_correct is not None
        },
        
        # Comparison metrics
        "comparison": {
            "accuracy": ground_truth_accuracy,
            "details": accuracy_details
        },
        
        # Visualization path
        "visualization_path": evaluation_result.visualization_path if evaluation_result.visualization_path else None
    }
    
    return response


@router.get("/training-data-evaluations")
async def get_training_data_evaluations(
    limit: int = 100,
    offset: int = 0,
    has_ground_truth: bool = None,
    task_type: str = None,
    source_format: str = None,
    db: Session = Depends(get_db)
):
    """
    Get list of training data images suitable for evaluation.
    
    Args:
        limit: Maximum number of results
        offset: Offset for pagination
        has_ground_truth: Filter by presence of ground truth
        task_type: Filter by task type
        source_format: Filter by source format
        db: Database session
    
    Returns:
        List of training data images with ground truth status
    """
    from sqlalchemy import func
    from sqlalchemy.orm import load_only
    
    # Build filter conditions (shared between count and fetch queries)
    filters = []
    
    # Filter by ground truth presence
    if has_ground_truth is not None:
        if has_ground_truth:
            filters.append(TrainingDataImage.ground_truth_correct.isnot(None))
        else:
            filters.append(TrainingDataImage.ground_truth_correct.is_(None))
    
    # Filter by task type
    if task_type:
        filters.append(TrainingDataImage.task_type == task_type)
    
    # Filter by source format
    if source_format:
        filters.append(TrainingDataImage.source_format == source_format)
    
    # Get total count using func.count (fast - doesn't load data)
    count_query = db.query(func.count(TrainingDataImage.id))
    for f in filters:
        count_query = count_query.filter(f)
    total = count_query.scalar()
    
    # Fetch data using load_only to exclude BLOB columns (critical for performance)
    query = db.query(TrainingDataImage).options(
        load_only(
            TrainingDataImage.id,
            TrainingDataImage.patient_id,
            TrainingDataImage.task_type,
            TrainingDataImage.source_format,
            TrainingDataImage.test_name,
            TrainingDataImage.extraction_metadata,
            TrainingDataImage.uploaded_at,
            TrainingDataImage.ground_truth_correct,
            TrainingDataImage.ground_truth_extra
        )
    )
    for f in filters:
        query = query.filter(f)
    
    # Order by most recent first and apply pagination
    images = query.order_by(TrainingDataImage.uploaded_at.desc()).offset(offset).limit(limit).all()
    
    # Prepare results
    results = []
    for img in images:
        metadata = json.loads(img.extraction_metadata) if img.extraction_metadata else {}
        
        results.append({
            "id": img.id,
            "patient_id": img.patient_id,
            "task_type": img.task_type,
            "source_format": img.source_format,
            "test_name": img.test_name,
            "uploaded_at": img.uploaded_at.isoformat(),
            "has_ground_truth": img.ground_truth_correct is not None,
            "ground_truth_correct": img.ground_truth_correct,
            "ground_truth_extra": img.ground_truth_extra,
            "width": metadata.get("width"),
            "height": metadata.get("height")
        })
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "evaluations": results
    }
