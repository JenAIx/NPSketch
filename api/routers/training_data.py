"""
Training Data Extraction Router

Handles file uploads and processing for AI training data:
- MATLAB .mat files → MAT Extractor
- OCS PNG/JPG images → OCS Extractor

Stores original + processed data in database with duplicate detection.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.orm import Session
from typing import List
import os
import subprocess
import shutil
import tempfile
from datetime import datetime
import json
import hashlib
import re
import cv2
import numpy as np
import io
import scipy.io
from PIL import Image
import csv

from database import get_db, TrainingDataImage

router = APIRouter(prefix="/api", tags=["training_data"])

# Temporary upload directory
UPLOAD_DIR = "/app/data/tmp/uploads"
OUTPUT_DIR = "/app/data/tmp/extracted"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


@router.post("/extract-training-data")
async def extract_training_data(
    format: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
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
    
    # Create unique session
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    session_output_dir = os.path.join(OUTPUT_DIR, session_id)
    
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
                        print(f"  Skipping duplicate: {extracted_filename} (ID {existing.id})")
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
            print(f"Warning: Could not clean tmp root: {e}")


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
        print(f"Error running MAT extractor: {e}")
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
        print(f"Error running OCS extractor: {e}")
        return False


@router.get("/training-data-images")
async def get_training_data_images(
    limit: int = 100,
    offset: int = 0,
    patient_id: str = None,
    task_type: str = None,
    source_format: str = None,
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
        db: Database session
    
    Returns:
        List of training data images with metadata
    """
    query = db.query(TrainingDataImage)
    
    if patient_id:
        query = query.filter(TrainingDataImage.patient_id == patient_id)
    if task_type:
        query = query.filter(TrainingDataImage.task_type == task_type)
    if source_format:
        query = query.filter(TrainingDataImage.source_format == source_format)
    
    query = query.order_by(TrainingDataImage.uploaded_at.desc())
    
    total = query.count()
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
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "uploaded_at": img.uploaded_at.isoformat(),
            "session_id": img.session_id,
            "has_features": has_features
        })
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "images": results
    }


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
            print(f"Error extracting MAT original drawing: {e}")
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


@router.get("/training-data-image/{image_id}/features")
async def get_training_data_features(image_id: int, db: Session = Depends(get_db)):
    """Get features/labels for a training data image."""
    img = db.query(TrainingDataImage).filter(TrainingDataImage.id == image_id).first()
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


@router.get("/training-data-features-template")
async def download_features_template(db: Session = Depends(get_db)):
    """
    Generate CSV template with all training data entries for bulk feature upload.
    
    Returns:
        CSV file with columns: Patient, Task, Total_Score, Data_Quality
    """
    images = db.query(TrainingDataImage).order_by(
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
    correct_lines: int = Form(0),
    missing_lines: int = Form(0),
    extra_lines: int = Form(0),
    db: Session = Depends(get_db)
):
    """
    Save manually drawn image as training data.
    
    Source format will be 'DRAWN', can have both line counts and features.
    """
    import hashlib
    
    try:
        # Read image
        content = await file.read()
        image_hash = hashlib.sha256(content).hexdigest()
        
        # Check for duplicate name
        existing = db.query(TrainingDataImage).filter(
            TrainingDataImage.test_name == name
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail=f"Name '{name}' already exists")
        
        # Create entry
        training_image = TrainingDataImage(
            patient_id=name,  # Use name as patient_id for drawn images
            task_type='DRAWN',
            source_format='DRAWN',
            original_filename=f"{name}.png",
            original_file_data=content,
            processed_image_data=content,  # Same as original for drawn images
            image_hash=image_hash,
            expected_correct=correct_lines if correct_lines > 0 else None,
            expected_missing=missing_lines if missing_lines > 0 else None,
            expected_extra=extra_lines if extra_lines > 0 else None,
            test_name=name,
            session_id='manual_draw',
            extraction_metadata=json.dumps({
                "width": 568,
                "height": 274,
                "manually_drawn": True
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
            print(f"Error cleaning up {directory}: {e}")
