#!/usr/bin/env python3
"""
Comprehensive Validation of Oxford Dataset Import

Validates that database entries correctly match:
1. Original images in imgs/ directory
2. Normalized images in imgs_normalized_568x274/ directory
3. TotalScore values from CSV file
4. All required fields are present and correct
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db, TrainingDataImage

# Paths
BASE_DIR = Path("/app/templates/training_data_oxford_manual_rater_202512")
CSV_PATH = BASE_DIR / "Rater1_simple.csv"
ORIGINAL_IMGS_DIR = BASE_DIR / "imgs"
NORMALIZED_IMGS_DIR = BASE_DIR / "imgs_normalized_568x274"


def validate_oxford_import(session_id='oxford_20251222'):
    """
    Comprehensive validation of Oxford dataset import.
    
    Returns:
        dict: Validation results with statistics
    """
    print("=" * 80)
    print("OXFORD DATASET IMPORT VALIDATION")
    print("=" * 80)
    print()
    
    db = next(get_db())
    
    # Step 1: Get all OXFORD entries from database
    print("[STEP 1] Loading database entries...")
    db_entries = db.query(TrainingDataImage).filter(
        TrainingDataImage.source_format == 'OXFORD'
    ).all()
    print(f"✓ Found {len(db_entries)} entries in database")
    print()
    
    # Step 2: Load CSV data
    print("[STEP 2] Loading CSV data...")
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df)} rows from CSV")
    
    # Create mapping: {ID}_{Cond} -> TotalScore
    csv_map = {}
    for _, row in df.iterrows():
        key = f"{row['ID']}_{row['Cond']}"
        csv_map[key] = int(row['TotalScore'])
    print(f"✓ Created mapping for {len(csv_map)} entries")
    print()
    
    # Step 3: Validate each database entry
    print("[STEP 3] Validating database entries...")
    print("-" * 80)
    
    validation_results = {
        'total': len(db_entries),
        'valid': 0,
        'errors': [],
        'warnings': []
    }
    
    for entry in db_entries:
        errors = []
        warnings = []
        
        # Check basic fields
        if entry.source_format != 'OXFORD':
            errors.append(f"source_format should be 'OXFORD', got '{entry.source_format}'")
        
        if entry.task_type not in ['COPY', 'RECALL']:
            errors.append(f"task_type should be 'COPY' or 'RECALL', got '{entry.task_type}'")
        
        # Check CSV match
        csv_key = f"{entry.patient_id}_{entry.task_type}"
        if csv_key not in csv_map:
            errors.append(f"Entry not found in CSV: {csv_key}")
        else:
            expected_score = csv_map[csv_key]
            try:
                features = json.loads(entry.features_data) if entry.features_data else {}
                actual_score = features.get('Total_Score')
                if actual_score != expected_score:
                    errors.append(f"TotalScore mismatch: expected {expected_score}, got {actual_score}")
            except:
                errors.append(f"Could not parse features_data")
        
        # Check original file exists
        original_path = ORIGINAL_IMGS_DIR / entry.original_filename
        if not original_path.exists():
            errors.append(f"Original file not found: {original_path}")
        else:
            # Verify original file matches database
            with open(original_path, 'rb') as f:
                file_data = f.read()
            if len(file_data) != len(entry.original_file_data):
                warnings.append(f"Original file size mismatch: file={len(file_data)}, db={len(entry.original_file_data)}")
        
        # Check normalized file exists
        normalized_path = NORMALIZED_IMGS_DIR / entry.original_filename
        if not normalized_path.exists():
            errors.append(f"Normalized file not found: {normalized_path}")
        else:
            # Verify normalized file matches database
            with open(normalized_path, 'rb') as f:
                file_data = f.read()
            if len(file_data) != len(entry.processed_image_data):
                warnings.append(f"Normalized file size mismatch: file={len(file_data)}, db={len(entry.processed_image_data)}")
            
            # Verify hash matches
            file_hash = hashlib.sha256(file_data).hexdigest()
            if file_hash != entry.image_hash:
                errors.append(f"Image hash mismatch: file={file_hash[:16]}..., db={entry.image_hash[:16]}...")
            
            # Verify image dimensions
            try:
                img = Image.open(normalized_path)
                if img.size != (568, 274):
                    errors.append(f"Image resolution mismatch: expected 568×274, got {img.size[0]}×{img.size[1]}")
            except Exception as e:
                errors.append(f"Could not read normalized image: {e}")
        
        # Check metadata
        if entry.extraction_metadata:
            try:
                metadata = json.loads(entry.extraction_metadata)
                if metadata.get('width') != 568 or metadata.get('height') != 274:
                    errors.append(f"Metadata resolution mismatch: expected 568×274")
                if metadata.get('line_thickness') != 2.0:
                    errors.append(f"Metadata line_thickness mismatch: expected 2.0")
            except:
                warnings.append("Could not parse extraction_metadata")
        
        # Record results
        if errors:
            validation_results['errors'].append({
                'id': entry.id,
                'patient_id': entry.patient_id,
                'task_type': entry.task_type,
                'errors': errors
            })
        else:
            validation_results['valid'] += 1
            if warnings:
                validation_results['warnings'].append({
                    'id': entry.id,
                    'patient_id': entry.patient_id,
                    'task_type': entry.task_type,
                    'warnings': warnings
                })
    
    # Step 4: Check for missing entries (CSV entries without database entries)
    print()
    print("[STEP 4] Checking for missing entries...")
    db_keys = {f"{e.patient_id}_{e.task_type}" for e in db_entries}
    csv_keys = set(csv_map.keys())
    missing_in_db = csv_keys - db_keys
    
    # Filter out entries where files don't exist
    missing_with_files = []
    for key in missing_in_db:
        parts = key.split('_')
        if len(parts) >= 2:
            patient_id = parts[0]
            task_type = '_'.join(parts[1:])
            filename = f"{patient_id}_{task_type}.png"
            if (ORIGINAL_IMGS_DIR / filename).exists() and (NORMALIZED_IMGS_DIR / filename).exists():
                missing_with_files.append(key)
    
    print(f"  CSV entries: {len(csv_keys)}")
    print(f"  Database entries: {len(db_keys)}")
    print(f"  Missing in DB (total): {len(missing_in_db)}")
    print(f"  Missing in DB (with files): {len(missing_with_files)}")
    if missing_with_files:
        print(f"  Sample missing: {missing_with_files[:5]}")
    print()
    
    # Print summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total database entries: {validation_results['total']}")
    print(f"Valid entries:          {validation_results['valid']} ({100*validation_results['valid']/validation_results['total']:.1f}%)")
    print(f"Entries with errors:    {len(validation_results['errors'])} ({100*len(validation_results['errors'])/validation_results['total']:.1f}%)")
    print(f"Entries with warnings:  {len(validation_results['warnings'])}")
    print()
    
    if validation_results['errors']:
        print("ERRORS FOUND:")
        for err in validation_results['errors'][:10]:  # Show first 10
            print(f"  ID {err['id']}: {err['patient_id']} {err['task_type']}")
            for error in err['errors']:
                print(f"    - {error}")
        if len(validation_results['errors']) > 10:
            print(f"  ... and {len(validation_results['errors']) - 10} more errors")
        print()
    
    if validation_results['warnings']:
        print("WARNINGS:")
        for warn in validation_results['warnings'][:5]:  # Show first 5
            print(f"  ID {warn['id']}: {warn['patient_id']} {warn['task_type']}")
            for warning in warn['warnings']:
                print(f"    - {warning}")
        if len(validation_results['warnings']) > 5:
            print(f"  ... and {len(validation_results['warnings']) - 5} more warnings")
        print()
    
    # Statistics
    print("STATISTICS:")
    task_types = {}
    score_range = {'min': float('inf'), 'max': float('-inf')}
    for entry in db_entries:
        task_types[entry.task_type] = task_types.get(entry.task_type, 0) + 1
        if entry.features_data:
            try:
                features = json.loads(entry.features_data)
                score = features.get('Total_Score')
                if score is not None:
                    score_range['min'] = min(score_range['min'], score)
                    score_range['max'] = max(score_range['max'], score)
            except:
                pass
    
    print(f"  Task types:")
    for task, count in sorted(task_types.items()):
        print(f"    {task}: {count}")
    print(f"  TotalScore range: {score_range['min']} - {score_range['max']}")
    print()
    
    print("=" * 80)
    
    db.close()
    return validation_results


if __name__ == '__main__':
    import sys
    session_id = sys.argv[1] if len(sys.argv) > 1 else 'oxford_20251222'
    validate_oxford_import(session_id)

