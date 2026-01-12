#!/usr/bin/env python3
"""
Algorithm Dataset Database Populator

Populates the training_data_images table with algorithm training data.

Usage:
    python3 algorithm_db_populator.py [--test] [--limit N]

Author: NPSketch Team
Date: 2026-01-12
"""

import os
import sys
import hashlib
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy.orm import Session
from database import TrainingDataImage, get_db, init_database


# Configuration
BASE_DIR = Path("/app/templates/algorithm_training_data_20260112")
CSV_PATH = BASE_DIR / "FIGURECOPY_ALGORITHM_SCORED_2025-01-08.csv"
ORIGINAL_IMGS_DIR = BASE_DIR / "imgs"
NORMALIZED_IMGS_DIR = BASE_DIR / "imgs_normalized_568x274"


def map_cond_to_task_type(cond_value):
    """
    Map CSV Cond values to database task_type values.
    
    CSV uses: "CPY" and "MEM"
    Database uses: "COPY" and "RECALL"
    """
    mapping = {
        "CPY": "COPY",
        "MEM": "RECALL"
    }
    
    cond_upper = str(cond_value).upper().strip()
    if cond_upper not in mapping:
        raise ValueError(f"Unknown Cond value: {cond_value}. Expected 'CPY' or 'MEM'")
    
    return mapping[cond_upper]


def populate_algorithm_data(
    csv_path: Path,
    original_imgs_dir: Path,
    normalized_imgs_dir: Path,
    db: Session,
    limit: int = None,
    test_mode: bool = False
):
    """Populate training_data_images table with algorithm dataset."""
    print("=" * 80)
    print("ALGORITHM DATASET DATABASE POPULATOR")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CSV file: {csv_path}")
    print(f"Original images: {original_imgs_dir}")
    print(f"Normalized images: {normalized_imgs_dir}")
    if test_mode:
        print(f"Mode: TEST (first 5 files only)")
    elif limit:
        print(f"Mode: LIMITED (max {limit} files)")
    else:
        print(f"Mode: FULL (all files)")
    print("=" * 80)
    print()
    
    # Load CSV (semicolon-separated)
    print("[STEP 1] Loading CSV file...")
    if not csv_path.exists():
        print(f"✗ [ERROR] CSV file not found: {csv_path}")
        return {'success': 0, 'errors': 0, 'duplicates': 0, 'skipped': 0}
    
    try:
        df = pd.read_csv(csv_path, sep=';')
        print(f"✓ Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"✗ [ERROR] Failed to load CSV: {e}")
        return {'success': 0, 'errors': 0, 'duplicates': 0, 'skipped': 0}
    
    if test_mode:
        df = df.head(5)
        print(f"  → Test mode: Processing first 5 rows")
    elif limit:
        df = df.head(limit)
        print(f"  → Limited mode: Processing first {limit} rows")
    
    print()
    
    # Validate directories
    print("[STEP 2] Validating directories...")
    if not original_imgs_dir.exists():
        print(f"✗ [ERROR] Original images directory not found: {original_imgs_dir}")
        return {'success': 0, 'errors': 0, 'duplicates': 0, 'skipped': 0}
    
    if not normalized_imgs_dir.exists():
        print(f"✗ [ERROR] Normalized images directory not found: {normalized_imgs_dir}")
        return {'success': 0, 'errors': 0, 'duplicates': 0, 'skipped': 0}
    
    print(f"✓ Original images directory exists")
    print(f"✓ Normalized images directory exists")
    print()
    
    # Process each row
    print("[STEP 3] Processing images...")
    print("-" * 80)
    
    session_id = f"algorithm_{datetime.now().strftime('%Y%m%d')}"
    success_count = 0
    error_count = 0
    duplicate_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        patient_id = str(row['ID'])
        csv_cond = str(row['Cond'])
        total_score = int(row['TotalScore'])
        
        # Map CSV Cond to database task_type
        try:
            task_type = map_cond_to_task_type(csv_cond)
        except ValueError as e:
            print(f"[{idx+1:4d}/{len(df)}] Error: {e}")
            error_count += 1
            skipped_count += 1
            continue
        
        # Images are named with COPY/RECALL, not CPY/MEM
        filename = f"{patient_id}_{task_type}.png"
        original_path = original_imgs_dir / filename
        normalized_path = normalized_imgs_dir / filename
        
        # Check if files exist
        if not original_path.exists():
            print(f"[{idx+1:4d}/{len(df)}] ✗ Original file not found: {filename}")
            error_count += 1
            skipped_count += 1
            continue
        
        if not normalized_path.exists():
            print(f"[{idx+1:4d}/{len(df)}] ✗ Normalized file not found: {filename}")
            error_count += 1
            skipped_count += 1
            continue
        
        try:
            # Read files
            with open(original_path, 'rb') as f:
                original_data = f.read()
            
            with open(normalized_path, 'rb') as f:
                processed_data = f.read()
            
            # Calculate hash from ORIGINAL image
            image_hash = hashlib.sha256(original_data).hexdigest()
            
            # Check for duplicates
            existing = db.query(TrainingDataImage).filter(
                TrainingDataImage.image_hash == image_hash
            ).first()
            
            if existing:
                print(f"[{idx+1:4d}/{len(df)}] ⚠ Duplicate: {filename} (ID: {existing.id})")
                duplicate_count += 1
                continue
            
            # Create metadata
            extraction_metadata = {
                "width": 568,
                "height": 274,
                "line_thickness": 2.0,
                "auto_crop": True,
                "padding_px": 5,
                "original_resolution": "variable",
                "normalization_method": "Zhang-Suen + dilation",
                "source": "Algorithm training dataset"
            }
            
            # Create features data
            features_data = {
                "Total_Score": total_score
            }
            
            # Create database entry
            training_image = TrainingDataImage(
                patient_id=patient_id,
                task_type=task_type,
                source_format="ALGORITHM",
                original_filename=filename,
                original_file_data=original_data,
                processed_image_data=processed_data,
                image_hash=image_hash,
                extraction_metadata=json.dumps(extraction_metadata),
                features_data=json.dumps(features_data),
                session_id=session_id
            )
            
            db.add(training_image)
            db.commit()
            db.refresh(training_image)
            
            success_count += 1
            
            # Progress indicator
            if success_count % 50 == 0:
                print(f"  Progress: {success_count} imported...")
            
        except Exception as e:
            db.rollback()
            print(f"[{idx+1:4d}/{len(df)}] ✗ Error: {e}")
            error_count += 1
    
    # Print summary
    print()
    print("=" * 80)
    print("POPULATION COMPLETE")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processed: {len(df)}")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Duplicates: {duplicate_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Session ID: {session_id}")
    print("=" * 80)
    
    return {
        'success': success_count,
        'errors': error_count,
        'duplicates': duplicate_count,
        'skipped': skipped_count,
        'session_id': session_id
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Populate database with algorithm dataset')
    parser.add_argument('--test', action='store_true', help='Test mode: first 5 files')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files')
    
    args = parser.parse_args()
    
    # Initialize database
    print("[INIT] Initializing database...")
    init_database()
    print("✓ Database initialized")
    print()
    
    # Get database session
    db = next(get_db())
    
    try:
        # Run population
        stats = populate_algorithm_data(
            csv_path=CSV_PATH,
            original_imgs_dir=ORIGINAL_IMGS_DIR,
            normalized_imgs_dir=NORMALIZED_IMGS_DIR,
            db=db,
            limit=args.limit,
            test_mode=args.test
        )
        
        # Exit with appropriate code
        if stats['errors'] > 0:
            sys.exit(1)
        elif stats['success'] == 0:
            sys.exit(2)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        db.rollback()
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ [FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == '__main__':
    main()
