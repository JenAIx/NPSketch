#!/usr/bin/env python3
"""
Oxford Dataset Database Populator

Populates the training_data_images table with Oxford manual rater data.

Usage:
    # Test run with first 5 files
    python3 oxford_db_populator.py --test
    
    # Full run with all files
    python3 oxford_db_populator.py
    
    # Custom limit
    python3 oxford_db_populator.py --limit 10

Author: NPSketch Team
Date: 2025-12-22
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
BASE_DIR = Path("/app/templates/training_data_oxford_manual_rater_202512")
CSV_PATH = BASE_DIR / "Rater1_simple.csv"
ORIGINAL_IMGS_DIR = BASE_DIR / "imgs"
NORMALIZED_IMGS_DIR = BASE_DIR / "imgs_normalized_568x274"


def populate_oxford_data(
    csv_path: Path,
    original_imgs_dir: Path,
    normalized_imgs_dir: Path,
    db: Session,
    limit: int = None,
    test_mode: bool = False
):
    """
    Populate training_data_images table with Oxford dataset.
    
    Args:
        csv_path: Path to Rater1_simple.csv
        original_imgs_dir: Directory with original PNG files
        normalized_imgs_dir: Directory with normalized 568×274 PNG files
        db: Database session
        limit: Maximum number of files to process (None = all)
        test_mode: If True, only process first 5 files and show detailed logging
    
    Returns:
        dict: Statistics with 'success', 'errors', 'duplicates', 'skipped'
    """
    print("=" * 80)
    print("OXFORD DATASET DATABASE POPULATOR")
    print("=" * 80)
    print(f"Started at:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CSV file:        {csv_path}")
    print(f"Original images: {original_imgs_dir}")
    print(f"Normalized images: {normalized_imgs_dir}")
    if test_mode:
        print(f"Mode:            TEST (first 5 files only)")
    elif limit:
        print(f"Mode:            LIMITED (max {limit} files)")
    else:
        print(f"Mode:            FULL (all files)")
    print("=" * 80)
    print()
    
    # Step 1: Load CSV
    print("[STEP 1] Loading CSV file...")
    if not csv_path.exists():
        print(f"✗ [ERROR] CSV file not found: {csv_path}")
        return {'success': 0, 'errors': 0, 'duplicates': 0, 'skipped': 0}
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"✗ [ERROR] Failed to load CSV: {e}")
        return {'success': 0, 'errors': 0, 'duplicates': 0, 'skipped': 0}
    
    # Apply limit if specified
    if test_mode:
        # For test mode, get first 5 RECALL images to show different TotalScore values
        df_recall = df[df['Cond'] == 'RECALL'].head(5)
        if len(df_recall) < 5:
            print(f"  ⚠ Warning: Only {len(df_recall)} RECALL entries found")
        df = df_recall
        print(f"  → Test mode: Processing first 5 RECALL rows (to show different TotalScore values)")
    elif limit:
        df = df.head(limit)
        print(f"  → Limited mode: Processing first {limit} rows")
    
    print()
    
    # Step 2: Validate directories
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
    
    # Step 3: Process each row
    print("[STEP 3] Processing images...")
    print("-" * 80)
    
    session_id = f"oxford_{datetime.now().strftime('%Y%m%d')}"
    success_count = 0
    error_count = 0
    duplicate_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        patient_id = str(row['ID'])
        task_type = str(row['Cond'])  # Already "COPY" or "RECALL"
        total_score = int(row['TotalScore'])
        
        filename = f"{patient_id}_{task_type}.png"
        original_path = original_imgs_dir / filename
        normalized_path = normalized_imgs_dir / filename
        
        print(f"[{idx+1:4d}/{len(df)}] Processing {filename}...")
        print(f"    Patient ID: {patient_id}")
        print(f"    Task Type:  {task_type}")
        print(f"    TotalScore: {total_score}")
        
        # Check if files exist
        if not original_path.exists():
            print(f"    ✗ Original file not found: {original_path}")
            error_count += 1
            skipped_count += 1
            print()
            continue
        
        if not normalized_path.exists():
            print(f"    ✗ Normalized file not found: {normalized_path}")
            error_count += 1
            skipped_count += 1
            print()
            continue
        
        try:
            # Read files
            print(f"    Reading original file...")
            with open(original_path, 'rb') as f:
                original_data = f.read()
            original_size = len(original_data)
            print(f"      → Original size: {original_size:,} bytes")
            
            print(f"    Reading normalized file...")
            with open(normalized_path, 'rb') as f:
                processed_data = f.read()
            processed_size = len(processed_data)
            print(f"      → Normalized size: {processed_size:,} bytes")
            
            # Calculate hash (use processed image for duplicate detection)
            print(f"    Calculating image hash...")
            image_hash = hashlib.sha256(processed_data).hexdigest()
            print(f"      → Hash: {image_hash[:16]}...")
            
            # Check for duplicates
            print(f"    Checking for duplicates...")
            existing = db.query(TrainingDataImage).filter(
                TrainingDataImage.image_hash == image_hash
            ).first()
            
            if existing:
                print(f"    ⚠ Duplicate image found (ID: {existing.id})")
                print(f"      → Skipping (already in database)")
                duplicate_count += 1
                print()
                continue
            
            print(f"      → No duplicates found")
            
            # Create extraction metadata
            extraction_metadata = {
                "width": 568,
                "height": 274,
                "line_thickness": 2.0,
                "auto_crop": True,
                "padding_px": 5,
                "original_resolution": "variable",
                "normalization_method": "Zhang-Suen + dilation",
                "source": "Oxford manual rater dataset",
                "normalized_filename": filename,
                "original_file_size": original_size,
                "processed_file_size": processed_size
            }
            
            # Create features data
            features_data = {
                "Total_Score": total_score
            }
            
            print(f"    Creating database entry...")
            print(f"      → extraction_metadata: {json.dumps(extraction_metadata, indent=2)}")
            print(f"      → features_data: {json.dumps(features_data)}")
            
            # Create database entry
            training_image = TrainingDataImage(
                patient_id=patient_id,
                task_type=task_type,
                source_format="OXFORD",
                original_filename=filename,
                original_file_data=original_data,
                processed_image_data=processed_data,
                image_hash=image_hash,
                extraction_metadata=json.dumps(extraction_metadata),
                features_data=json.dumps(features_data),
                session_id=session_id
            )
            
            print(f"    Saving to database...")
            db.add(training_image)
            db.commit()
            db.refresh(training_image)
            
            print(f"    ✓ Success! Database ID: {training_image.id}")
            success_count += 1
            print()
            
        except Exception as e:
            db.rollback()
            print(f"    ✗ Error: {e}")
            if test_mode:
                import traceback
                traceback.print_exc()
            error_count += 1
            print()
    
    # Print summary
    print("=" * 80)
    print("POPULATION COMPLETE")
    print("=" * 80)
    print(f"Finished at:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processed: {len(df)}")
    print(f"Success:         {success_count}")
    print(f"Errors:          {error_count}")
    print(f"Duplicates:      {duplicate_count}")
    print(f"Skipped:         {skipped_count}")
    print(f"Session ID:      {session_id}")
    print("=" * 80)
    
    if test_mode:
        print()
        print("=" * 80)
        print("TEST MODE COMPLETE - Review database entries before proceeding")
        print("=" * 80)
        print("To view entries in database:")
        print("  SELECT * FROM training_data_images WHERE session_id = '{}';".format(session_id))
        print()
        print("To proceed with all files, run:")
        print("  python3 oxford_db_populator.py")
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
    
    parser = argparse.ArgumentParser(
        description='Populate database with Oxford dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test run with first 5 files
  python3 oxford_db_populator.py --test
  
  # Process first 10 files
  python3 oxford_db_populator.py --limit 10
  
  # Process all files
  python3 oxford_db_populator.py
        """
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only first 5 files with detailed logging'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process'
    )
    
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
        stats = populate_oxford_data(
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

