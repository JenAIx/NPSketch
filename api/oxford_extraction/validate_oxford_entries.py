#!/usr/bin/env python3
"""
Validate Oxford Database Entries

Quick script to view and validate Oxford dataset entries in the database.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db, TrainingDataImage
import json

def validate_entries(session_id='oxford_20251222'):
    """Validate Oxford entries in database."""
    db = next(get_db())
    
    entries = db.query(TrainingDataImage).filter(
        TrainingDataImage.session_id == session_id
    ).order_by(TrainingDataImage.id).all()
    
    print("=" * 80)
    print("OXFORD DATABASE ENTRIES VALIDATION")
    print("=" * 80)
    print(f"Session ID: {session_id}")
    print(f"Total entries: {len(entries)}\n")
    
    for entry in entries:
        print(f"Entry ID: {entry.id}")
        print(f"  patient_id:        {entry.patient_id}")
        print(f"  task_type:         {entry.task_type}")
        print(f"  source_format:     {entry.source_format}")
        print(f"  original_filename: {entry.original_filename}")
        print(f"  image_hash:        {entry.image_hash[:32]}...")
        print(f"  session_id:        {entry.session_id}")
        print(f"  uploaded_at:       {entry.uploaded_at}")
        
        # Parse metadata
        if entry.extraction_metadata:
            try:
                metadata = json.loads(entry.extraction_metadata)
                print(f"  extraction_metadata:")
                print(f"    width:              {metadata.get('width')}px")
                print(f"    height:             {metadata.get('height')}px")
                print(f"    line_thickness:     {metadata.get('line_thickness')}px")
                print(f"    auto_crop:          {metadata.get('auto_crop')}")
                print(f"    padding_px:         {metadata.get('padding_px')}")
                print(f"    source:             {metadata.get('source')}")
                print(f"    original_file_size: {metadata.get('original_file_size'):,} bytes")
                print(f"    processed_file_size: {metadata.get('processed_file_size'):,} bytes")
            except:
                print(f"    (metadata parse error)")
        
        # Parse features
        if entry.features_data:
            try:
                features = json.loads(entry.features_data)
                print(f"  features_data:")
                for key, value in features.items():
                    print(f"    {key}: {value}")
            except:
                print(f"    (features parse error)")
        
        # File sizes
        print(f"  original_file_data:   {len(entry.original_file_data):,} bytes")
        print(f"  processed_image_data: {len(entry.processed_image_data):,} bytes")
        print()
    
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    # Check consistency
    issues = []
    for entry in entries:
        # Check source_format
        if entry.source_format != "OXFORD":
            issues.append(f"ID {entry.id}: source_format should be 'OXFORD', got '{entry.source_format}'")
        
        # Check task_type
        if entry.task_type not in ["COPY", "RECALL"]:
            issues.append(f"ID {entry.id}: task_type should be 'COPY' or 'RECALL', got '{entry.task_type}'")
        
        # Check metadata
        if entry.extraction_metadata:
            try:
                metadata = json.loads(entry.extraction_metadata)
                if metadata.get('width') != 568 or metadata.get('height') != 274:
                    issues.append(f"ID {entry.id}: Resolution should be 568×274")
                if metadata.get('line_thickness') != 2.0:
                    issues.append(f"ID {entry.id}: Line thickness should be 2.0px")
            except:
                issues.append(f"ID {entry.id}: Invalid extraction_metadata JSON")
        
        # Check features
        if entry.features_data:
            try:
                features = json.loads(entry.features_data)
                if 'Total_Score' not in features:
                    issues.append(f"ID {entry.id}: Missing 'Total_Score' in features_data")
            except:
                issues.append(f"ID {entry.id}: Invalid features_data JSON")
    
    if issues:
        print("⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All entries validated successfully!")
        print("  - source_format: OXFORD ✓")
        print("  - task_type: COPY/RECALL ✓")
        print("  - Resolution: 568×274 ✓")
        print("  - Line thickness: 2.0px ✓")
        print("  - features_data: Total_Score present ✓")
    
    print("=" * 80)
    
    db.close()

if __name__ == '__main__':
    import sys
    session_id = sys.argv[1] if len(sys.argv) > 1 else 'oxford_20251222'
    validate_entries(session_id)

