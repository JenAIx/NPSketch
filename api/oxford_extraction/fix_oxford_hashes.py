#!/usr/bin/env python3
"""
Fix OXFORD Image Hashes

Problem: OXFORD entries were imported with hashes calculated from processed_image_data,
but duplicate checking uses hashes from original_file_data. This script fixes the
inconsistency by recalculating all hashes from original_file_data.

Usage:
    docker exec npsketch-api python3 /app/oxford_extraction/fix_oxford_hashes.py
"""

import sys
sys.path.insert(0, '/app')

from database import SessionLocal, TrainingDataImage
import hashlib
from datetime import datetime


def fix_oxford_hashes(dry_run=False):
    """
    Recalculate image_hash for all OXFORD entries using original_file_data.
    
    Args:
        dry_run: If True, only report changes without committing
    """
    print("=" * 80)
    print("FIX OXFORD IMAGE HASHES")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will update database)'}")
    print()
    
    db = SessionLocal()
    
    try:
        # Get all OXFORD entries
        print("Loading OXFORD entries...")
        oxford_entries = db.query(TrainingDataImage).filter(
            TrainingDataImage.source_format == 'OXFORD'
        ).all()
        
        total = len(oxford_entries)
        print(f"✓ Found {total} OXFORD entries")
        print()
        
        if total == 0:
            print("No entries to update!")
            return
        
        # Statistics
        updated_count = 0
        unchanged_count = 0
        error_count = 0
        
        print("Processing entries...")
        print("-" * 80)
        
        for i, entry in enumerate(oxford_entries, 1):
            try:
                # Calculate hash from original_file_data
                if not entry.original_file_data:
                    print(f"[{i}/{total}] ID {entry.id}: ⚠️  No original_file_data")
                    error_count += 1
                    continue
                
                new_hash = hashlib.sha256(entry.original_file_data).hexdigest()
                old_hash = entry.image_hash
                
                if new_hash != old_hash:
                    if not dry_run:
                        entry.image_hash = new_hash
                    
                    updated_count += 1
                    
                    # Log every 100th update or if different
                    if i % 100 == 0 or updated_count <= 5:
                        print(f"[{i}/{total}] ID {entry.id} ({entry.patient_id}_{entry.task_type}):")
                        print(f"  Old: {old_hash[:16]}...")
                        print(f"  New: {new_hash[:16]}...")
                        print(f"  {'Would update' if dry_run else 'Updated'} ✓")
                else:
                    unchanged_count += 1
                    if i % 200 == 0:
                        print(f"[{i}/{total}] ID {entry.id}: Unchanged (hash already correct)")
                
            except Exception as e:
                print(f"[{i}/{total}] ID {entry.id}: ❌ Error: {e}")
                error_count += 1
                continue
        
        # Commit if not dry run
        if not dry_run:
            print()
            print("Committing changes to database...")
            db.commit()
            print("✓ Changes committed!")
        else:
            print()
            print("DRY RUN - No changes committed")
        
        # Summary
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total entries:   {total}")
        print(f"Updated:         {updated_count}")
        print(f"Unchanged:       {unchanged_count}")
        print(f"Errors:          {error_count}")
        print()
        
        if dry_run:
            print("✓ Dry run complete. Run without --dry-run to apply changes.")
        else:
            print("✓ All OXFORD hashes fixed!")
            print("  Duplicate checking should now work correctly.")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix OXFORD image hashes')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show changes without committing')
    
    args = parser.parse_args()
    
    fix_oxford_hashes(dry_run=args.dry_run)

