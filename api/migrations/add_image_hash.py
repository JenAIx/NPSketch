#!/usr/bin/env python3
"""
Database migration: Add image_hash column to uploaded_images table.

This migration:
1. Adds the image_hash column (String(64), nullable, indexed)
2. Calculates hashes for existing images
3. Updates the database with computed hashes

Run this script to update an existing database.
"""

import sys
import os
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, Column, String, text
from sqlalchemy.orm import sessionmaker
from database import Base, UploadedImage

def migrate_database(database_url: str = "sqlite:///./data/npsketch.db"):
    """
    Migrate database to add image_hash column.
    
    Args:
        database_url: Database connection URL
    """
    print("=" * 70)
    print("DATABASE MIGRATION: Add image_hash Column")
    print("=" * 70)
    
    # Create engine and session
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        # Check if column already exists
        print("\n1. Checking if image_hash column exists...")
        result = db.execute(text("PRAGMA table_info(uploaded_images)"))
        columns = [row[1] for row in result]
        
        if 'image_hash' in columns:
            print("   ✓ image_hash column already exists, skipping creation")
        else:
            print("   ➜ Adding image_hash column...")
            # Add column (SQLite-specific)
            db.execute(text(
                "ALTER TABLE uploaded_images ADD COLUMN image_hash VARCHAR(64)"
            ))
            db.commit()
            print("   ✓ Column added successfully")
        
        # Create index if it doesn't exist
        print("\n2. Creating index on image_hash...")
        try:
            db.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_uploaded_images_image_hash "
                "ON uploaded_images (image_hash)"
            ))
            db.commit()
            print("   ✓ Index created successfully")
        except Exception as e:
            print(f"   ⚠️  Index creation skipped (may already exist): {e}")
        
        # Calculate hashes for existing images
        print("\n3. Calculating hashes for existing images...")
        images = db.query(UploadedImage).filter(
            (UploadedImage.image_hash == None) | (UploadedImage.image_hash == "")
        ).all()
        
        if not images:
            print("   ✓ All images already have hashes")
        else:
            print(f"   ➜ Found {len(images)} images without hashes")
            
            for i, image in enumerate(images, 1):
                if image.processed_image_data:
                    # Calculate hash of processed image
                    hash_value = hashlib.sha256(image.processed_image_data).hexdigest()
                    image.image_hash = hash_value
                    
                    if i % 10 == 0:
                        print(f"   ➜ Processed {i}/{len(images)} images...")
            
            db.commit()
            print(f"   ✓ Updated {len(images)} images with hashes")
        
        # Show statistics
        print("\n4. Migration Statistics:")
        total = db.query(UploadedImage).count()
        with_hash = db.query(UploadedImage).filter(
            UploadedImage.image_hash != None
        ).count()
        
        print(f"   Total images: {total}")
        print(f"   With hash: {with_hash}")
        print(f"   Without hash: {total - with_hash}")
        
        # Check for potential duplicates
        print("\n5. Checking for potential duplicates...")
        duplicates_query = text("""
            SELECT image_hash, COUNT(*) as count
            FROM uploaded_images
            WHERE image_hash IS NOT NULL
            GROUP BY image_hash
            HAVING COUNT(*) > 1
        """)
        duplicates = db.execute(duplicates_query).fetchall()
        
        if duplicates:
            print(f"   ⚠️  Found {len(duplicates)} duplicate image(s):")
            for dup_hash, count in duplicates:
                print(f"      - Hash {dup_hash[:16]}... appears {count} times")
        else:
            print("   ✓ No duplicates found")
        
        print("\n" + "=" * 70)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return False
    finally:
        db.close()
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate database to add image_hash column")
    parser.add_argument(
        "--database-url",
        default="sqlite:///./data/npsketch.db",
        help="Database connection URL (default: sqlite:///./data/npsketch.db)"
    )
    
    args = parser.parse_args()
    
    success = migrate_database(args.database_url)
    sys.exit(0 if success else 1)

