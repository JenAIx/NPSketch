"""
Admin Router - Administrative endpoints for NPSketch API

Contains endpoints for:
- Database migrations
- System administration tasks
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, UploadedImage

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/migrate-add-image-hash")
async def migrate_add_image_hash(db: Session = Depends(get_db)):
    """
    Run database migration to add image_hash column.
    This is a one-time migration for duplicate detection.
    
    Returns:
        Migration status and statistics
    """
    import hashlib
    from sqlalchemy import text
    
    try:
        column_added = False
        index_created = False
        
        # Try to add column (will fail if it already exists)
        try:
            db.execute(text(
                "ALTER TABLE uploaded_images ADD COLUMN image_hash VARCHAR(64)"
            ))
            db.commit()
            column_added = True
        except Exception as e:
            # Column probably already exists
            db.rollback()
            if "duplicate column name" not in str(e).lower():
                # Re-raise if it's not a duplicate column error
                raise
        
        # Create index (IF NOT EXISTS handles existing index)
        try:
            db.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_uploaded_images_image_hash "
                "ON uploaded_images (image_hash)"
            ))
            db.commit()
            index_created = True
        except Exception:
            db.rollback()
        
        # Calculate hashes for existing images without hashes
        images = db.query(UploadedImage).filter(
            (UploadedImage.image_hash == None) | (UploadedImage.image_hash == "")
        ).all()
        
        updated_count = 0
        for image in images:
            if image.processed_image_data:
                hash_value = hashlib.sha256(image.processed_image_data).hexdigest()
                image.image_hash = hash_value
                updated_count += 1
        
        db.commit()
        
        # Statistics
        total = db.query(UploadedImage).count()
        with_hash = db.query(UploadedImage).filter(
            UploadedImage.image_hash != None
        ).count()
        
        # Check for duplicates
        duplicates_query = text("""
            SELECT image_hash, COUNT(*) as count
            FROM uploaded_images
            WHERE image_hash IS NOT NULL
            GROUP BY image_hash
            HAVING COUNT(*) > 1
        """)
        duplicates = db.execute(duplicates_query).fetchall()
        
        return {
            "success": True,
            "column_added": column_added,
            "index_created": index_created,
            "images_updated": updated_count,
            "statistics": {
                "total_images": total,
                "with_hash": with_hash,
                "without_hash": total - with_hash,
                "duplicate_groups": len(duplicates)
            },
            "duplicates": [
                {"hash": dup[0][:16] + "...", "count": dup[1]} 
                for dup in duplicates
            ]
        }
        
    except Exception as e:
        db.rollback()
        import traceback
        error_detail = f"Migration failed: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log to console
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")
