"""
Admin Router - Administrative endpoints for NPSketch API

Contains endpoints for:
- Database migrations
- System administration tasks
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, UploadedImage, TrainingDataImage, Base, engine
import os

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


@router.post("/reset-database")
async def reset_database(confirm: str = ""):
    """
    Reset the entire database. WARNING: This deletes ALL data!
    
    Args:
        confirm: Must be "RESET_ALL_DATA" to proceed
    
    Returns:
        Reset status
    """
    if confirm != "RESET_ALL_DATA":
        raise HTTPException(
            status_code=400, 
            detail="Confirmation required. Pass confirm='RESET_ALL_DATA' to proceed."
        )
    
    try:
        # Close all sessions
        db = next(get_db())
        db.close()
        
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        print("✓ All tables dropped")
        
        # Recreate all tables
        Base.metadata.create_all(bind=engine)
        print("✓ All tables recreated")
        
        # Clear visualization files
        viz_dir = "/app/data/visualizations"
        if os.path.exists(viz_dir):
            for file in os.listdir(viz_dir):
                file_path = os.path.join(viz_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        print("✓ Visualization files cleared")
        
        return {
            "success": True,
            "message": "Database reset successfully",
            "tables_recreated": True,
            "visualizations_cleared": True
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Reset failed: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@router.post("/cleanup-tmp")
async def cleanup_tmp_directory():
    """
    Clean up temporary extraction files in /app/data/tmp.
    
    Returns:
        Cleanup statistics
    """
    import shutil
    
    tmp_root = "/app/data/tmp"
    cleaned_files = 0
    cleaned_dirs = 0
    
    try:
        # Clean PNG files in root
        for file in os.listdir(tmp_root):
            file_path = os.path.join(tmp_root, file)
            
            if os.path.isfile(file_path) and file.endswith('.png'):
                os.unlink(file_path)
                cleaned_files += 1
        
        # Clean subdirectories (uploads, extracted)
        for subdir in ['uploads', 'extracted']:
            subdir_path = os.path.join(tmp_root, subdir)
            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                # Remove all session subdirectories
                for session_dir in os.listdir(subdir_path):
                    session_path = os.path.join(subdir_path, session_dir)
                    if os.path.isdir(session_path):
                        shutil.rmtree(session_path)
                        cleaned_dirs += 1
        
        return {
            "success": True,
            "cleaned_files": cleaned_files,
            "cleaned_directories": cleaned_dirs,
            "message": f"Cleaned {cleaned_files} files and {cleaned_dirs} directories from tmp"
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Cleanup failed: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
