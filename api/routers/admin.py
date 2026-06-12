"""
Admin Router - Administrative endpoints for NPSketch API

Contains endpoints for:
- Database migrations
- System administration tasks
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, TrainingDataImage, Base, engine
import os
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


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
        logger.info("All tables dropped")
        
        # Recreate all tables
        Base.metadata.create_all(bind=engine)
        logger.info("All tables recreated")
        
        # Clear visualization files
        viz_dir = "/app/data/visualizations"
        if os.path.exists(viz_dir):
            for file in os.listdir(viz_dir):
                file_path = os.path.join(viz_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"Error deleting {file_path}: {e}")
        
        logger.info("Visualization files cleared")
        
        return {
            "success": True,
            "message": "Database reset successfully",
            "tables_recreated": True,
            "visualizations_cleared": True
        }
        
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
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
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
