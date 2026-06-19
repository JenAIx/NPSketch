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
    """Wipe ALL scratch under /app/data/tmp — every file (any type) AND every subdirectory.
    tmp is pure scratch; the real data (npsketch.db, models/, visualizations/, logs/) lives
    elsewhere under /app/data, so this is safe. The tmp root itself is kept/recreated."""
    import shutil

    tmp_root = "/app/data/tmp"
    cleaned_files = 0
    cleaned_dirs = 0
    freed = 0

    def dir_size(p):
        s = 0
        for root, _, files in os.walk(p):
            for f in files:
                try:
                    s += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return s

    try:
        os.makedirs(tmp_root, exist_ok=True)
        for entry in os.listdir(tmp_root):
            path = os.path.join(tmp_root, entry)
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    freed += dir_size(path)
                    shutil.rmtree(path)
                    cleaned_dirs += 1
                else:
                    freed += os.path.getsize(path)
                    os.unlink(path)
                    cleaned_files += 1
            except OSError as e:
                logger.warning(f"cleanup-tmp: could not remove {path}: {e}")

        freed_mb = round(freed / (1024 * 1024), 1)
        return {
            "success": True,
            "cleaned_files": cleaned_files,
            "cleaned_directories": cleaned_dirs,
            "freed_mb": freed_mb,
            "message": f"Removed {cleaned_files} files and {cleaned_dirs} subfolders ({freed_mb} MB) from tmp",
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/label-breakdown")
async def label_breakdown(db: Session = Depends(get_db)):
    """Breakdown of label completeness so the Review/Label work can target what's missing:
    how many rows have a Total_Score vs the full 60 components, how many are unscored, and how
    many are human-validated (write-protected)."""
    import json
    from sqlalchemy.orm import load_only

    rows = db.query(TrainingDataImage).options(load_only(
        TrainingDataImage.features_data, TrainingDataImage.validated)).all()
    out = {"total": len(rows), "labeled": 0, "with_total_score": 0,
           "with_components": 0, "score_only": 0, "unscored": 0, "validated": 0}
    for r in rows:
        if r.validated:
            out["validated"] += 1
        if not r.features_data or r.features_data in ('{}', 'null', ''):
            out["unscored"] += 1
            continue
        try:
            fd = json.loads(r.features_data)
        except (ValueError, TypeError):
            out["unscored"] += 1
            continue
        out["labeled"] += 1
        has_score = fd.get("Total_Score") is not None
        comp = fd.get("components")
        has_comp = bool(comp and comp.get("presence") and comp.get("accuracy") and comp.get("position"))
        if has_score:
            out["with_total_score"] += 1
        if has_comp:
            out["with_components"] += 1
        elif has_score:
            out["score_only"] += 1
    return out


@router.get("/cloudflare-status")
async def cloudflare_status():
    """Status + public URL of the Cloudflare tunnel, read from cloudflared's metrics server
    (http://cloudflared:2000 on the internal network). Quick-tunnel URLs are ephemeral."""
    import urllib.request, urllib.error, json as _json
    base = "http://cloudflared:2000"
    out = {"running": False, "url": None, "ready": False, "connections": 0}
    try:
        h = _json.loads(urllib.request.urlopen(base + "/quicktunnel", timeout=4).read().decode())
        host = h.get("hostname")
        if host:
            out["url"] = "https://" + host
        out["running"] = True
    except Exception:
        return out  # cloudflared unreachable -> tunnel not running
    try:
        r = urllib.request.urlopen(base + "/ready", timeout=4)
        out["ready"] = (r.status == 200)
        try:
            out["connections"] = _json.loads(r.read().decode()).get("readyConnections", 0)
        except Exception:
            pass
    except urllib.error.HTTPError as e:           # 503 while still connecting
        try:
            out["connections"] = _json.loads(e.read().decode()).get("readyConnections", 0)
        except Exception:
            pass
    except Exception:
        pass
    return out
