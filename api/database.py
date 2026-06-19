"""
Database models and setup for NPSketch (AI-only).

Single table: training_data_images — drawings + their CNN training labels
(Total_Score and the optional 60 component sub-labels).
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database configuration
# Data directory is mounted from host at /app/data via docker-compose volume
DATABASE_DIR = '/app/data'
# Note: Directory is created via volume mount in docker-compose.yml
DATABASE_URL = f"sqlite:///{os.path.join(DATABASE_DIR, 'npsketch.db')}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TrainingDataImage(Base):
    """
    The single table for all training / test images and their CNN labels.

    ATTRIBUTES:
    -----------
    Core Identity:
        id: Primary key
        uid: Stable unique key from the consolidated base (templates/labels.csv),
             e.g. "TF-2020_08_27-257-COPY", "ALG-PC0001-COPY". Traceable to img/<uid>.<ext>.
        patient_id: Patient identifier, source-prefixed, groups COPY+RECALL of one
             patient (e.g. TF-257, ALG-PC0001, OXF-C0078, OCSM-PC0460)
        task_type: COPY | RECALL (condition) — REFERENCE for templates
        source_format: TELEFRED | OXFORD | ALGORITHM | OCS_MACHINE (+ legacy MAT/OCS/DRAWN)
        test_name: Human-readable name (for drawn images)
    
    Image Data:
        original_filename: Original uploaded filename
        original_file_data: Original uploaded file (BLOB)
        processed_image_data: Normalized image 568×274, 2px lines (BLOB)
        image_hash: SHA256 hash for duplicate detection
        extraction_metadata: JSON with technical details
    
    Clinical Features (for CNN Training):
        features_data: JSON. Holistic score plus optional per-component sub-labels
            (the OCS-Plus 20 elements x 3 binary aspects). Shape:
                {"Total_Score": 45,
                 "components": {"presence":[..20..], "accuracy":[..20..], "position":[..20..]}}
            "components" is null when the source has only a holistic score (OXFORD).

    Metadata:
        session_id: Upload session identifier
        uploaded_at: Timestamp
    """
    __tablename__ = "training_data_images"
    
    # Core identity
    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String, unique=True, index=True, nullable=True)  # from templates/labels.csv
    patient_id = Column(String, index=True)
    task_type = Column(String, index=True)  # COPY, RECALL, REFERENCE
    source_format = Column(String, index=True)  # TELEFRED, OXFORD, ALGORITHM, OCS_MACHINE (legacy: MAT/OCS/DRAWN)
    test_name = Column(String, nullable=True, index=True)  # Human-readable name
    
    # Image data
    original_filename = Column(String)
    original_file_data = Column(LargeBinary)
    processed_image_data = Column(LargeBinary)
    image_hash = Column(String(64), index=True)
    extraction_metadata = Column(String, nullable=True)  # JSON

    # CNN training labels (nullable - only if available)
    features_data = Column(String, nullable=True)  # JSON: Total_Score + optional components
    # Best-model prediction, stored IN PARALLEL with the human features_data (never overwritten
    # by human edits). JSON: {model, Total_Score, components:{presence,accuracy,position}}.
    model_prediction = Column(String, nullable=True)
    # Write-protect: True once a human has manually validated/corrected the labels. Protected
    # rows must survive DB cleans (e.g. dropping synthetic) and not be overwritten by reimport.
    validated = Column(Boolean, default=False, index=True)
    validated_at = Column(DateTime, nullable=True)

    # Metadata
    session_id = Column(String, index=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, index=True)  # Index for ORDER BY performance
    
    def __repr__(self):
        return f"<TrainingDataImage(id={self.id}, patient={self.patient_id}, task={self.task_type})>"


def init_database():
    """
    Initialize the database by creating all tables.
    Should be called at application startup.
    """
    Base.metadata.create_all(bind=engine)
    
    # Run migrations for new columns (SQLite doesn't support IF NOT EXISTS for columns)
    _run_migrations()


def _run_migrations():
    """
    Add new columns to existing tables if they don't exist.
    Also creates indexes for performance.
    """
    from sqlalchemy import text
    
    migrations = [
        ("training_data_images", "model_prediction", "TEXT"),
        ("training_data_images", "validated", "BOOLEAN DEFAULT 0"),
        ("training_data_images", "validated_at", "DATETIME"),
    ]

    # Indexes to create for performance
    indexes = [
        # Index on uploaded_at for ORDER BY optimization (critical for large datasets)
        ("ix_training_data_images_uploaded_at", "training_data_images", "uploaded_at"),
    ]
    
    with engine.connect() as conn:
        for table, column, col_type in migrations:
            try:
                # Check if column exists
                result = conn.execute(text(f"SELECT {column} FROM {table} LIMIT 1"))
            except Exception:
                # Column doesn't exist, add it
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                    conn.commit()
                    print(f"✓ Added column {column} to {table}")
                except Exception as e:
                    print(f"Warning: Could not add column {column} to {table}: {e}")
        
        # Create indexes
        for index_name, table, column in indexes:
            try:
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({column})"))
                conn.commit()
                print(f"✓ Created index {index_name}")
            except Exception as e:
                # Index may already exist or other error
                pass


def get_db():
    """
    Dependency function to get database session.
    Use with FastAPI's Depends().
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

