"""
Database models and setup for NPSketch.

This module defines the SQLAlchemy models for storing:
- Reference images and their features
- Uploaded images for comparison
- Extracted features from uploaded images
- Evaluation results
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


class ReferenceImage(Base):
    """
    Stores reference images and their extracted features.
    
    Attributes:
        id: Primary key
        name: Reference image name (e.g., "house_of_nikolaus")
        image_data: Original image as binary blob
        processed_image_data: Preprocessed/normalized image
        feature_data: Extracted line features as JSON
        created_at: Timestamp of creation
    """
    __tablename__ = "reference_images"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    image_data = Column(LargeBinary)
    processed_image_data = Column(LargeBinary)
    feature_data = Column(String)  # JSON string
    width = Column(Integer)
    height = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class UploadedImage(Base):
    """
    Stores uploaded images for comparison.
    
    Attributes:
        id: Primary key
        filename: Original filename
        image_data: Original uploaded image
        processed_image_data: Preprocessed/normalized image
        uploader: Optional uploader identifier
        uploaded_at: Timestamp of upload
    """
    __tablename__ = "uploaded_images"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    image_data = Column(LargeBinary)
    processed_image_data = Column(LargeBinary)
    uploader = Column(String, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    features = relationship("ExtractedFeature", back_populates="image")
    evaluations = relationship("EvaluationResult", back_populates="image")


class ExtractedFeature(Base):
    """
    Stores extracted features from uploaded images.
    
    Attributes:
        id: Primary key
        image_id: Foreign key to uploaded_images
        feature_data: Extracted line features as JSON
        num_lines: Number of detected lines
        extracted_at: Timestamp of extraction
    """
    __tablename__ = "extracted_features"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("uploaded_images.id"))
    feature_data = Column(String)  # JSON string
    num_lines = Column(Integer)
    extracted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    image = relationship("UploadedImage", back_populates="features")


class EvaluationResult(Base):
    """
    Stores evaluation results comparing uploaded images to reference.
    
    Includes both automated detection results and optional user evaluation
    for AI training purposes.
    
    Attributes:
        id: Primary key
        image_id: Foreign key to uploaded_images
        reference_id: Foreign key to reference_images
        correct_lines: Number of correctly matched lines (automated detection)
        missing_lines: Number of lines present in reference but missing (automated)
        extra_lines: Number of extra lines not in reference (automated)
        similarity_score: Overall similarity score (0-1)
        visualization_path: Path to visualization image
        evaluated_at: Timestamp of evaluation
        user_evaluated: Whether user has manually evaluated this result
        evaluated_correct: User-evaluated correct lines count
        evaluated_missing: User-evaluated missing lines count
        evaluated_extra: User-evaluated extra lines count
        evaluated_at_timestamp: When user evaluation was performed
    """
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("uploaded_images.id"))
    reference_id = Column(Integer, ForeignKey("reference_images.id"))
    
    # Automated detection results
    correct_lines = Column(Integer)
    missing_lines = Column(Integer)
    extra_lines = Column(Integer)
    similarity_score = Column(Float)
    visualization_path = Column(String, nullable=True)
    evaluated_at = Column(DateTime, default=datetime.utcnow)
    
    # User evaluation for AI training (optional)
    user_evaluated = Column(Boolean, default=False, nullable=False)
    evaluated_correct = Column(Integer, nullable=True)
    evaluated_missing = Column(Integer, nullable=True)
    evaluated_extra = Column(Integer, nullable=True)
    evaluated_at_timestamp = Column(DateTime, nullable=True)
    
    # Relationships
    image = relationship("UploadedImage", back_populates="evaluations")
    reference = relationship("ReferenceImage")


class TestImage(Base):
    """
    Stores manually created test images with expected scores for testing.
    
    Attributes:
        id: Primary key
        test_name: Name/description of the test
        image_data: Image binary data
        expected_correct: Expected number of correct lines
        expected_missing: Expected number of missing lines
        expected_extra: Expected number of extra lines
        created_at: Timestamp of creation
    """
    __tablename__ = "test_images"
    
    id = Column(Integer, primary_key=True, index=True)
    test_name = Column(String, unique=True, index=True)
    image_data = Column(LargeBinary)
    expected_correct = Column(Integer)
    expected_missing = Column(Integer)
    expected_extra = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<TestImage(id={self.id}, name='{self.test_name}')>"


def init_database():
    """
    Initialize the database by creating all tables.
    Should be called at application startup.
    """
    Base.metadata.create_all(bind=engine)


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

