"""
Reference image management service.

This module handles initialization, storage, and retrieval of reference images
and their extracted features.
"""

import os
from sqlalchemy.orm import Session
from typing import Optional
import numpy as np

from database import ReferenceImage
from image_processing import (
    load_image_from_bytes,
    normalize_image,
    image_to_bytes,
    LineDetector
)


class ReferenceService:
    """
    Service for managing reference images.
    
    Handles loading, processing, and storing reference images and their features.
    """
    
    def __init__(self, db: Session):
        """
        Initialize the reference service.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self.line_detector = LineDetector()
    
    def initialize_default_reference(self, name: str = "default_reference"):
        """
        Initialize a default reference image (simple geometric shape).
        
        Creates a simple "House of Nikolaus" pattern as default reference.
        
        Args:
            name: Name for the reference image
        """
        # Check if reference already exists
        existing = self.get_reference_by_name(name)
        if existing:
            return existing
        
        # Create a simple reference image (House of Nikolaus pattern)
        img = self._create_default_pattern()
        
        # Store the reference
        return self.store_reference(name, img)
    
    def _create_default_pattern(self) -> np.ndarray:
        """
        Create default reference pattern (House of Nikolaus).
        
        Returns:
            Image as numpy array
        """
        import cv2
        
        # Create white canvas
        img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # Define the House of Nikolaus pattern
        # This is a classic drawing puzzle
        points = [
            (156, 356),  # Bottom left
            (356, 356),  # Bottom right
            (356, 156),  # Top right of square
            (156, 156),  # Top left of square
            (256, 56),   # Roof peak
            (356, 156),  # Back to top right
            (156, 356),  # Diagonal to bottom left
            (156, 156),  # Up to top left
            (256, 56),   # To roof peak
            (356, 356),  # Diagonal to bottom right
        ]
        
        # Draw lines
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], (0, 0, 0), 3)
        
        return img
    
    def store_reference(self, name: str, image: np.ndarray) -> ReferenceImage:
        """
        Store a reference image and extract its features.
        
        Args:
            name: Name for the reference image
            image: Image as numpy array
            
        Returns:
            Created ReferenceImage database record
        """
        # Normalize image
        normalized = normalize_image(image)
        
        # Extract features
        features = self.line_detector.extract_features(normalized)
        feature_json = self.line_detector.features_to_json(features)
        
        # Convert images to bytes
        original_bytes = image_to_bytes(image)
        normalized_bytes = image_to_bytes(normalized)
        
        # Create database record
        ref_image = ReferenceImage(
            name=name,
            image_data=original_bytes,
            processed_image_data=normalized_bytes,
            feature_data=feature_json,
            width=normalized.shape[1],
            height=normalized.shape[0]
        )
        
        self.db.add(ref_image)
        self.db.commit()
        self.db.refresh(ref_image)
        
        return ref_image
    
    def get_reference_by_name(self, name: str) -> Optional[ReferenceImage]:
        """
        Retrieve a reference image by name.
        
        Args:
            name: Name of the reference image
            
        Returns:
            ReferenceImage or None if not found
        """
        return self.db.query(ReferenceImage).filter(
            ReferenceImage.name == name
        ).first()
    
    def get_reference_by_id(self, ref_id: int) -> Optional[ReferenceImage]:
        """
        Retrieve a reference image by ID.
        
        Args:
            ref_id: ID of the reference image
            
        Returns:
            ReferenceImage or None if not found
        """
        return self.db.query(ReferenceImage).filter(
            ReferenceImage.id == ref_id
        ).first()
    
    def list_all_references(self):
        """
        List all reference images.
        
        Returns:
            List of ReferenceImage records
        """
        return self.db.query(ReferenceImage).all()

