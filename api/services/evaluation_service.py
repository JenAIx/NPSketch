"""
Evaluation service for comparing images.

This module handles the evaluation workflow: processing uploaded images,
comparing them to references, and storing results.
"""

import os
from sqlalchemy.orm import Session
from typing import Optional
import numpy as np

from database import UploadedImage, ExtractedFeature, EvaluationResult, ReferenceImage
from image_processing import (
    load_image_from_bytes,
    normalize_image,
    image_to_bytes,
    create_visualization,
    LineDetector,
    LineComparator,
    ImageRegistration
)


class EvaluationService:
    """
    Service for evaluating uploaded images against references.
    
    Handles the complete workflow from upload to evaluation and visualization.
    """
    
    def __init__(self, db: Session, use_registration: bool = True, registration_motion: str = "euclidean", max_rotation_degrees: float = 30.0):
        """
        Initialize the evaluation service.
        
        Args:
            db: SQLAlchemy database session
            use_registration: Whether to use image registration for alignment
            registration_motion: Motion model ('euclidean', 'affine', 'translation')
            max_rotation_degrees: Maximum allowed rotation in degrees
        """
        self.db = db
        self.line_detector = LineDetector()
        self.comparator = LineComparator()
        self.registration = ImageRegistration()
        self.use_registration = use_registration
        self.registration_motion = registration_motion
        self.max_rotation_degrees = max_rotation_degrees
        
        # Create directories for storing visualizations
        # Static files are stored in the mounted data directory
        self.viz_dir = '/app/data/visualizations'
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def process_upload(
        self,
        image_bytes: bytes,
        filename: str,
        reference_name: str = "default_reference",
        uploader: Optional[str] = None
    ) -> tuple[UploadedImage, EvaluationResult]:
        """
        Process an uploaded image and evaluate it against a reference.
        
        Args:
            image_bytes: Image data as bytes
            filename: Original filename
            reference_name: Name of the reference to compare against
            uploader: Optional uploader identifier
            
        Returns:
            Tuple of (UploadedImage, EvaluationResult)
        """
        # Load and normalize image
        image = load_image_from_bytes(image_bytes)
        normalized = normalize_image(image)
        
        # Store uploaded image
        uploaded_image = UploadedImage(
            filename=filename,
            image_data=image_bytes,
            processed_image_data=image_to_bytes(normalized),
            uploader=uploader
        )
        self.db.add(uploaded_image)
        self.db.commit()
        self.db.refresh(uploaded_image)
        
        # Extract features
        features = self.line_detector.extract_features(normalized)
        feature_json = self.line_detector.features_to_json(features)
        
        # Store extracted features
        extracted_feature = ExtractedFeature(
            image_id=uploaded_image.id,
            feature_data=feature_json,
            num_lines=features['num_lines']
        )
        self.db.add(extracted_feature)
        self.db.commit()
        
        # Get reference image
        reference = self.db.query(ReferenceImage).filter(
            ReferenceImage.name == reference_name
        ).first()
        
        if not reference:
            raise ValueError(f"Reference '{reference_name}' not found")
        
        # Compare with reference
        reference_features = self.line_detector.features_from_json(reference.feature_data)
        comparison = self.comparator.compare_lines(
            features['lines'],
            reference_features['lines']
        )
        
        # Create visualization
        viz_filename = f"eval_{uploaded_image.id}.png"
        viz_path = os.path.join(self.viz_dir, viz_filename)
        
        ref_image = load_image_from_bytes(reference.processed_image_data)
        visualization = self._create_comparison_visualization(
            normalized,
            ref_image,
            features['lines'],
            reference_features['lines'],
            comparison
        )
        
        import cv2
        cv2.imwrite(viz_path, visualization)
        
        # Store evaluation result
        evaluation = EvaluationResult(
            image_id=uploaded_image.id,
            reference_id=reference.id,
            correct_lines=comparison['correct_lines'],
            missing_lines=comparison['missing_lines'],
            extra_lines=comparison['extra_lines'],
            similarity_score=comparison['similarity_score'],
            visualization_path=f"/api/visualizations/{viz_filename}"
        )
        self.db.add(evaluation)
        self.db.commit()
        self.db.refresh(evaluation)
        
        return uploaded_image, evaluation
    
    def _create_comparison_visualization(
        self,
        uploaded_img: np.ndarray,
        reference_img: np.ndarray,
        detected_lines: list,
        reference_lines: list,
        comparison: dict
    ) -> np.ndarray:
        """
        Create a side-by-side comparison visualization.
        
        Args:
            uploaded_img: Uploaded image
            reference_img: Reference image
            detected_lines: Detected lines from upload
            reference_lines: Lines from reference
            comparison: Comparison results
            
        Returns:
            Combined visualization image
        """
        import cv2
        
        # Create copies for drawing
        upload_vis = uploaded_img.copy()
        ref_vis = reference_img.copy()
        
        # Draw detected lines on uploaded image
        # Green for matched lines, red for extra lines
        matched_indices = set(comparison['matched_detected_indices'])
        for i, line in enumerate(detected_lines):
            x1, y1, x2, y2 = line
            color = (0, 255, 0) if i in matched_indices else (0, 0, 255)
            cv2.line(upload_vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw reference lines
        # Green for matched, blue for missing
        matched_ref = set(comparison['matched_reference_indices'])
        for i, line in enumerate(reference_lines):
            x1, y1, x2, y2 = line
            color = (0, 255, 0) if i in matched_ref else (255, 0, 0)
            cv2.line(ref_vis, (x1, y1), (x2, y2), color, 2)
        
        # Combine side by side
        combined = np.hstack([upload_vis, ref_vis])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Your Drawing", (10, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(combined, "Reference", (upload_vis.shape[1] + 10, 30), font, 1, (0, 0, 0), 2)
        
        # Add metrics at bottom
        metrics_text = f"Correct: {comparison['correct_lines']} | Missing: {comparison['missing_lines']} | Extra: {comparison['extra_lines']} | Score: {comparison['similarity_score']:.2%}"
        cv2.putText(combined, metrics_text, (10, combined.shape[0] - 20), font, 0.6, (0, 0, 0), 2)
        
        return combined
    
    def get_evaluation_by_id(self, eval_id: int) -> Optional[EvaluationResult]:
        """
        Retrieve an evaluation result by ID.
        
        Args:
            eval_id: Evaluation ID
            
        Returns:
            EvaluationResult or None
        """
        return self.db.query(EvaluationResult).filter(
            EvaluationResult.id == eval_id
        ).first()
    
    def list_recent_evaluations(self, limit: int = 10):
        """
        List recent evaluations.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of EvaluationResult records
        """
        return self.db.query(EvaluationResult).order_by(
            EvaluationResult.evaluated_at.desc()
        ).limit(limit).all()
    
    def evaluate_test_image(
        self,
        image: np.ndarray,
        reference_id: int,
        test_name: str
    ) -> EvaluationResult:
        """
        Evaluate a test image against a reference.
        Does not store the image itself, only creates an evaluation result.
        
        Args:
            image: Test image as numpy array
            reference_id: Reference image ID
            test_name: Name of the test
            
        Returns:
            EvaluationResult
        """
        # Normalize image
        normalized = normalize_image(image)
        
        # Get reference image
        reference = self.db.query(ReferenceImage).filter(
            ReferenceImage.id == reference_id
        ).first()
        
        if not reference:
            raise ValueError(f"Reference with ID {reference_id} not found")
        
        ref_image = load_image_from_bytes(reference.processed_image_data)
        
        # Perform image registration if enabled
        registered_image = normalized
        registration_info = {'used': False}
        
        if self.use_registration:
            try:
                registered_image, registration_info = self.registration.register_images(
                    normalized,
                    ref_image,
                    method="ecc",
                    motion_type=self.registration_motion,
                    max_rotation_degrees=self.max_rotation_degrees
                )
                # Only mark as 'used' if registration was successful
                registration_info['used'] = registration_info.get('success', False)
                registration_info['motion_type'] = self.registration_motion
            except Exception as e:
                # If registration fails, use original
                registered_image = normalized
                registration_info = {
                    'used': False,
                    'error': str(e)
                }
        
        # Extract features from registered image
        features = self.line_detector.extract_features(registered_image)
        
        # Compare with reference
        reference_features = self.line_detector.features_from_json(reference.feature_data)
        comparison = self.comparator.compare_lines(
            features['lines'],
            reference_features['lines']
        )
        
        # Create 3-way visualization: Original | Registered | Reference
        viz_filename = f"test_{test_name}.png"
        viz_path = os.path.join(self.viz_dir, viz_filename)
        
        visualization = self._create_3way_comparison_visualization(
            normalized,
            registered_image,
            ref_image,
            features['lines'],
            reference_features['lines'],
            comparison,
            registration_info
        )
        
        import cv2
        cv2.imwrite(viz_path, visualization)
        
        # Create evaluation result (without storing in DB yet)
        evaluation = EvaluationResult(
            image_id=None,  # No uploaded image for tests
            reference_id=reference.id,
            correct_lines=comparison['correct_lines'],
            missing_lines=comparison['missing_lines'],
            extra_lines=comparison['extra_lines'],
            similarity_score=comparison['similarity_score'],
            visualization_path=f"/api/visualizations/{viz_filename}"
        )
        
        return evaluation
    
    def _create_3way_comparison_visualization(
        self,
        original_img: np.ndarray,
        registered_img: np.ndarray,
        reference_img: np.ndarray,
        detected_lines: list,
        reference_lines: list,
        comparison: dict,
        registration_info: dict
    ) -> np.ndarray:
        """
        Create a 3-way comparison visualization showing Original | Registered | Reference.
        
        Args:
            original_img: Original uploaded image
            registered_img: Image after registration
            reference_img: Reference image
            detected_lines: Detected lines from registered image
            reference_lines: Lines from reference
            comparison: Comparison results
            registration_info: Information about the registration
            
        Returns:
            Combined visualization image
        """
        import cv2
        
        # Create copies for drawing
        orig_vis = original_img.copy()
        reg_vis = registered_img.copy()
        ref_vis = reference_img.copy()
        
        # Draw detected lines on registered image
        # Green for matched lines, red for extra lines
        matched_indices = set(comparison['matched_detected_indices'])
        for i, line in enumerate(detected_lines):
            x1, y1, x2, y2 = line
            color = (0, 255, 0) if i in matched_indices else (0, 0, 255)
            cv2.line(reg_vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw reference lines
        # Green for matched, blue for missing
        matched_ref = set(comparison['matched_reference_indices'])
        for i, line in enumerate(reference_lines):
            x1, y1, x2, y2 = line
            color = (0, 255, 0) if i in matched_ref else (255, 0, 0)
            cv2.line(ref_vis, (x1, y1), (x2, y2), color, 2)
        
        # Combine three images horizontally
        combined = np.hstack([orig_vis, reg_vis, ref_vis])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 30), font, 0.8, (0, 0, 0), 2)
        cv2.putText(combined, "Registered", (orig_vis.shape[1] + 10, 30), font, 0.8, (0, 0, 0), 2)
        cv2.putText(combined, "Reference", (orig_vis.shape[1] + reg_vis.shape[1] + 10, 30), font, 0.8, (0, 0, 0), 2)
        
        # Add registration info if used
        if registration_info.get('used', False):
            reg_text = f"Tx:{registration_info.get('translation_x', 0):.1f} Ty:{registration_info.get('translation_y', 0):.1f} Rot:{registration_info.get('rotation_degrees', 0):.1f}deg Scale:{registration_info.get('scale', 1.0):.2f}x"
            cv2.putText(combined, reg_text, (orig_vis.shape[1] + 10, 50), font, 0.5, (0, 0, 255), 1)
        
        # Add metrics at bottom
        metrics_text = f"Correct: {comparison['correct_lines']} | Missing: {comparison['missing_lines']} | Extra: {comparison['extra_lines']} | Score: {comparison['similarity_score']:.2%}"
        cv2.putText(combined, metrics_text, (10, combined.shape[0] - 20), font, 0.6, (0, 0, 0), 2)
        
        return combined

