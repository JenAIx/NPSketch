"""
Synthetic Bad Image Generator

Generates realistic bad-quality training images to address data imbalance
for low scores/classes in neuropsychological drawing assessment.

Combines multiple strategies:
1. Lines from real bad images (score < 20)
2. Modified reference lines (shortened, shifted)
3. Random straight lines (horizontal/vertical/diagonal)
4. Modifications: Curves (Bezier), Tremor (hand shake), Distortions

Uses 5 complexity levels to create a spectrum from simple to complex bad images.
"""

import numpy as np
import cv2
import json
import io
from typing import List, Dict, Tuple
from PIL import Image
from utils.logger import get_logger

logger = get_logger(__name__)


class LinePoolGenerator:
    """
    Generates synthetic bad images from multiple line sources.
    
    Line Sources:
    - Real bad lines: Extracted from images with score < 20
    - Reference lines: From reference image (modified)
    - Random lines: Procedurally generated
    """
    
    def __init__(self):
        self.real_bad_lines = []
        self.reference_lines = []
    
    def load_real_bad_lines(self, db, score_threshold: float = 20.0):
        """
        Extract lines from real bad-quality images.
        
        Args:
            db: Database session
            score_threshold: Maximum score to consider (default: 20.0)
        """
        from image_processing import LineDetector
        from database import TrainingDataImage
        
        detector = LineDetector()
        
        logger.info(f"Extracting lines from real bad images (score < {score_threshold})...")
        
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        count = 0
        for img in images:
            try:
                features = json.loads(img.features_data)
                if 'Total_Score' in features and features['Total_Score'] < score_threshold:
                    pil_img = Image.open(io.BytesIO(img.processed_image_data))
                    img_array = np.array(pil_img)
                    
                    extracted = detector.extract_features(img_array)
                    lines = extracted.get('lines', [])
                    
                    if len(lines) > 0:
                        self.real_bad_lines.extend(lines)
                        count += 1
            except:
                continue
        
        logger.info(f"Loaded {len(self.real_bad_lines)} lines from {count} bad images")
    
    def load_reference_lines(self, db):
        """
        Extract lines from reference image.
        
        Args:
            db: Database session
        """
        from database import ReferenceImage
        
        ref = db.query(ReferenceImage).first()
        if ref and ref.feature_data:
            features = json.loads(ref.feature_data)
            self.reference_lines = features.get('lines', [])
            logger.info(f"Loaded {len(self.reference_lines)} lines from reference")
    
    def generate_random_lines(
        self,
        num_lines: int,
        image_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate random straight lines (horizontal/vertical/diagonal).
        
        Args:
            num_lines: Number of lines to generate
            image_size: (width, height) of target image
        
        Returns:
            List of lines [(x1, y1, x2, y2), ...]
        """
        width, height = image_size
        lines = []
        
        for _ in range(num_lines):
            orientation = np.random.choice(['h', 'v', 'd'], p=[0.4, 0.3, 0.3])
            
            if orientation == 'h':  # Horizontal
                y = np.random.randint(20, height - 20)
                x1 = np.random.randint(20, width - 100)
                x2 = x1 + np.random.randint(50, 200)
                x2 = min(x2, width - 20)
                lines.append((x1, y, x2, y))
            
            elif orientation == 'v':  # Vertical
                x = np.random.randint(20, width - 20)
                y1 = np.random.randint(20, height - 100)
                y2 = y1 + np.random.randint(50, 150)
                y2 = min(y2, height - 20)
                lines.append((x, y1, x, y2))
            
            else:  # Diagonal
                x1 = np.random.randint(20, width - 100)
                y1 = np.random.randint(20, height - 100)
                length = np.random.randint(50, 150)
                angle = np.random.uniform(-np.pi, np.pi)
                x2 = int(x1 + length * np.cos(angle))
                y2 = int(y1 + length * np.sin(angle))
                x2 = np.clip(x2, 20, width - 20)
                y2 = np.clip(y2, 20, height - 20)
                lines.append((x1, y1, x2, y2))
        
        return lines
    
    def add_tremor(
        self,
        points: List[Tuple[int, int]],
        strength: float
    ) -> List[Tuple[int, int]]:
        """
        Add hand tremor (wobble) to line points.
        
        Args:
            points: List of (x, y) points
            strength: Tremor strength in pixels (1-3 realistic)
        
        Returns:
            Points with tremor added
        """
        tremored = []
        for i, (x, y) in enumerate(points):
            # Less tremor at endpoints
            if i == 0 or i == len(points) - 1:
                dx = np.random.normal(0, strength * 0.3)
                dy = np.random.normal(0, strength * 0.3)
            else:
                dx = np.random.normal(0, strength)
                dy = np.random.normal(0, strength)
            tremored.append((int(x + dx), int(y + dy)))
        return tremored
    
    def make_curved(
        self,
        line: Tuple[int, int, int, int],
        curvature: float
    ) -> List[Tuple[int, int]]:
        """
        Convert straight line to curved line using Bezier curve.
        
        Args:
            line: (x1, y1, x2, y2)
            curvature: 0.0 (straight) to 1.0 (very curved)
        
        Returns:
            List of points along the curve
        """
        x1, y1, x2, y2 = line
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 1:
            return [(x1, y1), (x2, y2)]
        
        # Perpendicular direction for control point
        perp_x, perp_y = -dy / length, dx / length
        offset = length * curvature
        control_x = mid_x + perp_x * offset
        control_y = mid_y + perp_y * offset
        
        # Generate Bezier curve
        points = []
        num_points = max(10, int(length / 10))
        for i in range(num_points + 1):
            t = i / num_points
            # Quadratic Bezier formula
            x = (1-t)**2 * x1 + 2*(1-t)*t * control_x + t**2 * x2
            y = (1-t)**2 * y1 + 2*(1-t)*t * control_y + t**2 * y2
            points.append((int(x), int(y)))
        
        return points
    
    def generate_synthetic_bad_image(
        self,
        complexity: float,
        image_size: Tuple[int, int],
        random_seed: int = None
    ) -> np.ndarray:
        """
        Generate synthetic bad-quality image.
        
        Complexity levels (0.0 - 1.0):
        - 0.0: Simple, few straight lines from real bad images
        - 0.25: Mostly real bad lines with minimal modifications
        - 0.5: Mix of real/reference/random with moderate modifications
        - 0.75: More random and complex modifications
        - 1.0: Complex mix with strong curves and tremor
        
        Args:
            complexity: 0.0 (simple) to 1.0 (complex)
            image_size: (width, height)
            random_seed: Random seed for reproducibility
        
        Returns:
            Image as numpy array (H×W×3, RGB)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        width, height = image_size
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Determine line composition based on complexity
        total_lines = int(5 + complexity * 15)  # 5-20 lines
        
        if complexity < 0.3:
            # Low complexity: mostly real bad lines, straight
            real_bad_ratio, ref_ratio, random_ratio = 0.8, 0.2, 0.0
        elif complexity < 0.7:
            # Medium complexity: balanced mix
            real_bad_ratio, ref_ratio, random_ratio = 0.5, 0.3, 0.2
        else:
            # High complexity: more random and modifications
            real_bad_ratio, ref_ratio, random_ratio = 0.3, 0.3, 0.4
        
        num_real = int(total_lines * real_bad_ratio)
        num_ref = int(total_lines * ref_ratio)
        num_random = total_lines - num_real - num_ref
        
        all_lines = []
        
        # 1. Add lines from real bad images
        if len(self.real_bad_lines) > 0 and num_real > 0:
            indices = np.random.choice(
                len(self.real_bad_lines),
                min(num_real, len(self.real_bad_lines)),
                replace=False
            )
            all_lines.extend([self.real_bad_lines[idx] for idx in indices])
        
        # 2. Add reference lines (sometimes modified)
        if len(self.reference_lines) > 0 and num_ref > 0:
            indices = np.random.choice(
                len(self.reference_lines),
                min(num_ref, len(self.reference_lines)),
                replace=False
            )
            for idx in indices:
                line = self.reference_lines[idx]
                
                # Sometimes shorten reference lines at higher complexity
                if complexity > 0.5 and np.random.random() < 0.4:
                    x1, y1, x2, y2 = line
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    factor = np.random.uniform(0.6, 0.9)
                    line = (
                        int(mid_x + (x1 - mid_x) * factor),
                        int(mid_y + (y1 - mid_y) * factor),
                        int(mid_x + (x2 - mid_x) * factor),
                        int(mid_y + (y2 - mid_y) * factor)
                    )
                
                all_lines.append(line)
        
        # 3. Add random lines
        if num_random > 0:
            all_lines.extend(self.generate_random_lines(num_random, image_size))
        
        # Draw all lines with optional modifications
        modification_chance = complexity * 0.6  # Max 60% at full complexity
        
        for line in all_lines:
            x1, y1, x2, y2 = line
            
            # Clip to image bounds
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            
            # Skip invalid lines
            if (x1, y1) == (x2, y2):
                continue
            
            # Decide on modifications
            apply_curve = np.random.random() < modification_chance * 0.5
            apply_tremor = np.random.random() < modification_chance
            
            if apply_curve:
                # Make line curved
                curvature = np.random.uniform(0.1, 0.4) * complexity
                points = self.make_curved((x1, y1, x2, y2), curvature)
                
                if apply_tremor:
                    tremor_strength = 1.5 + complexity * 1.5
                    points = self.add_tremor(points, tremor_strength)
                
                # Draw curved line
                for i in range(len(points) - 1):
                    p1 = (
                        max(0, min(width - 1, points[i][0])),
                        max(0, min(height - 1, points[i][1]))
                    )
                    p2 = (
                        max(0, min(width - 1, points[i+1][0])),
                        max(0, min(height - 1, points[i+1][1]))
                    )
                    cv2.line(img, p1, p2, (0, 0, 0), 2)
            
            elif apply_tremor:
                # Straight line with tremor
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                num_points = max(5, int(length / 10))
                points = [
                    (int(x1 + t / num_points * (x2 - x1)), int(y1 + t / num_points * (y2 - y1)))
                    for t in range(num_points + 1)
                ]
                
                tremor_strength = 1.0 + complexity * 2.0
                points = self.add_tremor(points, tremor_strength)
                
                for i in range(len(points) - 1):
                    p1 = (
                        max(0, min(width - 1, points[i][0])),
                        max(0, min(height - 1, points[i][1]))
                    )
                    p2 = (
                        max(0, min(width - 1, points[i+1][0])),
                        max(0, min(height - 1, points[i+1][1]))
                    )
                    cv2.line(img, p1, p2, (0, 0, 0), 2)
            
            else:
                # Straight line, no modifications
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        # Normalize line thickness to match training data
        from line_normalizer import normalize_line_thickness
        img = normalize_line_thickness(img, target_thickness=2.0)
        
        return img


def generate_synthetic_bad_images(
    db,
    n_samples: int = 50,
    complexity_levels: int = 5,
    score_threshold: float = 20.0,
    image_size: Tuple[int, int] = (568, 274),
    random_seed: int = 42
) -> List[Dict]:
    """
    Generate a dataset of synthetic bad-quality images.
    
    Args:
        db: Database session
        n_samples: Total number of images to generate
        complexity_levels: Number of complexity levels (5 recommended)
        score_threshold: Maximum score for real bad images extraction
        image_size: Output image size (width, height)
        random_seed: Base random seed
    
    Returns:
        List of dicts with 'image_data' (bytes) and metadata
    """
    logger.info(f"Generating {n_samples} synthetic bad images...")
    
    # Initialize generator
    generator = LinePoolGenerator()
    generator.load_real_bad_lines(db, score_threshold=score_threshold)
    generator.load_reference_lines(db)
    
    if len(generator.real_bad_lines) == 0:
        logger.warning("No real bad lines found! Using reference and random only.")
    
    # Generate images across complexity levels
    synthetic_images = []
    samples_per_level = n_samples // complexity_levels
    remainder = n_samples % complexity_levels
    
    # Distribute remainder across levels (add to later levels for more complex images)
    logger.debug(f"Distribution: {samples_per_level} per level, +{remainder} to last levels")
    
    for level in range(complexity_levels):
        complexity = level / (complexity_levels - 1) if complexity_levels > 1 else 0.5
        
        # Add one extra to later levels if we have remainder
        num_samples_this_level = samples_per_level
        if level >= (complexity_levels - remainder):
            num_samples_this_level += 1
        
        logger.info(f"Level {level + 1}/{complexity_levels} (complexity={complexity:.2f}): Generating {num_samples_this_level} images...")
        
        for i in range(num_samples_this_level):
            # Generate image
            img_array = generator.generate_synthetic_bad_image(
                complexity=complexity,
                image_size=image_size,
                random_seed=random_seed + level * 1000 + i
            )
            
            # Convert to PNG bytes
            img_pil = Image.fromarray(img_array)
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            synthetic_images.append({
                'image_data': img_bytes,
                'complexity_level': level,
                'complexity_value': complexity
            })
    
    logger.info(f"Generated {len(synthetic_images)} synthetic bad images (requested: {n_samples})")
    
    # Verify we generated exactly n_samples
    if len(synthetic_images) != n_samples:
        logger.warning(f"Generated {len(synthetic_images)} images but {n_samples} were requested!")
    
    return synthetic_images

