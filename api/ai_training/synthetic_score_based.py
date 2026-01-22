"""
Synthetic Score-Based Image Generator

Purpose
-------
Generate synthetic training images with specific Total_Score targets based on
how human raters evaluate drawings.

Scoring System (per feature, integer)
-------------------------------------
- Presence (0/1): Is the feature there?
- Position (0/1): Is it in the correct location? (offset < 25px)
- Accuracy (0/1): Is it well-drawn? (tremor/curvature/shortening threshold)

Total: num_features × 3 = max score (e.g., 21 features × 3 = 63 max)

Strategies by target score
--------------------------
- 0-5: random lines only, away from features
- 6-15: 2-5 features, poor position/accuracy
- 16-25: 5-8 features, moderate quality
- 26-35: 8-12 features, mixed quality
- 36-45: 12-15 features, decent quality
- 46-55: 15-18 features, good quality
- 56-63: 18-21 features, excellent quality

Feature selection
-----------------
- Preferred features are weighted higher (frame lines).
- Subsequent features prefer nearby features (proximity bias).
- Random extra lines avoid feature zones and feature angles.

Padding / pre-shrink
--------------------
A 15px padding is enforced by pre-shrinking reference features around the center.
All random lines are also constrained to the padded area.

Usage (inside Docker)
---------------------
docker exec -e PYTHONPATH=/app npsketch-api \\
  python3 /app/ai_training/synthetic_score_based.py \\
    --scores 0,10,20,30,40 \\
    --samples-per-score 10

Options
-------
--output-dir        Output directory (default: /app/data/tmp/synthetic_images)
--scores            Comma-separated target scores (default: 0,5,10,15,20,25,30,35,40,45,50,55,60)
--samples-per-score Number of images per score (default: 3)
--random-seed       Base seed for reproducibility (default: 42)

Outputs
-------
For each image, a PNG and JSON are written. JSON fields include:
- target_score (actual score), target_value (actual score), batch_score
- metadata: correct_pos_num, correct_accuracy_num, correct_exists_num
"""

import numpy as np
import cv2
import json
import io
import os
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
from dataclasses import dataclass, field


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FeatureLine:
    """Represents a single feature line from reference."""
    index: int
    x1: int
    y1: int
    x2: int
    y2: int
    length: float
    angle: float
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class DrawnFeature:
    """A feature as it will be drawn (possibly modified)."""
    original: FeatureLine
    is_present: bool = True
    position_offset: Tuple[int, int] = (0, 0)  # (dx, dy)
    accuracy: float = 1.0  # 1.0 = perfect, 0.0 = very poor
    tremor_strength: float = 0.0  # 0 = none, 3+ = strong
    curvature: float = 0.0  # 0 = straight, 0.5 = curved
    shortening: float = 1.0  # 1.0 = full length, 0.5 = half
    
    def score(self) -> float:
        """Calculate score for this feature (0-3 points)."""
        if not self.is_present:
            return 0.0
        
        # Presence: 1 point if present
        presence_score = 1.0
        
        # Position: binary (0 or 1)
        offset_dist = np.sqrt(self.position_offset[0]**2 + self.position_offset[1]**2)
        position_score = 1.0 if offset_dist < 25 else 0.0
        
        # Accuracy: binary (0 or 1) based on tremor, curvature, shortening
        accuracy_penalty = self.tremor_strength / 5.0 + self.curvature + (1 - self.shortening)
        accuracy_score = 1.0 if accuracy_penalty <= 0.5 else 0.0
        
        return presence_score + position_score + accuracy_score

    def component_flags(self) -> Tuple[int, int, int]:
        """Return (exists, correct_position, correct_accuracy) as 0/1 ints."""
        if not self.is_present:
            return 0, 0, 0
        
        offset_dist = np.sqrt(self.position_offset[0]**2 + self.position_offset[1]**2)
        correct_position = 1 if offset_dist < 25 else 0
        
        accuracy_penalty = self.tremor_strength / 5.0 + self.curvature + (1 - self.shortening)
        correct_accuracy = 1 if accuracy_penalty <= 0.5 else 0
        
        return 1, correct_position, correct_accuracy


# ============================================================================
# FEATURE LOADER
# ============================================================================

class ReferenceFeatureLoader:
    """Loads and provides access to reference image features."""
    
    def __init__(self, db):
        self.db = db
        self.features: List[FeatureLine] = []
        self.image_size = (568, 274)  # width, height
        self._load_features()
    
    def _load_features(self):
        """Load features from reference image in database."""
        from database import ReferenceImage
        
        ref = self.db.query(ReferenceImage).first()
        if not ref or not ref.feature_data:
            raise ValueError("No reference image with features found in database")
        
        data = json.loads(ref.feature_data)
        lines = data.get('lines', [])
        angles = data.get('line_angles', [])
        lengths = data.get('line_lengths', [])
        
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line
            angle = angles[i] if i < len(angles) else self._calc_angle(x1, y1, x2, y2)
            length = lengths[i] if i < len(lengths) else self._calc_length(x1, y1, x2, y2)
            
            self.features.append(FeatureLine(
                index=i,
                x1=int(x1), y1=int(y1),
                x2=int(x2), y2=int(y2),
                length=length,
                angle=angle
            ))
        
        print(f"Loaded {len(self.features)} reference features")
    
    def _calc_angle(self, x1, y1, x2, y2) -> float:
        return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    
    def _calc_length(self, x1, y1, x2, y2) -> float:
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def get_feature_zones(self) -> List[Tuple[int, int, int, int]]:
        """Get bounding boxes around each feature (for avoidance)."""
        zones = []
        margin = 30
        for f in self.features:
            min_x = min(f.x1, f.x2) - margin
            max_x = max(f.x1, f.x2) + margin
            min_y = min(f.y1, f.y2) - margin
            max_y = max(f.y1, f.y2) + margin
            zones.append((min_x, min_y, max_x, max_y))
        return zones


# ============================================================================
# DRAWING UTILITIES
# ============================================================================

class LineDrawer:
    """Draws lines with various modifications (tremor, curves, etc.)."""
    
    def __init__(self, image_size: Tuple[int, int] = (568, 274)):
        self.width, self.height = image_size
    
    def draw_line(
        self,
        img: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        tremor: float = 0.0,
        curvature: float = 0.0,
        thickness: int = 2
    ):
        """Draw a line with optional tremor and curvature."""
        # Generate points along the line
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        num_points = max(10, int(length / 5))
        
        # Reduce curveness slightly for long lines
        if length > 150:
            curvature *= 0.6
        
        if curvature > 0:
            points = self._generate_curved_points(x1, y1, x2, y2, curvature, num_points)
        else:
            points = self._generate_straight_points(x1, y1, x2, y2, num_points)
        
        if tremor > 0:
            points = self._apply_tremor(points, tremor)
        
        # Clip and draw
        self._draw_polyline(img, points, thickness)
    
    def _generate_straight_points(
        self, x1, y1, x2, y2, num_points
    ) -> List[Tuple[int, int]]:
        """Generate points along a straight line."""
        return [
            (int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1)))
            for t in np.linspace(0, 1, num_points)
        ]
    
    def _generate_curved_points(
        self, x1, y1, x2, y2, curvature, num_points
    ) -> List[Tuple[int, int]]:
        """Generate points along a Bezier curve."""
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 1:
            return [(x1, y1), (x2, y2)]
        
        # Perpendicular offset for control point
        perp_x, perp_y = -dy / length, dx / length
        offset = length * curvature * np.random.choice([-1, 1])
        ctrl_x = mid_x + perp_x * offset
        ctrl_y = mid_y + perp_y * offset
        
        # Quadratic Bezier
        points = []
        for t in np.linspace(0, 1, num_points):
            x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
            y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
            points.append((int(x), int(y)))
        
        return points
    
    def _apply_tremor(
        self, points: List[Tuple[int, int]], strength: float
    ) -> List[Tuple[int, int]]:
        """Add hand tremor to points."""
        tremored = []
        for i, (x, y) in enumerate(points):
            # Less tremor at endpoints
            if i == 0 or i == len(points) - 1:
                s = strength * 0.3
            else:
                s = strength
            dx = np.random.normal(0, s)
            dy = np.random.normal(0, s)
            tremored.append((int(x + dx), int(y + dy)))
        return tremored
    
    def _draw_polyline(
        self, img: np.ndarray, points: List[Tuple[int, int]], thickness: int
    ):
        """Draw connected line segments."""
        for i in range(len(points) - 1):
            p1 = (
                max(0, min(self.width - 1, points[i][0])),
                max(0, min(self.height - 1, points[i][1]))
            )
            p2 = (
                max(0, min(self.width - 1, points[i+1][0])),
                max(0, min(self.height - 1, points[i+1][1]))
            )
            cv2.line(img, p1, p2, (0, 0, 0), thickness)


# ============================================================================
# SCORE-BASED GENERATOR
# ============================================================================

class ScoreBasedGenerator:
    """
    Generates synthetic images targeting specific Total_Score values.
    
    Strategy by score range:
    - 0-5: Random lines only, away from features, heavy tremor
    - 6-15: 2-5 features, poor quality
    - 16-25: 5-8 features, moderate quality  
    - 26-35: 8-12 features, mixed quality
    - 36-45: 12-15 features, decent quality
    - 46-55: 15-18 features, good quality
    - 56-63: 18-21 features, excellent quality
    """
    
    def __init__(self, db):
        self.loader = ReferenceFeatureLoader(db)
        self.drawer = LineDrawer(self.loader.image_size)
        self.width, self.height = self.loader.image_size
        self.num_features = len(self.loader.features)
        self.max_score = self.num_features * 3
        self.padding = 15  # pixels
        self.scale_x = (self.width - 2 * self.padding) / self.width
        self.scale_y = (self.height - 2 * self.padding) / self.height
        self.center_x = self.width / 2
        self.center_y = self.height / 2
        self.feature_midpoints = [
            f.midpoint for f in self.loader.features
        ]
        
        # Preferred features (1-based indices from user note; ignore missing)
        # These are weighted higher in selection (e.g., frame/box lines).
        self.preferred_primary = self._normalize_indices([3, 31, 20, 19, 18, 5])
        self.preferred_secondary = self._normalize_indices([4, 2])
        
        print(f"Generator initialized: {self.num_features} features, max score = {self.max_score}")
    
    def generate(
        self,
        target_score: int,
        random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a synthetic image targeting a specific score.
        
        Returns:
            (image_array, metadata_dict)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create white canvas
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Pick patient-level tremor (consistent across all lines in this image)
        tremor_rate = self._sample_patient_tremor(target_score)
        
        # Determine strategy based on target score
        if target_score <= 5:
            drawn_features, extra_lines = self._generate_very_bad(target_score, tremor_rate)
        elif target_score <= 15:
            drawn_features, extra_lines = self._generate_bad(target_score, tremor_rate)
        elif target_score <= 25:
            drawn_features, extra_lines = self._generate_poor(target_score, tremor_rate)
        elif target_score <= 35:
            drawn_features, extra_lines = self._generate_below_average(target_score, tremor_rate)
        elif target_score <= 45:
            drawn_features, extra_lines = self._generate_average(target_score, tremor_rate)
        elif target_score <= 55:
            drawn_features, extra_lines = self._generate_good(target_score, tremor_rate)
        else:
            drawn_features, extra_lines = self._generate_excellent(target_score, tremor_rate)
        
        # Draw the features
        actual_score = 0.0
        correct_exists_num = 0
        correct_pos_num = 0
        correct_accuracy_num = 0
        for df in drawn_features:
            if df.is_present:
                self._draw_feature(img, df)
                actual_score += df.score()
            exists, correct_pos, correct_acc = df.component_flags()
            correct_exists_num += exists
            correct_pos_num += correct_pos
            correct_accuracy_num += correct_acc
        
        # Add extra random lines (1-10, biased to lower counts)
        extra_lines.extend(self._generate_non_matching_lines(tremor_rate))

        # Draw extra random lines (non-matching)
        for line in extra_lines:
            x1, y1, x2, y2, tremor = line
            self.drawer.draw_line(img, x1, y1, x2, y2, tremor=tremor, curvature=0.1)
        
        # Normalize line thickness
        from line_normalizer import normalize_line_thickness
        img = normalize_line_thickness(img, target_thickness=2.0)
        
        actual_score_int = int(actual_score)
        metadata = {
            'target_score': actual_score_int,
            'actual_score': actual_score_int,
            'batch_score': target_score,
            'num_features_drawn': sum(1 for df in drawn_features if df.is_present),
            'num_extra_lines': len(extra_lines),
            'strategy': self._get_strategy_name(target_score),
            'patient_tremor': round(tremor_rate, 2),
            'correct_pos_num': correct_pos_num,
            'correct_accuracy_num': correct_accuracy_num,
            'correct_exists_num': correct_exists_num
        }
        
        return img, metadata
    
    def _get_strategy_name(self, target_score: int) -> str:
        if target_score <= 5:
            return "very_bad"
        elif target_score <= 15:
            return "bad"
        elif target_score <= 25:
            return "poor"
        elif target_score <= 35:
            return "below_average"
        elif target_score <= 45:
            return "average"
        elif target_score <= 55:
            return "good"
        else:
            return "excellent"
    
    def _draw_feature(self, img: np.ndarray, df: DrawnFeature):
        """Draw a single feature with its modifications."""
        f = df.original
        
        # Apply position offset
        dx, dy = df.position_offset
        x1, y1 = self._transform_point(f.x1, f.y1)
        x2, y2 = self._transform_point(f.x2, f.y2)
        x1 += dx
        y1 += dy
        x2 += dx
        y2 += dy
        
        # Apply shortening (from center)
        if df.shortening < 1.0:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            x1 = int(mx + (x1 - mx) * df.shortening)
            y1 = int(my + (y1 - my) * df.shortening)
            x2 = int(mx + (x2 - mx) * df.shortening)
            y2 = int(my + (y2 - my) * df.shortening)
        
        self.drawer.draw_line(
            img, x1, y1, x2, y2,
            tremor=df.tremor_strength,
            curvature=df.curvature
        )

    def _transform_point(self, x: int, y: int) -> Tuple[int, int]:
        """Apply pre-shrink scaling to create padding margins."""
        tx = self.center_x + (x - self.center_x) * self.scale_x
        ty = self.center_y + (y - self.center_y) * self.scale_y
        return int(tx), int(ty)

    def _get_transformed_feature_zones(self) -> List[Tuple[int, int, int, int]]:
        """Get feature zones after padding transform."""
        zones = []
        margin = 30
        for f in self.loader.features:
            x1, y1 = self._transform_point(f.x1, f.y1)
            x2, y2 = self._transform_point(f.x2, f.y2)
            min_x = min(x1, x2) - margin
            max_x = max(x1, x2) + margin
            min_y = min(y1, y2) - margin
            max_y = max(y1, y2) + margin
            zones.append((min_x, min_y, max_x, max_y))
        return zones

    def _normalize_indices(self, indices_1_based: List[int]) -> List[int]:
        """Convert 1-based indices to valid 0-based indices, dropping missing ones."""
        normalized = []
        for idx in indices_1_based:
            zero_idx = idx - 1
            if 0 <= zero_idx < self.num_features:
                normalized.append(zero_idx)
        return normalized

    def _select_feature_indices(
        self,
        num_features: int,
        prefer_nearby: bool = True
    ) -> List[int]:
        """
        Select feature indices with weighted preference:
        - Primary preferred features (frame) get higher weight.
        - Secondary preferred features get medium weight.
        - Subsequent selections prefer features near already chosen ones.
        """
        if num_features <= 0:
            return []
        
        # Base weights
        weights = np.ones(self.num_features, dtype=float)
        for idx in self.preferred_primary:
            weights[idx] *= 3.0
        for idx in self.preferred_secondary:
            weights[idx] *= 2.0
        
        selected = []
        available = set(range(self.num_features))
        
        # First selection by weighted random choice
        first_idx = self._weighted_choice(weights, available)
        selected.append(first_idx)
        available.remove(first_idx)
        
        # Subsequent selections prefer nearby features
        while len(selected) < num_features and available:
            if prefer_nearby:
                proximity_weights = self._compute_proximity_weights(selected, available)
                combined = weights.copy()
                for i in range(self.num_features):
                    if i in available:
                        combined[i] *= proximity_weights.get(i, 1.0)
            else:
                combined = weights
            
            next_idx = self._weighted_choice(combined, available)
            selected.append(next_idx)
            available.remove(next_idx)
        
        return selected
    
    def _weighted_choice(self, weights: np.ndarray, available: set) -> int:
        """Weighted random choice constrained to available indices."""
        available_list = sorted(list(available))
        w = np.array([weights[i] for i in available_list], dtype=float)
        w_sum = w.sum()
        if w_sum <= 0:
            return np.random.choice(available_list)
        w = w / w_sum
        return np.random.choice(available_list, p=w)
    
    def _compute_proximity_weights(self, selected: List[int], available: set) -> Dict[int, float]:
        """Boost weights for features near already-selected ones."""
        proximity = {}
        # Scale controls how strong proximity is (smaller = stronger preference)
        scale = 60.0
        for idx in available:
            mx, my = self.feature_midpoints[idx]
            # Distance to nearest selected feature
            min_dist = min(
                np.sqrt((mx - self.feature_midpoints[s][0])**2 + (my - self.feature_midpoints[s][1])**2)
                for s in selected
            )
            # Convert distance to weight: closer => higher
            proximity[idx] = np.exp(-min_dist / scale)
        return proximity
    
    # ========================================================================
    # SCORE RANGE STRATEGIES
    # ========================================================================
    
    def _generate_very_bad(self, target_score: int, tremor_rate: float) -> Tuple[List[DrawnFeature], List]:
        """
        Score 0-5: Random lines only, placed away from features.
        No actual features, just chaos.
        """
        drawn_features = []
        extra_lines = []
        
        # Generate 3-5 random lines away from feature zones
        num_lines = np.random.randint(3, 6)
        zones = self._get_transformed_feature_zones()
        
        for _ in range(num_lines):
            line = self._generate_random_line_avoiding_zones(zones)
            if line:
                extra_lines.append((*line, tremor_rate))
        
        return drawn_features, extra_lines
    
    def _generate_bad(self, target_score: int, tremor_rate: float) -> Tuple[List[DrawnFeature], List]:
        """
        Score 6-15: 2-5 features, poor position/accuracy.
        """
        # Calculate how many features needed (roughly target_score / 2)
        num_features = max(2, min(5, target_score // 3))
        
        # Select random features
        indices = self._select_feature_indices(num_features)
        
        drawn_features = []
        for idx in indices:
            f = self.loader.features[idx]
            df = DrawnFeature(
                original=f,
                is_present=True,
                position_offset=(
                    np.random.randint(-30, 31),
                    np.random.randint(-30, 31)
                ),
                tremor_strength=tremor_rate,
                curvature=np.random.uniform(0.1, 0.3),
                shortening=np.random.uniform(0.5, 0.8)
            )
            drawn_features.append(df)
        
        # Maybe add 1-2 random extra lines
        extra_lines = []
        if np.random.random() < 0.5:
            zones = self._get_transformed_feature_zones()
            line = self._generate_random_line_avoiding_zones(zones)
            if line:
                extra_lines.append((*line, tremor_rate))
        
        return drawn_features, extra_lines
    
    def _generate_poor(self, target_score: int, tremor_rate: float) -> Tuple[List[DrawnFeature], List]:
        """
        Score 16-25: 5-8 features, moderate quality.
        """
        num_features = max(5, min(8, target_score // 3))
        
        indices = self._select_feature_indices(num_features)
        
        drawn_features = []
        for idx in indices:
            f = self.loader.features[idx]
            df = DrawnFeature(
                original=f,
                is_present=True,
                position_offset=(
                    np.random.randint(-20, 21),
                    np.random.randint(-20, 21)
                ),
                tremor_strength=tremor_rate,
                curvature=np.random.uniform(0.05, 0.2),
                shortening=np.random.uniform(0.6, 0.9)
            )
            drawn_features.append(df)
        
        return drawn_features, []
    
    def _generate_below_average(self, target_score: int, tremor_rate: float) -> Tuple[List[DrawnFeature], List]:
        """
        Score 26-35: 8-12 features, mixed quality.
        """
        num_features = max(8, min(12, target_score // 3))
        
        indices = self._select_feature_indices(num_features)
        
        drawn_features = []
        for idx in indices:
            f = self.loader.features[idx]
            # Mix of good and poor features
            if np.random.random() < 0.4:
                # Good feature
                df = DrawnFeature(
                    original=f,
                    is_present=True,
                    position_offset=(
                        np.random.randint(-5, 6),
                        np.random.randint(-5, 6)
                    ),
                    tremor_strength=tremor_rate,
                    curvature=np.random.uniform(0.0, 0.1),
                    shortening=np.random.uniform(0.85, 1.0)
                )
            else:
                # Poor feature
                df = DrawnFeature(
                    original=f,
                    is_present=True,
                    position_offset=(
                        np.random.randint(-15, 16),
                        np.random.randint(-15, 16)
                    ),
                    tremor_strength=tremor_rate,
                    curvature=np.random.uniform(0.05, 0.15),
                    shortening=np.random.uniform(0.7, 0.9)
                )
            drawn_features.append(df)
        
        return drawn_features, []
    
    def _generate_average(self, target_score: int, tremor_rate: float) -> Tuple[List[DrawnFeature], List]:
        """
        Score 36-45: 12-15 features, decent quality.
        """
        num_features = max(12, min(15, target_score // 3))
        
        indices = self._select_feature_indices(num_features)
        
        drawn_features = []
        for idx in indices:
            f = self.loader.features[idx]
            # Mostly decent with some variation
            df = DrawnFeature(
                original=f,
                is_present=True,
                position_offset=(
                    np.random.randint(-10, 11),
                    np.random.randint(-10, 11)
                ),
                tremor_strength=tremor_rate,
                curvature=np.random.uniform(0.0, 0.1),
                shortening=np.random.uniform(0.8, 1.0)
            )
            drawn_features.append(df)
        
        return drawn_features, []
    
    def _generate_good(self, target_score: int, tremor_rate: float) -> Tuple[List[DrawnFeature], List]:
        """
        Score 46-55: 15-18 features, good quality.
        """
        num_features = max(15, min(18, target_score // 3))
        
        indices = self._select_feature_indices(num_features)
        
        drawn_features = []
        for idx in indices:
            f = self.loader.features[idx]
            df = DrawnFeature(
                original=f,
                is_present=True,
                position_offset=(
                    np.random.randint(-5, 6),
                    np.random.randint(-5, 6)
                ),
                tremor_strength=tremor_rate,
                curvature=np.random.uniform(0.0, 0.05),
                shortening=np.random.uniform(0.9, 1.0)
            )
            drawn_features.append(df)
        
        return drawn_features, []
    
    def _generate_excellent(self, target_score: int, tremor_rate: float) -> Tuple[List[DrawnFeature], List]:
        """
        Score 56-63: 18-21 features, excellent quality.
        """
        num_features = max(18, min(self.num_features, target_score // 3))
        
        indices = self._select_feature_indices(num_features)
        
        drawn_features = []
        for idx in indices:
            f = self.loader.features[idx]
            df = DrawnFeature(
                original=f,
                is_present=True,
                position_offset=(
                    np.random.randint(-3, 4),
                    np.random.randint(-3, 4)
                ),
                tremor_strength=tremor_rate,
                curvature=np.random.uniform(0.0, 0.02),
                shortening=np.random.uniform(0.95, 1.0)
            )
            drawn_features.append(df)
        
        return drawn_features, []
    
    def _generate_random_line_avoiding_zones(
        self, zones: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Generate a random line that avoids feature zones."""
        for _ in range(50):  # Max attempts
            # Random start point
            x1 = np.random.randint(self.padding, self.width - self.padding)
            y1 = np.random.randint(self.padding, self.height - self.padding)
            
            # Check if in any zone
            in_zone = any(
                z[0] <= x1 <= z[2] and z[1] <= y1 <= z[3]
                for z in zones
            )
            if in_zone:
                continue
            
            # Random direction and length
            length = np.random.randint(40, 150)
            angle = np.random.uniform(0, 2 * np.pi)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            
            # Clip to bounds
            x2 = max(self.padding, min(self.width - self.padding, x2))
            y2 = max(self.padding, min(self.height - self.padding, y2))
            
            # Check end point
            in_zone = any(
                z[0] <= x2 <= z[2] and z[1] <= y2 <= z[3]
                for z in zones
            )
            if not in_zone:
                return (x1, y1, x2, y2)
        
        # Fallback: just return something
        return (
            np.random.randint(self.padding, self.padding + 80),
            np.random.randint(self.padding, self.padding + 80),
            np.random.randint(self.padding + 80, self.padding + 180),
            np.random.randint(self.padding + 80, self.padding + 180)
        )

    def _generate_non_matching_lines(self, tremor_rate: float) -> List[Tuple[int, int, int, int, float]]:
        """Generate 1-10 random lines that avoid feature zones and matching angles."""
        count = self._sample_extra_lines_count()
        zones = self._get_transformed_feature_zones()
        feature_angles = [f.angle for f in self.loader.features]
        lines = []
        attempts = 0
        max_attempts = 200

        while len(lines) < count and attempts < max_attempts:
            attempts += 1
            candidate = self._generate_random_line_avoiding_zones(zones)
            if not candidate:
                continue
            x1, y1, x2, y2 = candidate
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Avoid matching feature angles (within 10 degrees)
            if any(abs(angle - fa) < 10 for fa in feature_angles):
                continue

            lines.append((x1, y1, x2, y2, tremor_rate))

        return lines

    def _sample_patient_tremor(self, target_score: int) -> float:
        """
        Sample a patient-level tremor rate.
        Use consistent tremor across all scores (match score ~40 range).
        """
        return float(np.random.uniform(0.4, 1.0))

    def _sample_extra_lines_count(self) -> int:
        """Sample number of extra lines with higher probability for small counts."""
        # Geometric-like distribution capped at 10
        probs = np.array([1.0 / (i + 1) for i in range(10)], dtype=float)
        probs = probs / probs.sum()
        return int(np.random.choice(np.arange(1, 11), p=probs))


# ============================================================================
# DATA PIPELINE INTEGRATION
# ============================================================================

def generate_synthetic_score_images_for_training(
    db,
    n_samples: int = 100,
    random_seed: int = 42,
    image_size: Tuple[int, int] = (568, 274)
) -> List[Dict]:
    """
    Generate synthetic images with score-based distribution for training pipeline.
    
    Distribution:
    - Score 0: 40% of total
    - Score 1-10: 15% of total (evenly distributed across 5, 10)
    - Score 11-20: 15% of total (evenly distributed across 15, 20)
    - Score 21-30: 15% of total (evenly distributed across 25, 30)
    - Score 31-40: 15% of total (evenly distributed across 35, 40)
    
    Args:
        db: Database session
        n_samples: Total number of images to generate
        random_seed: Base random seed for reproducibility
        image_size: Output image size (width, height)
    
    Returns:
        List of dicts with 'image_data' (bytes), 'target_score', and metadata
    """
    from utils.logger import get_logger
    logger = get_logger(__name__)
    
    logger.info(f"Generating {n_samples} score-based synthetic images...")
    
    # Calculate distribution
    n_score_0 = int(n_samples * 0.40)  # 40% for score 0
    n_per_range = int(n_samples * 0.15)  # 15% per range
    
    # Ensure we hit exactly n_samples
    total_allocated = n_score_0 + 4 * n_per_range
    remainder = n_samples - total_allocated
    n_score_0 += remainder  # Add remainder to score 0
    
    # Scores and their sample counts
    # Range 1-10: use scores 5, 10
    # Range 11-20: use scores 15, 20
    # Range 21-30: use scores 25, 30
    # Range 31-40: use scores 35, 40
    score_distribution = [
        (0, n_score_0),
        (5, n_per_range // 2),
        (10, n_per_range - n_per_range // 2),
        (15, n_per_range // 2),
        (20, n_per_range - n_per_range // 2),
        (25, n_per_range // 2),
        (30, n_per_range - n_per_range // 2),
        (35, n_per_range // 2),
        (40, n_per_range - n_per_range // 2),
    ]
    
    logger.info(f"Distribution: {[(s, c) for s, c in score_distribution]}")
    
    # Initialize generator
    generator = ScoreBasedGenerator(db)
    
    synthetic_images = []
    idx = 0
    
    for target_score, count in score_distribution:
        if count <= 0:
            continue
            
        logger.info(f"  Score {target_score}: generating {count} images...")
        
        for sample_idx in range(count):
            seed = random_seed + target_score * 1000 + sample_idx
            
            img_array, metadata = generator.generate(target_score, random_seed=seed)
            
            # Convert to PNG bytes
            img_pil = Image.fromarray(img_array)
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            synthetic_images.append({
                'image_data': img_bytes,
                'target_score': int(metadata['actual_score']),
                'batch_score': target_score,
                'metadata': metadata
            })
            
            idx += 1
    
    logger.info(f"Generated {len(synthetic_images)} synthetic score-based images")
    
    # Log distribution summary
    score_counts = {}
    for img in synthetic_images:
        s = img['target_score']
        score_counts[s] = score_counts.get(s, 0) + 1
    
    logger.info(f"Actual score distribution: {sorted(score_counts.items())}")
    
    return synthetic_images


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def clean_output_directory(output_dir: Path):
    """Remove all files from output directory."""
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.is_file():
                f.unlink()
        print(f"Cleaned output directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic images with specific Total_Score targets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/app/data/tmp/synthetic_images',
        help='Output directory'
    )
    parser.add_argument(
        '--scores',
        type=str,
        default='0,5,10,15,20,25,30,35,40,45,50,55,60',
        help='Comma-separated list of target scores'
    )
    parser.add_argument(
        '--samples-per-score',
        type=int,
        default=3,
        help='Number of samples per target score'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    scores = [int(s.strip()) for s in args.scores.split(',')]
    
    # Clean output directory
    clean_output_directory(output_dir)
    
    # Initialize generator
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        generator = ScoreBasedGenerator(db)
        
        print(f"\nGenerating images for scores: {scores}")
        print(f"Samples per score: {args.samples_per_score}")
        print()
        
        idx = 0
        manifest = {
            'generation_params': {
                'scores': scores,
                'samples_per_score': args.samples_per_score,
                'random_seed': args.random_seed,
                'num_features': generator.num_features,
                'max_score': generator.max_score
            },
            'images': []
        }
        
        for target_score in scores:
            print(f"Score {target_score:3d}: ", end='', flush=True)
            
            for sample_idx in range(args.samples_per_score):
                seed = args.random_seed + target_score * 100 + sample_idx
                
                img, metadata = generator.generate(target_score, random_seed=seed)
                
                # Save image
                filename = f"synth_score{target_score:02d}_{sample_idx:02d}.png"
                img_path = output_dir / filename
                
                img_pil = Image.fromarray(img)
                img_pil.save(str(img_path))
                
                # Save per-image JSON
                json_path = output_dir / f"synth_score{target_score:02d}_{sample_idx:02d}.json"
                json_data = {
                    'image_id': f'synth_score{target_score}_{sample_idx}',
                    'patient_id': f'SYNTHETIC_SCORE_{target_score}',
                    'target_feature': 'Total_Score',
                    'target_score': metadata['actual_score'],
                    'target_value': metadata['actual_score'],
                    'target_value_original': metadata['actual_score'],
                    'batch_score': target_score,
                    'augmentation': None,
                    'synthetic': True,
                    'metadata': metadata
                }
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                manifest['images'].append({
                    'filename': filename,
                    'target_score': target_score,
                    'actual_score': metadata['actual_score'],
                    'strategy': metadata['strategy']
                })
                
                print('.', end='', flush=True)
                idx += 1
            
            print(f" ({args.samples_per_score} images)")
        
        # Save manifest
        manifest_path = output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ Generated {idx} images in {output_dir}")
        print(f"   Manifest saved to {manifest_path}")
        
    finally:
        db.close()


if __name__ == '__main__':
    main()
