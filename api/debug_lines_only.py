#!/usr/bin/env python3
"""
Simplest possible test - just compare lines without registration.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage
from image_processing import LineDetector, normalize_image, LineComparator, load_image_from_bytes

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 60)
print("🔍 SIMPLE LINE COMPARISON (NO REGISTRATION)")
print("=" * 60)

# Load reference
ref = db.query(ReferenceImage).first()
ref_image = load_image_from_bytes(ref.processed_image_data)
detector = LineDetector()
ref_data = detector.features_from_json(ref.feature_data)
ref_lines = ref_data['lines']

print(f"\n📘 Reference: {len(ref_lines)} lines")
for i, line in enumerate(ref_lines):
    print(f"   [{i}] {line}")

# Load test image
test_img = db.query(TestImage).filter(TestImage.test_name == "Supper").first()
nparr = np.frombuffer(test_img.image_data, np.uint8)
test_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
test_normalized = normalize_image(test_image)

# Extract lines
test_features = detector.extract_features(test_normalized)
test_lines = test_features['lines']

print(f"\n📗 Test: {len(test_lines)} lines")
for i, line in enumerate(test_lines):
    print(f"   [{i}] {line}")

# Compare with different tolerance settings
tolerances = [
    (10, 10, 0.2),
    (20, 15, 0.3),
    (30, 20, 0.4),
    (50, 30, 0.5),
    (100, 45, 0.7),
]

print(f"\n📊 Testing different tolerances:")
for pos_tol, ang_tol, len_tol in tolerances:
    comparator = LineComparator(
        position_tolerance=pos_tol,
        angle_tolerance=ang_tol,
        length_tolerance=len_tol
    )
    
    comparison = comparator.compare_lines(test_lines, ref_lines)
    
    print(f"\n   Pos={pos_tol}px, Ang={ang_tol}°, Len={int(len_tol*100)}%:")
    print(f"      Correct: {comparison['correct_lines']}")
    print(f"      Missing: {comparison['missing_lines']}")
    print(f"      Extra: {comparison['extra_lines']}")
    print(f"      Score: {comparison['similarity_score']:.2%}")

print("=" * 60)
