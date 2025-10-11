#!/usr/bin/env python3
"""
Verify that optimized parameters are working.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage
from image_processing import LineComparator, LineDetector, normalize_image

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 60)
print("✅ VERIFY OPTIMIZED PARAMETERS")
print("=" * 60)

# Test 1: Check LineComparator defaults
comparator = LineComparator()
print(f"\n📊 LineComparator Defaults:")
print(f"   Position Tolerance: {comparator.position_tolerance}px")
print(f"   Angle Tolerance: {comparator.angle_tolerance}°")
print(f"   Length Tolerance: {comparator.length_tolerance*100:.0f}%")

expected = (100.0, 45.0, 0.7)
actual = (comparator.position_tolerance, comparator.angle_tolerance, comparator.length_tolerance)

if actual == expected:
    print(f"   ✅ CORRECT!")
else:
    print(f"   ❌ WRONG! Expected {expected}, got {actual}")

# Test 2: Quick comparison with test image
ref = db.query(ReferenceImage).first()
test_img = db.query(TestImage).first()

if ref and test_img:
    detector = LineDetector()
    ref_data = detector.features_from_json(ref.feature_data)
    ref_lines = ref_data['lines']
    
    nparr = np.frombuffer(test_img.image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    normalized = normalize_image(image)
    test_features = detector.extract_features(normalized)
    test_lines = test_features['lines']
    
    # Compare with NEW defaults
    comparison = comparator.compare_lines(test_lines, ref_lines)
    
    print(f"\n📗 Quick Test ({test_img.test_name}):")
    print(f"   Correct: {comparison['correct_lines']}")
    print(f"   Missing: {comparison['missing_lines']}")
    print(f"   Extra: {comparison['extra_lines']}")
    print(f"   Score: {comparison['similarity_score']:.2%}")
    
    # Calculate accuracy
    total_ref = len(ref_lines)
    effective_correct = max(0, comparison['correct_lines'] - comparison['extra_lines'])
    accuracy = effective_correct / total_ref if total_ref > 0 else 0.0
    
    print(f"   🎯 Accuracy: {accuracy*100:.1f}%")
    
    if accuracy > 0:
        print(f"   ✅ IMPROVEMENT DETECTED!")
    else:
        print(f"   ⚠️  Still 0% - needs investigation")

print("\n" + "=" * 60)
print("✅ Verification Complete!")
print("=" * 60)
