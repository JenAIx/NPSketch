#!/usr/bin/env python3
"""
Analyze individual test image performance.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage
from services.evaluation_service import EvaluationService
from image_processing import LineDetector, LineComparator

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 80)
print("📊 DETAILED TEST PERFORMANCE ANALYSIS")
print("=" * 80)

# Load reference
ref = db.query(ReferenceImage).first()
ref_data = LineDetector().features_from_json(ref.feature_data)
total_ref_lines = len(ref_data['lines'])

print(f"\n📘 Reference: {total_ref_lines} lines detected")

# Get test images
test_images = db.query(TestImage).order_by(TestImage.id).all()

print(f"\n📊 Analyzing {len(test_images)} test images with CURRENT parameters:")
print("   Position: 100px, Angle: 45°, Length: 70%")

for idx, test_img in enumerate(test_images, 1):
    print(f"\n{'─' * 80}")
    print(f"[{idx}] {test_img.test_name} (ID: {test_img.id})")
    print(f"{'─' * 80}")
    print(f"Expected: C={test_img.expected_correct}, M={test_img.expected_missing}, E={test_img.expected_extra}")
    
    # Load and evaluate
    nparr = np.frombuffer(test_img.image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Use evaluation service
    eval_service = EvaluationService(
        db,
        use_registration=True,
        registration_motion="similarity",
        max_rotation_degrees=30.0
    )
    
    evaluation = eval_service.evaluate_test_image(image, ref.id, f"analysis_{test_img.id}")
    
    # Calculate accuracy
    effective_correct = max(0, evaluation.correct_lines - evaluation.extra_lines)
    accuracy = effective_correct / total_ref_lines if total_ref_lines > 0 else 0.0
    
    print(f"\nActual Results:")
    print(f"   Correct:  {evaluation.correct_lines}/{total_ref_lines}")
    print(f"   Missing:  {evaluation.missing_lines}/{total_ref_lines}")
    print(f"   Extra:    {evaluation.extra_lines}")
    print(f"   Score:    {evaluation.similarity_score:.2%}")
    print(f"   Accuracy: {accuracy*100:.1f}%")
    
    # Compare with expected
    correct_diff = evaluation.correct_lines - test_img.expected_correct
    missing_diff = evaluation.missing_lines - test_img.expected_missing
    extra_diff = evaluation.extra_lines - test_img.expected_extra
    
    print(f"\nDifference from Expected:")
    print(f"   Correct:  {correct_diff:+d}")
    print(f"   Missing:  {missing_diff:+d}")
    print(f"   Extra:    {extra_diff:+d}")
    
    # Performance rating
    if accuracy >= 0.8:
        rating = "🌟 EXCELLENT"
    elif accuracy >= 0.5:
        rating = "✅ GOOD"
    elif accuracy >= 0.3:
        rating = "⚠️  NEEDS IMPROVEMENT"
    else:
        rating = "❌ POOR"
    
    print(f"\n{rating}")

print("\n" + "=" * 80)
print("💡 RECOMMENDATIONS:")
print("=" * 80)

# Find worst performing
worst = min(
    [(t, max(0, db.query(TestImage).filter(TestImage.id == t.id).first().expected_correct - 0) / total_ref_lines) 
     for t in test_images],
    key=lambda x: x[1]
)

print(f"\n🎯 Focus on: {worst[0].test_name} (lowest expected accuracy)")
print(f"   This image needs special attention for parameter tuning.")

print("\n" + "=" * 80)
