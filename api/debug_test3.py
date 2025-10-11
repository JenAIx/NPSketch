#!/usr/bin/env python3
"""
Debug matching logic.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage
from services.evaluation_service import EvaluationService
from image_processing import LineDetector, normalize_image

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 60)
print("🔍 DEBUG: Why no matches?")
print("=" * 60)

# Load reference
ref = db.query(ReferenceImage).first()
ref_data = LineDetector().features_from_json(ref.feature_data)
print(f"\n📘 Reference Lines: {len(ref_data['lines'])}")
for i, line in enumerate(ref_data['lines']):
    print(f"   [{i}] {line}")

# Load test image
test_img = db.query(TestImage).filter(TestImage.test_name == "Supper").first()
print(f"\n📗 Test Image: {test_img.test_name}")
print(f"   Expected: C={test_img.expected_correct}, M={test_img.expected_missing}, E={test_img.expected_extra}")

# Load and process
nparr = np.frombuffer(test_img.image_data, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
normalized = normalize_image(image)

# Create evaluation service
eval_service = EvaluationService(
    db,
    use_registration=True,
    registration_motion="similarity",
    max_rotation_degrees=30.0
)

# Run evaluation
print(f"\n🔄 Running evaluation...")
evaluation = eval_service.evaluate_test_image(image, ref.id, "test_supper")

print(f"\n📊 Results:")
print(f"   Correct: {evaluation.correct_lines}")
print(f"   Missing: {evaluation.missing_lines}")
print(f"   Extra: {evaluation.extra_lines}")
print(f"   Score: {evaluation.similarity_score:.2%}")

# Calculate accuracy
total_ref_lines = len(ref_data['lines'])
effective_correct = max(0, evaluation.correct_lines - evaluation.extra_lines)
accuracy = effective_correct / total_ref_lines if total_ref_lines > 0 else 0.0
print(f"\n🎯 Accuracy: {accuracy*100:.1f}%")
print(f"   (effective_correct={effective_correct} / total_ref={total_ref_lines})")

print("=" * 60)
