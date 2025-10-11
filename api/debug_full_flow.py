#!/usr/bin/env python3
"""
Debug full evaluation flow with detailed error tracking.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage
from services.evaluation_service import EvaluationService

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 60)
print("🔍 DEBUG FULL EVALUATION FLOW")
print("=" * 60)

# Load images
ref = db.query(ReferenceImage).first()
test_img = db.query(TestImage).filter(TestImage.test_name == "Supper").first()

nparr = np.frombuffer(test_img.image_data, np.uint8)
test_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

print(f"📗 Test Image: {test_img.test_name}")
print(f"   Expected: C={test_img.expected_correct}, M={test_img.expected_missing}, E={test_img.expected_extra}")

# Create evaluation service with registration enabled
eval_service = EvaluationService(
    db,
    use_registration=True,
    registration_motion="similarity",
    max_rotation_degrees=30.0
)

print(f"\n🔧 EvaluationService config:")
print(f"   use_registration: {eval_service.use_registration}")
print(f"   registration_motion: {eval_service.registration_motion}")
print(f"   max_rotation_degrees: {eval_service.max_rotation_degrees}")

# Run evaluation
print(f"\n🔄 Running evaluation...")
try:
    evaluation = eval_service.evaluate_test_image(test_image, ref.id, "test_debug")
    
    print(f"\n📊 Results:")
    print(f"   Correct: {evaluation.correct_lines}")
    print(f"   Missing: {evaluation.missing_lines}")
    print(f"   Extra: {evaluation.extra_lines}")
    print(f"   Score: {evaluation.similarity_score:.2%}")
    
    # Check visualization
    print(f"\n💾 Visualization: {evaluation.visualization_path}")
    
except Exception as e:
    print(f"\n❌ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
