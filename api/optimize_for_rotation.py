#!/usr/bin/env python3
"""
Special optimization for rotated image (test_4).
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage
from services.evaluation_service import EvaluationService
from image_processing import LineComparator, LineDetector

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 80)
print("🔄 OPTIMIZATION FOR ROTATED IMAGE (test_4)")
print("=" * 80)

# Get test_4
test_img = db.query(TestImage).filter(TestImage.id == 4).first()
ref = db.query(ReferenceImage).first()

if not test_img:
    print("❌ test_4 not found!")
    sys.exit(1)

print(f"\n📗 Target: {test_img.test_name}")
print(f"   Expected: C={test_img.expected_correct}, M={test_img.expected_missing}, E={test_img.expected_extra}")

# Load image
nparr = np.frombuffer(test_img.image_data, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Test different parameter combinations focused on rotation
configs = [
    # (max_rot, pos_tol, ang_tol, len_tol, description)
    (30, 100, 45, 0.7, "Current defaults"),
    (45, 100, 45, 0.7, "Higher rotation"),
    (60, 100, 45, 0.7, "Very high rotation"),
    (45, 120, 50, 0.8, "Higher rotation + tolerances"),
    (60, 120, 50, 0.8, "Max rotation + tolerances"),
    (45, 150, 60, 0.9, "Extreme tolerances"),
]

print(f"\n🔬 Testing {len(configs)} configurations...")
print("─" * 80)

results = []

detector = LineDetector()
ref_data = detector.features_from_json(ref.feature_data)
total_ref = len(ref_data['lines'])

for max_rot, pos_tol, ang_tol, len_tol, desc in configs:
    # Create evaluation service
    eval_service = EvaluationService(
        db,
        use_registration=True,
        registration_motion="similarity",
        max_rotation_degrees=max_rot
    )
    
    # Custom comparator
    eval_service.comparator = LineComparator(
        position_tolerance=pos_tol,
        angle_tolerance=ang_tol,
        length_tolerance=len_tol
    )
    
    # Evaluate
    evaluation = eval_service.evaluate_test_image(image, ref.id, f"opt_test4")
    
    # Calculate accuracy
    effective_correct = max(0, evaluation.correct_lines - evaluation.extra_lines)
    accuracy = effective_correct / total_ref if total_ref > 0 else 0.0
    
    results.append({
        'config': (max_rot, pos_tol, ang_tol, len_tol),
        'desc': desc,
        'correct': evaluation.correct_lines,
        'missing': evaluation.missing_lines,
        'extra': evaluation.extra_lines,
        'accuracy': accuracy
    })
    
    print(f"{desc:30s} | MaxRot:{max_rot:2.0f}° Pos:{pos_tol:3.0f}px Ang:{ang_tol:2.0f}° Len:{len_tol*100:2.0f}%")
    print(f"{'':30s} | C:{evaluation.correct_lines}/8 M:{evaluation.missing_lines} E:{evaluation.extra_lines} → {accuracy*100:5.1f}%")
    print("─" * 80)

# Find best
best = max(results, key=lambda x: x['accuracy'])

print("\n🏆 BEST CONFIGURATION FOR test_4:")
print("=" * 80)
max_rot, pos_tol, ang_tol, len_tol = best['config']
print(f"Description:       {best['desc']}")
print(f"Max Rotation:      {max_rot}°")
print(f"Position Tolerance: {pos_tol}px")
print(f"Angle Tolerance:    {ang_tol}°")
print(f"Length Tolerance:   {len_tol*100:.0f}%")
print(f"\nResults:")
print(f"   Correct:  {best['correct']}/8")
print(f"   Missing:  {best['missing']}/8")
print(f"   Extra:    {best['extra']}")
print(f"   Accuracy: {best['accuracy']*100:.1f}%")

improvement = best['accuracy'] - results[0]['accuracy']
print(f"\n📈 Improvement: {improvement*100:+.1f}% (from {results[0]['accuracy']*100:.1f}% to {best['accuracy']*100:.1f}%)")

print("\n💡 RECOMMENDATION:")
if best['config'] != configs[0][:4]:
    print(f"   Update defaults to:")
    print(f"   • Max Rotation: {max_rot}°")
    print(f"   • Position: {pos_tol}px")
    print(f"   • Angle: {ang_tol}°")
    print(f"   • Length: {len_tol*100:.0f}%")
else:
    print(f"   Current defaults are already optimal for this image.")

print("=" * 80)
