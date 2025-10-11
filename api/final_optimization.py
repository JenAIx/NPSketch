#!/usr/bin/env python3
"""
Find the sweet spot: Best average across all images.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage
from image_processing import normalize_image, LineComparator
from image_processing.line_detector import LineDetector

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 80)
print("🎯 FINAL OPTIMIZATION: Finding Sweet Spot")
print("=" * 80)

ref = db.query(ReferenceImage).first()
detector_ref = LineDetector()
ref_data = detector_ref.features_from_json(ref.feature_data)
ref_lines = ref_data['lines']

test_images = db.query(TestImage).all()

# Test combinations of detection + matching
configs = [
    # (thr, minlen, gap, pos_tol, ang_tol, len_tol, desc)
    (100, 80, 25, 100, 45, 0.7, "Current (strict det + moderate tol)"),
    (60, 60, 35, 100, 45, 0.7, "Moderate det + moderate tol"),
    (50, 50, 40, 120, 50, 0.8, "Relaxed det + higher tol"),
    (60, 60, 35, 120, 50, 0.8, "Moderate det + higher tol ⭐"),
    (70, 70, 30, 120, 50, 0.8, "Balanced det + higher tol"),
    (60, 60, 35, 150, 60, 0.85, "Moderate det + very high tol"),
]

print(f"\n🔬 Testing {len(configs)} complete configurations...")
print("─" * 80)

results = []

for thr, minlen, gap, pos_tol, ang_tol, len_tol, desc in configs:
    detector = LineDetector(
        threshold=thr,
        min_line_length=minlen,
        max_line_gap=gap
    )
    
    comparator = LineComparator(
        position_tolerance=pos_tol,
        angle_tolerance=ang_tol,
        length_tolerance=len_tol
    )
    
    total_accuracy = 0
    test_results = []
    
    for test_img in test_images:
        nparr = np.frombuffer(test_img.image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        normalized = normalize_image(image)
        
        features = detector.extract_features(normalized)
        test_lines = features['lines']
        
        comparison = comparator.compare_lines(test_lines, ref_lines)
        
        effective_correct = max(0, comparison['correct_lines'] - comparison['extra_lines'])
        accuracy = effective_correct / len(ref_lines) if len(ref_lines) > 0 else 0.0
        
        test_results.append({
            'name': test_img.test_name.split()[0][:10],  # Shorten name
            'accuracy': accuracy
        })
        
        total_accuracy += accuracy
    
    avg_accuracy = total_accuracy / len(test_images)
    
    results.append({
        'config': (thr, minlen, gap, pos_tol, ang_tol, len_tol),
        'desc': desc,
        'avg_accuracy': avg_accuracy,
        'details': test_results
    })
    
    acc_str = ' '.join([f"{r['accuracy']*100:4.0f}%" for r in test_results])
    print(f"{avg_accuracy*100:5.1f}% | {desc:45s} [{acc_str}]")

print("─" * 80)

# Find best
best = max(results, key=lambda x: x['avg_accuracy'])

print("\n🏆 OPTIMAL CONFIGURATION:")
print("=" * 80)
thr, minlen, gap, pos_tol, ang_tol, len_tol = best['config']
print(f"Description: {best['desc']}")
print(f"\nLine Detection:")
print(f"   • Threshold:        100 → {thr}")
print(f"   • Min Line Length:  80 → {minlen}")
print(f"   • Max Line Gap:     25 → {gap}")
print(f"\nLine Matching:")
print(f"   • Position:         100px → {pos_tol}px")
print(f"   • Angle:            45° → {ang_tol}°")
print(f"   • Length:           70% → {len_tol*100:.0f}%")
print(f"\nAverage Accuracy: {best['avg_accuracy']*100:.1f}%")

print(f"\n📊 Individual Performance:")
for detail in best['details']:
    print(f"   {detail['name']:12s} {detail['accuracy']*100:5.1f}%")

print("\n" + "=" * 80)
print("💡 RECOMMENDATION: Apply this configuration!")
print("=" * 80)
