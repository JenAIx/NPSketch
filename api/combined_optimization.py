#!/usr/bin/env python3
"""
Combined optimization: Relaxed line detection + very high tolerances.
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
print("🎯 COMBINED OPTIMIZATION: Relaxed Detection + High Tolerances")
print("=" * 80)

ref = db.query(ReferenceImage).first()
detector_ref = LineDetector()
ref_data = detector_ref.features_from_json(ref.feature_data)
ref_lines = ref_data['lines']

test_images = db.query(TestImage).all()

# Use very relaxed line detection (finds 7 lines in test_4)
best_line_detection = (40, 40, 50)  # From earlier: finds 7 lines in test_4

print(f"\nLine Detection: Thr={best_line_detection[0]}, MinLen={best_line_detection[1]}, MaxGap={best_line_detection[2]}")

detector = LineDetector(
    threshold=best_line_detection[0],
    min_line_length=best_line_detection[1],
    max_line_gap=best_line_detection[2]
)

# Test with EXTREME tolerances for matching
tolerance_configs = [
    (100, 45, 0.7, "Current"),
    (120, 50, 0.8, "Higher"),
    (150, 60, 0.9, "Very high"),
    (180, 70, 0.95, "Extreme"),
]

print(f"\n🔬 Testing {len(tolerance_configs)} tolerance levels...")
print("─" * 80)

results = []

for pos_tol, ang_tol, len_tol, desc in tolerance_configs:
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
            'name': test_img.test_name,
            'lines': len(test_lines),
            'correct': comparison['correct_lines'],
            'missing': comparison['missing_lines'],
            'extra': comparison['extra_lines'],
            'accuracy': accuracy
        })
        
        total_accuracy += accuracy
    
    avg_accuracy = total_accuracy / len(test_images)
    
    results.append({
        'config': (pos_tol, ang_tol, len_tol),
        'desc': desc,
        'avg_accuracy': avg_accuracy,
        'details': test_results
    })
    
    print(f"{desc:15s} Pos:{pos_tol:3.0f}px Ang:{ang_tol:2.0f}° Len:{len_tol*100:2.0f}% → Avg:{avg_accuracy*100:5.1f}%")
    for r in test_results:
        print(f"              {r['name']:25s} {r['lines']:2d}L → C:{r['correct']} M:{r['missing']} E:{r['extra']} = {r['accuracy']*100:5.1f}%")
    print("─" * 80)

# Find best
best = max(results, key=lambda x: x['avg_accuracy'])

print("\n🏆 BEST COMBINED CONFIGURATION:")
print("=" * 80)
print(f"Line Detection:")
print(f"   • Threshold:        {best_line_detection[0]}")
print(f"   • Min Line Length:  {best_line_detection[1]}")
print(f"   • Max Line Gap:     {best_line_detection[2]}")
print(f"\nLine Matching:")
pos_tol, ang_tol, len_tol = best['config']
print(f"   • Position Tolerance: {pos_tol}px")
print(f"   • Angle Tolerance:    {ang_tol}°")
print(f"   • Length Tolerance:   {len_tol*100:.0f}%")
print(f"\nAverage Accuracy: {best['avg_accuracy']*100:.1f}%")

print(f"\n📊 Individual Results:")
for detail in best['details']:
    print(f"   {detail['name']:25s} | {detail['accuracy']*100:5.1f}%")

print("\n=" * 80)
