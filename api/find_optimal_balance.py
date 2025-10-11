#!/usr/bin/env python3
"""
Find optimal balance for all test images.
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
print("⚖️  FINDING OPTIMAL BALANCE FOR ALL IMAGES")
print("=" * 80)

ref = db.query(ReferenceImage).first()
detector_ref = LineDetector()
ref_data = detector_ref.features_from_json(ref.feature_data)
ref_lines = ref_data['lines']

test_images = db.query(TestImage).all()

# Test configurations looking for balance
configs = [
    (60, 60, 35),  # Moderate
    (50, 50, 40),  # Relaxed
    (55, 55, 37),  # Mid-point
    (45, 45, 45),  # Balanced
    (50, 55, 40),  # Custom 1
    (55, 50, 40),  # Custom 2
]

print(f"\n🔬 Testing {len(configs)} balanced configurations...")
print("─" * 80)

results = []

for threshold, min_len, max_gap in configs:
    detector = LineDetector(
        threshold=threshold,
        min_line_length=min_len,
        max_line_gap=max_gap
    )
    
    comparator = LineComparator(
        position_tolerance=100.0,
        angle_tolerance=45.0,
        length_tolerance=0.7
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
            'accuracy': accuracy
        })
        
        total_accuracy += accuracy
    
    avg_accuracy = total_accuracy / len(test_images)
    
    results.append({
        'config': (threshold, min_len, max_gap),
        'avg_accuracy': avg_accuracy,
        'details': test_results
    })
    
    lines_str = ', '.join([str(r['lines']) + 'L' for r in test_results])
    print(f"Thr:{threshold:2d} MinLen:{min_len:2d} MaxGap:{max_gap:2d} → Avg:{avg_accuracy*100:5.1f}%  [{lines_str}]")

print("─" * 80)

# Find best
best = max(results, key=lambda x: x['avg_accuracy'])

print("\n🏆 OPTIMAL BALANCED CONFIGURATION:")
print("=" * 80)
threshold, min_len, max_gap = best['config']
print(f"Threshold:        {threshold}")
print(f"Min Line Length:  {min_len}")
print(f"Max Line Gap:     {max_gap}")
print(f"Average Accuracy: {best['avg_accuracy']*100:.1f}%")

print(f"\n📊 Individual Results:")
for detail in best['details']:
    print(f"   {detail['name']:25s} | Lines:{detail['lines']:2d} Correct:{detail['correct']}/8 → {detail['accuracy']*100:5.1f}%")

print("\n💡 APPLYING THIS CONFIGURATION...")
print("=" * 80)
