#!/usr/bin/env python3
"""
Optimize line detection parameters for better line recognition.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage
from image_processing import normalize_image
from image_processing.line_detector import LineDetector

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 80)
print("🔧 LINE DETECTOR PARAMETER OPTIMIZATION")
print("=" * 80)

# Load test_4 (worst performer)
test_img = db.query(TestImage).filter(TestImage.id == 4).first()
nparr = np.frombuffer(test_img.image_data, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
normalized = normalize_image(image)

print(f"\n📗 Testing on: {test_img.test_name}")
print(f"   Target: Should detect ~8 lines")

# Test different parameter combinations
configs = [
    # (threshold, min_line_length, max_line_gap, description)
    (100, 80, 25, "Current (strict)"),
    (80, 70, 30, "Slightly relaxed"),
    (60, 60, 35, "Moderate"),
    (50, 50, 40, "Relaxed"),
    (40, 40, 50, "Very relaxed"),
    (30, 30, 60, "Extremely relaxed"),
]

print(f"\n🔬 Testing {len(configs)} configurations...")
print("─" * 80)

results = []

for threshold, min_len, max_gap, desc in configs:
    detector = LineDetector(
        threshold=threshold,
        min_line_length=min_len,
        max_line_gap=max_gap
    )
    
    features = detector.extract_features(normalized)
    num_lines = len(features['lines'])
    
    results.append({
        'config': (threshold, min_len, max_gap),
        'desc': desc,
        'lines': num_lines
    })
    
    print(f"{desc:25s} | Thr:{threshold:3d} MinLen:{min_len:2d} MaxGap:{max_gap:2d} → {num_lines} lines")

print("─" * 80)

# Find best (closest to 8)
best = min(results, key=lambda x: abs(x['lines'] - 8))

print("\n🏆 BEST CONFIGURATION:")
print("=" * 80)
threshold, min_len, max_gap = best['config']
print(f"Description:      {best['desc']}")
print(f"Threshold:        {threshold}")
print(f"Min Line Length:  {min_len}")
print(f"Max Line Gap:     {max_gap}")
print(f"Lines detected:   {best['lines']} (target: 8)")

print("\n💡 RECOMMENDATION:")
if best['config'] != configs[0][:3]:
    print(f"   Update LineDetector defaults:")
    print(f"   • threshold: 100 → {threshold}")
    print(f"   • min_line_length: 80 → {min_len}")
    print(f"   • max_line_gap: 25 → {max_gap}")
else:
    print(f"   Current defaults are optimal.")

# Test on all images with best config
print("\n" + "=" * 80)
print("📊 TESTING BEST CONFIG ON ALL IMAGES:")
print("=" * 80)

best_detector = LineDetector(
    threshold=threshold,
    min_line_length=min_len,
    max_line_gap=max_gap
)

test_images = db.query(TestImage).all()
for test_img in test_images:
    nparr = np.frombuffer(test_img.image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    normalized = normalize_image(image)
    
    features = best_detector.extract_features(normalized)
    num_lines = len(features['lines'])
    
    print(f"{test_img.test_name:25s} → {num_lines}/8 lines")

print("=" * 80)
