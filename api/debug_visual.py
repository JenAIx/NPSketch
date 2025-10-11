#!/usr/bin/env python3
"""
Visual debug - compare detected lines.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage
from image_processing import LineDetector, normalize_image, load_image_from_bytes
from image_processing.image_registration import ImageRegistration

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 60)
print("🔍 VISUAL DEBUG")
print("=" * 60)

# Load reference
ref = db.query(ReferenceImage).first()
ref_image = load_image_from_bytes(ref.processed_image_data)
ref_data = LineDetector().features_from_json(ref.feature_data)
print(f"\n📘 Reference: {len(ref_data['lines'])} lines")

# Load test image
test_img = db.query(TestImage).filter(TestImage.test_name == "Supper").first()
nparr = np.frombuffer(test_img.image_data, np.uint8)
test_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
test_normalized = normalize_image(test_image)

print(f"📗 Test Image: {test_img.test_name}")

# Extract lines from test
detector = LineDetector()
test_features = detector.extract_features(test_normalized)
print(f"   Lines detected (before registration): {len(test_features['lines'])}")
for i, line in enumerate(test_features['lines']):
    print(f"      [{i}] {line}")

# Register image
registration = ImageRegistration()
registered, reg_info = registration.register_images(
    test_normalized,
    ref_image,
    method="ecc",
    motion_type="similarity",
    max_rotation_degrees=30.0
)

print(f"\n🔄 Registration:")
print(f"   Used: {reg_info.get('used', False)}")
if reg_info.get('used'):
    print(f"   Tx: {reg_info.get('translation_x', 0):.1f}px")
    print(f"   Ty: {reg_info.get('translation_y', 0):.1f}px")
    print(f"   Rotation: {reg_info.get('rotation_degrees', 0):.1f}°")
    print(f"   Scale: {reg_info.get('scale', 1.0):.2f}x")

# Extract lines from registered
registered_features = detector.extract_features(registered)
print(f"\n📗 Lines detected (after registration): {len(registered_features['lines'])}")
for i, line in enumerate(registered_features['lines']):
    print(f"   [{i}] {line}")

# Compare with reference
print(f"\n📘 Reference lines:")
for i, line in enumerate(ref_data['lines']):
    print(f"   [{i}] {line}")

# Save visualization
output_path = "/app/data/test_output/debug_comparison.png"

# Create 3-way visualization
vis_orig = test_normalized.copy()
vis_reg = registered.copy()
vis_ref = ref_image.copy()

# Draw lines
for line in test_features['lines']:
    x1, y1, x2, y2 = line
    cv2.line(vis_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)

for line in registered_features['lines']:
    x1, y1, x2, y2 = line
    cv2.line(vis_reg, (x1, y1), (x2, y2), (0, 255, 0), 2)

for line in ref_data['lines']:
    x1, y1, x2, y2 = line
    cv2.line(vis_ref, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Combine
combined = np.hstack([vis_orig, vis_reg, vis_ref])

# Labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(combined, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
cv2.putText(combined, "Registered", (260, 30), font, 0.7, (255, 255, 255), 2)
cv2.putText(combined, "Reference", (520, 30), font, 0.7, (255, 255, 255), 2)

cv2.imwrite(output_path, combined)
print(f"\n💾 Saved: {output_path}")

print("=" * 60)
