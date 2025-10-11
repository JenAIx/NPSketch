#!/usr/bin/env python3
"""
Deep dive into test_4 problem.
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

print("=" * 80)
print("🔬 DEEP DIVE: test_4 (rotated image)")
print("=" * 80)

# Load images
test_img = db.query(TestImage).filter(TestImage.id == 4).first()
ref = db.query(ReferenceImage).first()

nparr = np.frombuffer(test_img.image_data, np.uint8)
test_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
test_normalized = normalize_image(test_image)

ref_image = load_image_from_bytes(ref.processed_image_data)

detector = LineDetector()
ref_data = detector.features_from_json(ref.feature_data)

print(f"\n📘 Reference Lines: {len(ref_data['lines'])}")
for i, line in enumerate(ref_data['lines']):
    print(f"   [{i}] {line}")

# Step 1: What lines are detected BEFORE registration?
print(f"\n📗 Test Image (BEFORE registration):")
test_features_before = detector.extract_features(test_normalized)
print(f"   Lines detected: {len(test_features_before['lines'])}")
for i, line in enumerate(test_features_before['lines']):
    print(f"   [{i}] {line}")

# Step 2: Try registration
print(f"\n🔄 Image Registration:")
registration = ImageRegistration()

try:
    registered, reg_info = registration.register_images(
        test_normalized,
        ref_image,
        method="ecc",
        motion_type="similarity",
        max_rotation_degrees=60.0  # Try high rotation
    )
    
    if reg_info.get('used', False):
        print(f"   ✅ Registration successful!")
        print(f"   Translation: ({reg_info.get('translation_x', 0):.1f}px, {reg_info.get('translation_y', 0):.1f}px)")
        print(f"   Rotation: {reg_info.get('rotation_degrees', 0):.1f}°")
        print(f"   Scale: {reg_info.get('scale', 1.0):.3f}x")
        print(f"   Correlation: {reg_info.get('correlation_coefficient', 0):.3f}")
    else:
        print(f"   ❌ Registration failed!")
        if 'error' in reg_info:
            print(f"   Error: {reg_info['error']}")
        registered = test_normalized

except Exception as e:
    print(f"   ❌ Registration error: {e}")
    registered = test_normalized

# Step 3: What lines are detected AFTER registration?
print(f"\n📗 Test Image (AFTER registration):")
test_features_after = detector.extract_features(registered)
print(f"   Lines detected: {len(test_features_after['lines'])}")
for i, line in enumerate(test_features_after['lines']):
    print(f"   [{i}] {line}")

# Step 4: Visual comparison - save images for inspection
print(f"\n💾 Saving debug images...")

# Save original
cv2.imwrite('/app/data/test_output/test4_1_original.png', test_normalized)
print(f"   • test4_1_original.png")

# Save registered
cv2.imwrite('/app/data/test_output/test4_2_registered.png', registered)
print(f"   • test4_2_registered.png")

# Save reference
cv2.imwrite('/app/data/test_output/test4_3_reference.png', ref_image)
print(f"   • test4_3_reference.png")

# Draw lines on images
orig_with_lines = test_normalized.copy()
for line in test_features_before['lines']:
    x1, y1, x2, y2 = line
    cv2.line(orig_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite('/app/data/test_output/test4_4_original_lines.png', orig_with_lines)
print(f"   • test4_4_original_lines.png (green=detected)")

reg_with_lines = registered.copy()
for line in test_features_after['lines']:
    x1, y1, x2, y2 = line
    cv2.line(reg_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite('/app/data/test_output/test4_5_registered_lines.png', reg_with_lines)
print(f"   • test4_5_registered_lines.png (green=detected)")

ref_with_lines = ref_image.copy()
for line in ref_data['lines']:
    x1, y1, x2, y2 = line
    cv2.line(ref_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite('/app/data/test_output/test4_6_reference_lines.png', ref_with_lines)
print(f"   • test4_6_reference_lines.png (red=reference)")

print("\n" + "=" * 80)
print("💡 DIAGNOSIS:")
print("=" * 80)

lines_before = len(test_features_before['lines'])
lines_after = len(test_features_after['lines'])

print(f"\nLines detected: {lines_before} → {lines_after} (after registration)")

if lines_before == 0 or lines_after == 0:
    print("❌ Line detection is failing! Check the images in ./data/test_output/")
elif not reg_info.get('used', False):
    print("❌ Registration is not working! Image not being aligned.")
elif abs(reg_info.get('rotation_degrees', 0)) < 5:
    print("⚠️  Registration detected very small rotation.")
    print("   The image might not be rotated as much as expected,")
    print("   or registration is not working properly.")
else:
    print("✅ Registration is working, but line matching still fails.")
    print("   Problem might be in the test image quality or line drawing.")

print(f"\n📁 Check files in: ./data/test_output/test4_*.png")
print("=" * 80)
