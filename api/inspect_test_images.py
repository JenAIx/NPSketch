#!/usr/bin/env python3
"""
Inspect what the test images actually contain.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 60)
print("🔍 INSPECT TEST IMAGES")
print("=" * 60)

# Get reference for comparison
ref = db.query(ReferenceImage).first()
print(f"\n📘 Reference Image:")
print(f"   ID: {ref.id}")
print(f"   Name: {ref.name}")
print(f"   Image size: {len(ref.processed_image_data)} bytes")

# Get all test images
test_images = db.query(TestImage).all()
print(f"\n📗 Test Images: {len(test_images)}")

for test_img in test_images:
    print(f"\n{test_img.id}. {test_img.test_name}")
    print(f"   Expected: C={test_img.expected_correct}, M={test_img.expected_missing}, E={test_img.expected_extra}")
    print(f"   Image size: {len(test_img.image_data)} bytes")
    
    # Decode image
    nparr = np.frombuffer(test_img.image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(f"   Decoded shape: {image.shape}")
    print(f"   Data type: {image.dtype}")
    
    # Check if image is meaningful
    mean_val = np.mean(image)
    std_val = np.std(image)
    print(f"   Mean pixel value: {mean_val:.1f}")
    print(f"   Std deviation: {std_val:.1f}")
    
    # Save for visual inspection
    output_path = f"/app/data/test_output/inspect_{test_img.id}_{test_img.test_name}.png"
    cv2.imwrite(output_path, image)
    print(f"   💾 Saved: {output_path}")

print("\n" + "=" * 60)
print("💡 Check ./data/test_output/inspect_*.png to see actual images")
print("=" * 60)
