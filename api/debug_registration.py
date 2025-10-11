#!/usr/bin/env python3
"""
Debug registration failure.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage
from image_processing import normalize_image, load_image_from_bytes
from image_processing.image_registration import ImageRegistration

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 60)
print("🔍 DEBUG REGISTRATION")
print("=" * 60)

# Load images
ref = db.query(ReferenceImage).first()
ref_image = load_image_from_bytes(ref.processed_image_data)
print(f"📘 Reference shape: {ref_image.shape}")

test_img = db.query(TestImage).filter(TestImage.test_name == "Supper").first()
nparr = np.frombuffer(test_img.image_data, np.uint8)
test_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
test_normalized = normalize_image(test_image)
print(f"📗 Test shape: {test_normalized.shape}")

# Try registration
registration = ImageRegistration()

print("\n🔄 Trying registration with 'similarity' motion...")
try:
    registered, info = registration.register_images(
        test_normalized,
        ref_image,
        method="ecc",
        motion_type="similarity",
        max_rotation_degrees=30.0
    )
    print("✅ Success!")
    print(f"   Info: {info}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🔄 Trying with 'euclidean' motion...")
try:
    registered, info = registration.register_images(
        test_normalized,
        ref_image,
        method="ecc",
        motion_type="euclidean",
        max_rotation_degrees=30.0
    )
    print("✅ Success!")
    print(f"   Info: {info}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("=" * 60)
