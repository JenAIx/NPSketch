#!/usr/bin/env python3
"""
Enhanced debug - check if extract_features works.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage
from image_processing import LineDetector, normalize_image

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 60)
print("🔍 ENHANCED DEBUG: Testing extract_features")
print("=" * 60)

# Load test image
test_img = db.query(TestImage).first()
if not test_img:
    print("No test images found!")
    sys.exit(1)

print(f"\n📗 Test Image: {test_img.test_name}")

# Load image
nparr = np.frombuffer(test_img.image_data, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
print(f"   Loaded image shape: {image.shape}")
print(f"   Image dtype: {image.dtype}")

# Normalize
normalized = normalize_image(image)
print(f"   Normalized shape: {normalized.shape}")
print(f"   Normalized dtype: {normalized.dtype}")

# Try extract_features
detector = LineDetector()
print(f"\n🔄 Calling extract_features...")
try:
    features = detector.extract_features(normalized)
    print(f"✅ Success!")
    print(f"   Lines detected: {len(features['lines'])}")
    print(f"   Lines: {features['lines'][:3]}")
except Exception as e:
    print(f"❌ FAILED:")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
