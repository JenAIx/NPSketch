#!/usr/bin/env python3
"""
Debug script to understand why line detection is failing.
"""

import sys
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, '/app')
from database import TestImage, ReferenceImage
from image_processing import LineDetector, load_image_from_bytes, normalize_image

# Database setup
DATABASE_URL = "sqlite:////app/data/npsketch.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

print("=" * 60)
print("🔍 DEBUG: Line Detection Analysis")
print("=" * 60)

# Load reference
ref = db.query(ReferenceImage).first()
if ref:
    print(f"\n📘 Reference Image:")
    ref_data = LineDetector().features_from_json(ref.feature_data)
    print(f"   Lines detected: {len(ref_data['lines'])}")
    print(f"   Lines: {ref_data['lines'][:3]}...") # Show first 3

# Load test images
test_images = db.query(TestImage).all()
print(f"\n📗 Test Images: {len(test_images)}")

detector = LineDetector()

for idx, test_img in enumerate(test_images, 1):
    print(f"\n[{idx}] {test_img.test_name} (ID: {test_img.id})")
    print(f"   Expected: C={test_img.expected_correct}, M={test_img.expected_missing}, E={test_img.expected_extra}")
    
    try:
        # Load and process image
        nparr = np.frombuffer(test_img.image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(f"   Image shape: {image.shape}")
        
        # Normalize
        normalized = normalize_image(image)
        print(f"   Normalized shape: {normalized.shape}")
        
        # Detect lines
        features = detector.detect_lines(normalized)
        lines = features.get('lines', [])
        print(f"   ❌ Lines detected: {len(lines)}")
        
        if len(lines) == 0:
            # Debug: Check if image is mostly white/blank
            gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray)
            black_pixels = np.sum(gray < 128)
            total_pixels = gray.shape[0] * gray.shape[1]
            black_pct = (black_pixels / total_pixels) * 100
            
            print(f"   📊 Image stats:")
            print(f"      Mean gray value: {mean_val:.1f}")
            print(f"      Black pixels: {black_pct:.1f}%")
            
            # Save debug image
            debug_path = f"/app/data/test_output/debug_{test_img.id}.png"
            cv2.imwrite(debug_path, normalized)
            print(f"   💾 Saved debug image: {debug_path}")
        else:
            print(f"   ✅ Lines: {lines[:2]}...")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("💡 Conclusion:")
print("   If lines detected = 0, the test images might be:")
print("   - Completely white/blank")
print("   - Too faint for Hough Transform")
print("   - Not normalized correctly")
print("=" * 60)
