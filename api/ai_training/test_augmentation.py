"""
Test script for data augmentation functionality.

Usage:
    python -m ai_training.test_augmentation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import cv2
from pathlib import Path

from ai_training.data_augmentation import (
    ImageAugmentor,
    AugmentedDatasetBuilder,
    load_augmented_dataset,
    get_augmentation_stats
)


def create_test_image(width=568, height=274) -> np.ndarray:
    """Create a simple test drawing (house shape)."""
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Draw a simple house
    # Base rectangle
    cv2.rectangle(img, (200, 150), (368, 250), 255, 2)
    
    # Roof triangle
    cv2.line(img, (200, 150), (284, 100), 255, 2)
    cv2.line(img, (368, 150), (284, 100), 255, 2)
    
    # Door
    cv2.rectangle(img, (260, 200), (308, 250), 255, 2)
    
    # Windows
    cv2.rectangle(img, (220, 170), (250, 190), 255, 2)
    cv2.rectangle(img, (318, 170), (348, 190), 255, 2)
    
    return img


def test_single_augmentation():
    """Test single image augmentation."""
    print("=" * 60)
    print("TEST 1: Single Image Augmentation")
    print("=" * 60)
    
    # Create test image
    img = create_test_image()
    
    # Create augmentor
    augmentor = ImageAugmentor(
        rotation_range=(-3, 3),
        translation_range=(-10, 10),
        scale_range=(0.95, 1.05),
        num_augmentations=5
    )
    
    print(f"\nOriginal image shape: {img.shape}")
    
    # Test with specific parameters
    aug_img = augmentor.augment_image(
        img,
        rotation=2.0,
        tx=5,
        ty=-5,
        scale=1.02
    )
    
    print(f"Augmented image shape: {aug_img.shape}")
    print(f"✅ Single augmentation successful!")
    
    # Test batch augmentation
    aug_batch = augmentor.augment_batch(img, num_augmentations=3)
    
    print(f"\nGenerated {len(aug_batch)} augmented versions:")
    for i, (aug_img, params) in enumerate(aug_batch):
        print(f"  Aug {i+1}: rotation={params['rotation']:.2f}°, "
              f"tx={params['translation_x']}px, ty={params['translation_y']}px, "
              f"scale={params['scale']:.3f}")
    
    print(f"✅ Batch augmentation successful!")
    
    return True


def test_dataset_builder():
    """Test augmented dataset builder."""
    print("\n" + "=" * 60)
    print("TEST 2: Augmented Dataset Builder")
    print("=" * 60)
    
    # Create mock images_data
    images_data = []
    for i in range(5):
        img = create_test_image()
        
        # Convert to PNG bytes
        pil_img = Image.fromarray(img)
        import io
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='PNG')
        
        images_data.append({
            'id': i + 1,
            'patient_id': f'TEST_{i+1:03d}',
            'processed_image_data': img_bytes.getvalue(),
            'features_data': f'{{"Total_Score": {20 + i * 2}}}'
        })
    
    print(f"\nCreated {len(images_data)} mock images")
    
    # Create augmentor
    augmentor = ImageAugmentor(
        rotation_range=(-2, 2),
        translation_range=(-5, 5),
        scale_range=(0.97, 1.03),
        num_augmentations=3
    )
    
    # Create output directory
    output_dir = "/tmp/test_augmentation"
    
    # Build dataset
    builder = AugmentedDatasetBuilder(
        output_dir=output_dir,
        augmentor=augmentor,
        include_original=True
    )
    
    # Create split indices
    split_indices = {
        'train': [0, 1, 2],
        'val': [3, 4]
    }
    
    print(f"\nBuilding augmented dataset...")
    print(f"  Train split: {len(split_indices['train'])} images")
    print(f"  Val split: {len(split_indices['val'])} images")
    
    stats = builder.prepare_augmented_dataset(
        images_data=images_data,
        split_indices=split_indices,
        target_feature='Total_Score',
        clean_existing=True
    )
    
    print(f"\n✅ Dataset built successfully!")
    print(f"\nStatistics:")
    print(f"  Train: {stats['train']['total']} total "
          f"({stats['train']['original']} original + {stats['train']['augmented']} augmented)")
    print(f"  Val:   {stats['val']['total']} total "
          f"({stats['val']['original']} original + {stats['val']['augmented']} augmented)")
    
    if stats['errors']:
        print(f"\n⚠️ Errors: {len(stats['errors'])}")
        for error in stats['errors'][:3]:
            print(f"    - {error}")
    
    # Test loading
    print(f"\nTesting dataset loading...")
    train_images, train_targets, train_metadata = load_augmented_dataset(
        output_dir,
        split='train'
    )
    
    print(f"  Loaded {len(train_images)} training images")
    print(f"  Target range: [{min(train_targets):.1f}, {max(train_targets):.1f}]")
    print(f"✅ Dataset loading successful!")
    
    # Get stats
    print(f"\nGetting augmentation stats...")
    aug_stats = get_augmentation_stats(output_dir)
    print(f"  Metadata: {aug_stats.get('target_feature', 'N/A')}")
    print(f"✅ Stats retrieval successful!")
    
    return True


def test_augmentation_quality():
    """Test augmentation quality (visual check)."""
    print("\n" + "=" * 60)
    print("TEST 3: Augmentation Quality Check")
    print("=" * 60)
    
    img = create_test_image()
    
    augmentor = ImageAugmentor(
        rotation_range=(-3, 3),
        translation_range=(-10, 10),
        scale_range=(0.95, 1.05),
        num_augmentations=5
    )
    
    print(f"\nGenerating augmentations with extreme parameters...")
    
    # Test extreme cases
    test_cases = [
        {'rotation': 3.0, 'tx': 10, 'ty': 10, 'scale': 1.05, 'name': 'Max positive'},
        {'rotation': -3.0, 'tx': -10, 'ty': -10, 'scale': 0.95, 'name': 'Max negative'},
        {'rotation': 0.0, 'tx': 0, 'ty': 0, 'scale': 1.0, 'name': 'No change'}
    ]
    
    for case in test_cases:
        aug_img = augmentor.augment_image(
            img,
            rotation=case['rotation'],
            tx=case['tx'],
            ty=case['ty'],
            scale=case['scale']
        )
        
        # Check image properties
        assert aug_img.shape == img.shape, "Shape mismatch!"
        assert aug_img.dtype == img.dtype, "Dtype mismatch!"
        
        # Check that image is not completely black or white
        assert np.mean(aug_img) > 0, "Image is completely black!"
        assert np.mean(aug_img) < 255, "Image is completely white!"
        
        print(f"  ✅ {case['name']}: OK (mean={np.mean(aug_img):.1f})")
    
    print(f"\n✅ All quality checks passed!")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DATA AUGMENTATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Single Augmentation", test_single_augmentation),
        ("Dataset Builder", test_dataset_builder),
        ("Augmentation Quality", test_augmentation_quality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

