#!/usr/bin/env python3
"""Test script for SVG dataset with on-the-fly augmentation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_svg import EmbroideryDatasetSVG
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Simple transform for testing
test_transform = A.Compose([
    A.LongestMaxSize(max_size=512),
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, fill=0, fill_mask=0),
    ToTensorV2()
])

# Test dataset
print("Testing SVG dataset...")
try:
    dataset = EmbroideryDatasetSVG(
        svg_dir="data/logo/train_svg", 
        transform=test_transform, 
        crops_per_image=1, 
        augment_color=True, 
        target_size=512
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    print("Loading first sample...")
    img, mask = dataset[0]
    
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"Mask unique values: {mask.unique().tolist()}")
    
    print("\n✓ Dataset test passed!")
    
except Exception as e:
    print(f"\n✗ Dataset test failed: {e}")
    import traceback
    traceback.print_exc()
