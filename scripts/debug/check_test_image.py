#!/usr/bin/env python3
"""
Debug script to check test image characteristics
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: cv2 not available")
    sys.exit(1)

TEST_DIR = os.path.join(PROJECT_ROOT, "data/test/logo")

print("Checking test images...")
for img_file in Path(TEST_DIR).glob("*.png"):
    print(f"\n=== {img_file.name} ===")
    img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
    print(f"Shape: {img.shape}")
    print(f"Has alpha: {img.shape[2] == 4 if len(img.shape) == 3 else False}")
    
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        print(f"Alpha range: {alpha.min()} - {alpha.max()}")
        print(f"Alpha unique values: {len(np.unique(alpha))}")
        print(f"Alpha mean: {alpha.mean():.2f}")
        
        # Check if alpha is mostly 0 or 255
        alpha_0_ratio = (alpha == 0).sum() / alpha.size
        alpha_255_ratio = (alpha == 255).sum() / alpha.size
        print(f"Alpha=0 ratio: {alpha_0_ratio:.2%}")
        print(f"Alpha=255 ratio: {alpha_255_ratio:.2%}")
