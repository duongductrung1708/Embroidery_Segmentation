#!/usr/bin/env python3
"""
Split SVG files from easy/medium/hard folders into train/val sets.
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def split_svg_files(source_dirs: List[str], train_dir: str, val_dir: str, 
                   train_ratio: float = 0.7, seed: int = 42):
    """Split SVG files from multiple source directories into train/val sets.
    
    Args:
        source_dirs: List of source directories containing SVG files (e.g., easy, medium, hard)
        train_dir: Output directory for training set
        val_dir: Output directory for validation set
        train_ratio: Ratio of training data (default: 0.7)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    all_train_files = []
    all_val_files = []
    
    for source_dir in source_dirs:
        source_path = os.path.join(PROJECT_ROOT, source_dir)
        if os.path.exists(source_path):
            svg_files = list(Path(source_path).rglob("*.svg"))
            print(f"Found {len(svg_files)} SVG files in {source_dir}")
        else:
            print(f"Source directory not found: {source_path}")
            continue
        
        if len(svg_files) == 0:
            print(f"No SVG files found in {source_dir}")
            continue
        
        # Shuffle and split for this directory
        random.shuffle(svg_files)
        split_idx = int(len(svg_files) * train_ratio)
        
        train_files = svg_files[:split_idx]
        val_files = svg_files[split_idx:]
        
        print(f"  - Train: {len(train_files)} files")
        print(f"  - Val: {len(val_files)} files")
        
        all_train_files.extend(train_files)
        all_val_files.extend(val_files)
    
    print(f"\nTotal across all directories:")
    print(f"Train: {len(all_train_files)} files")
    print(f"Val: {len(all_val_files)} files")
    
    # Copy files to train directory
    for svg_file in all_train_files:
        # Add prefix to avoid name conflicts if needed
        dest_name = svg_file.name
        shutil.copy2(svg_file, os.path.join(train_dir, dest_name))
    
    # Copy files to val directory
    for svg_file in all_val_files:
        dest_name = svg_file.name
        shutil.copy2(svg_file, os.path.join(val_dir, dest_name))
    
    print(f"\nCompleted!")
    print(f"Train files saved to: {train_dir}")
    print(f"Val files saved to: {val_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split SVG files from easy/medium/hard into train/val sets"
    )
    parser.add_argument("--source-dirs", nargs='+', 
                       default=["data/logo/easy", "data/logo/medium", "data/logo/hard"],
                       help="Source directories containing SVG files (default: data/logo/easy data/logo/medium data/logo/hard)")
    parser.add_argument("--train-dir", default="data/logo/train_svg",
                       help="Output directory for training set")
    parser.add_argument("--val-dir", default="data/logo/val_svg",
                       help="Output directory for validation set")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Ratio of training data (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    train_dir = os.path.join(PROJECT_ROOT, args.train_dir)
    val_dir = os.path.join(PROJECT_ROOT, args.val_dir)
    
    split_svg_files(args.source_dirs, train_dir, val_dir, args.train_ratio, args.seed)
