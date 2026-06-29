#!/usr/bin/env python3
"""
SVG to PNG Converter with Metadata Labels for Training
Convert SVG files with stitch type metadata (satin/fill) to PNG images and label masks.
"""

import os
import re
import copy
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from typing import Dict, Tuple
from tqdm import tqdm
import random
import hashlib

try:
    import cairosvg
except ImportError:
    print("Installing cairosvg...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg"])
    import cairosvg

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Label mapping
LABEL_BACKGROUND = 0
LABEL_FILL = 1
LABEL_SATIN = 2

# Color palettes for augmentation (bright colors to avoid confusion with background)
SATIN_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FFA500", "#800080", "#FFC0CB", "#FFD700", "#C0C0C0", "#A52A2A"
]

FILL_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD",
    "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8B500", "#FF6F61",
    "#88B04B", "#F7CAC9", "#92A8D1", "#B565A7"
]


def get_svg_dimensions(svg_path: str) -> Tuple[int, int]:
    """Get original SVG dimensions from viewBox or width/height attributes."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Try viewBox first
    viewbox = root.get("viewBox")
    if viewbox:
        try:
            vb_x, vb_y, vb_w, vb_h = map(float, viewbox.split())
            return int(vb_w), int(vb_h)
        except (ValueError, IndexError):
            pass
    
    # Fallback to width/height attributes with regex for unit parsing
    width_str = root.get("width")
    height_str = root.get("height")
    
    if width_str and height_str:
        # Extract numeric value using regex (handles px, mm, cm, %, etc.)
        width_match = re.search(r'([\d.]+)', width_str)
        height_match = re.search(r'([\d.]+)', height_str)
        
        if width_match and height_match:
            width = int(float(width_match.group(1)))
            height = int(float(height_match.group(1)))
            return width, height
    
    # Default fallback
    return 512, 512


def parse_svg_with_metadata(svg_path: str) -> Tuple[ET.Element, Dict[str, str]]:
    """Parse SVG file and extract stitch type metadata from paths using inkscape:label."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    metadata = {}
    
    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            path_id = child.get("id")
            # Read inkscape:label attribute
            stitch_type = child.get("{http://www.inkscape.org/namespaces/inkscape}label", "fill")
            # Normalize metadata
            if path_id is not None:
                metadata[path_id] = stitch_type.strip().lower() if stitch_type else "fill"
    
    return root, metadata


def augment_svg_colors(root: ET.Element, metadata: Dict[str, str], seed: int = None) -> None:
    """Apply random color augmentation to SVG paths based on metadata.
    
    Each path gets a unique color from the appropriate palette to avoid duplicates.
    Colors are bright to avoid confusion with background (0 in grayscale).
    Handles both 'fill' attribute and 'style' attribute.
    """
    if seed is not None:
        random.seed(seed)
    
    # Track used colors to avoid duplicates
    used_satin_colors = set()
    used_fill_colors = set()
    
    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            path_id = child.get("id")
            stitch_type = metadata.get(path_id, "fill")
            stitch_type = stitch_type.strip().lower() if stitch_type else "fill"
            
            # Generate unique color for each path
            if stitch_type == "satin":
                # Find unused satin color
                available_satin = [c for c in SATIN_COLORS if c not in used_satin_colors]
                if available_satin:
                    new_color = random.choice(available_satin)
                    used_satin_colors.add(new_color)
                else:
                    # If all colors used, pick random anyway
                    new_color = random.choice(SATIN_COLORS)
            elif stitch_type == "fill":
                # Find unused fill color
                available_fill = [c for c in FILL_COLORS if c not in used_fill_colors]
                if available_fill:
                    new_color = random.choice(available_fill)
                    used_fill_colors.add(new_color)
                else:
                    # If all colors used, pick random anyway
                    new_color = random.choice(FILL_COLORS)
            else:
                # Default to fill colors
                available_fill = [c for c in FILL_COLORS if c not in used_fill_colors]
                if available_fill:
                    new_color = random.choice(available_fill)
                    used_fill_colors.add(new_color)
                else:
                    new_color = random.choice(FILL_COLORS)
            
            # Set fill color in fill attribute
            child.set("fill", new_color)
            
            # Also update color in style attribute if present
            style = child.get("style")
            if style:
                # Replace fill color in style string
                # Pattern: fill:#XXXXXX or fill: #XXXXXX
                import re
                new_style = re.sub(r'fill\s*:\s*#[0-9A-Fa-f]{6}', f'fill:{new_color}', style)
                child.set("style", new_style)


def create_label_mask(svg_path: str, width: int, height: int, metadata: Dict[str, str]) -> np.ndarray:
    """Create label mask from SVG metadata by rendering paths in SVG order."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Initialize mask with background
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Parse viewBox if exists
    viewbox = root.get("viewBox")
    if viewbox:
        try:
            vb_x, vb_y, vb_w, vb_h = map(float, viewbox.split())
        except (ValueError, IndexError):
            # Fallback if viewBox is invalid
            svg_width = float(root.get("width", width))
            svg_height = float(root.get("height", height))
            viewbox = f"0 0 {svg_width} {svg_height}"
    else:
        svg_width = float(root.get("width", width))
        svg_height = float(root.get("height", height))
        viewbox = f"0 0 {svg_width} {svg_height}"
    
    # Render paths in SVG order to preserve layer order for overlapping regions
    from io import BytesIO
    from PIL import Image
    
    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            path_id = child.get("id")
            stitch_type = metadata.get(path_id, "fill")
            # Normalize stitch type
            stitch_type = stitch_type.strip().lower() if stitch_type else "fill"
            
            # Map stitch type to label
            if stitch_type == "satin":
                label_value = LABEL_SATIN
            elif stitch_type == "fill":
                label_value = LABEL_FILL
            else:
                label_value = LABEL_FILL  # Default to fill
            
            # Create SVG with this single path
            temp_root = ET.Element("svg")
            temp_root.set("width", str(width))
            temp_root.set("height", str(height))
            temp_root.set("viewBox", viewbox)
            temp_root.set("xmlns", "http://www.w3.org/2000/svg")
            
            temp_root.append(copy.deepcopy(child))
            
            # Convert to PNG bytes with unsafe=True for Inkscape compatibility
            svg_bytes = ET.tostring(temp_root, encoding='unicode')
            png_bytes = cairosvg.svg2png(bytestring=svg_bytes.encode('utf-8'), 
                                         output_width=width, 
                                         output_height=height,
                                         background_color=None,
                                         unsafe=True)
            
            # Load RGBA image and use alpha channel for mask
            img = Image.open(BytesIO(png_bytes))
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            img_array = np.array(img)
            
            # Use alpha channel: alpha >= 128 to avoid anti-aliasing artifacts
            alpha_channel = img_array[:, :, 3]
            mask[alpha_channel >= 128] = label_value
    
    return mask


def render_svg_to_png(svg_path: str, output_path: str, width: int = None, height: int = None):
    """Render SVG to PNG image with transparent background."""
    kwargs = {
        'url': svg_path,
        'write_to': output_path,
        'background_color': None,
        'unsafe': True  # For Inkscape compatibility
    }
    if width is not None:
        kwargs['output_width'] = width
    if height is not None:
        kwargs['output_height'] = height
    cairosvg.svg2png(**kwargs)


def render_svg_element_to_png(root: ET.Element, output_path: str, width: int, height: int) -> None:
    """Render SVG element tree directly to PNG without saving to file."""
    svg_bytes = ET.tostring(root, encoding='unicode')
    cairosvg.svg2png(bytestring=svg_bytes.encode('utf-8'),
                     write_to=output_path,
                     output_width=width,
                     output_height=height,
                     background_color=None,
                     unsafe=True)


def process_svg_folder(svg_dir: str, output_img_dir: str, output_mask_dir: str, 
                       width: int = None, height: int = None, augment: bool = True, 
                       num_variants: int = 1):
    """Process all SVG files in folder to PNG images and label masks.
    
    Args:
        svg_dir: Input SVG directory
        output_img_dir: Output image directory
        output_mask_dir: Output mask directory
        width: Output image width (default: original SVG size)
        height: Output image height (default: original SVG size)
        augment: Whether to apply color augmentation (default: True)
        num_variants: Number of augmented variants to generate per SVG (default: 1)
    """
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    svg_files = list(Path(svg_dir).rglob("*.svg"))
    print(f"Found {len(svg_files)} SVG files")
    
    total_generated = 0
    
    for svg_file in tqdm(svg_files, desc="Converting SVG to PNG"):
        try:
            # Parse metadata
            root, metadata = parse_svg_with_metadata(str(svg_file))
            
            if not metadata:
                print(f"  Warning: No metadata found in {svg_file.name}, skipping")
                continue
            
            # Get dimensions if not specified
            img_width = width
            img_height = height
            if img_width is None or img_height is None:
                img_width, img_height = get_svg_dimensions(str(svg_file))
            
            # Generate variants
            for variant_idx in range(num_variants):
                # Create a fresh copy of the root for each variant
                root_copy = copy.deepcopy(root)
                
                # Apply color augmentation if enabled
                if augment:
                    # Use stable seed based on filename + variant index for reproducibility
                    seed = int(hashlib.md5(f"{svg_file.stem}_variant{variant_idx}".encode()).hexdigest(), 16)
                    augment_svg_colors(root_copy, metadata, seed=seed)
                
                # Render augmented SVG to PNG
                if num_variants > 1:
                    img_output = os.path.join(output_img_dir, f"{svg_file.stem}_variant{variant_idx}.png")
                else:
                    img_output = os.path.join(output_img_dir, f"{svg_file.stem}.png")
                render_svg_element_to_png(root_copy, img_output, img_width, img_height)
                
                total_generated += 1
            
            # Create label mask for each variant (masks are identical but need separate files)
            # Reuse metadata already parsed - augment only changes colors, not shape or labels
            for variant_idx in range(num_variants):
                if num_variants > 1:
                    mask_output = os.path.join(output_mask_dir, f"{svg_file.stem}_variant{variant_idx}.png")
                else:
                    mask_output = os.path.join(output_mask_dir, f"{svg_file.stem}.png")
                mask = create_label_mask(str(svg_file), img_width, img_height, metadata)
                cv2.imwrite(mask_output, mask)
            
        except Exception as e:
            print(f"  Error processing {svg_file.name}: {e}")
            continue
    
    print(f"\nCompleted!")
    print(f"Generated {total_generated} augmented images")
    print(f"Images saved to: {output_img_dir}")
    print(f"Masks saved to: {output_mask_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SVG files with metadata to PNG images and label masks for training"
    )
    parser.add_argument("--svg-dir", default="data/svg/logo_new", help="Input SVG directory")
    parser.add_argument("--output-img", default="data/svg/logo_label/images", help="Output image directory")
    parser.add_argument("--output-mask", default="data/svg/logo_label/masks", help="Output mask directory")
    parser.add_argument("--width", type=int, default=None, help="Output image width (default: original SVG size)")
    parser.add_argument("--height", type=int, default=None, help="Output image height (default: original SVG size)")
    parser.add_argument("--no-augment", action="store_true", help="Disable color augmentation (use original colors)")
    parser.add_argument("--num-variants", type=int, default=1, help="Number of augmented variants per SVG (default: 1)")
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    svg_dir = os.path.join(PROJECT_ROOT, args.svg_dir)
    output_img_dir = os.path.join(PROJECT_ROOT, args.output_img)
    output_mask_dir = os.path.join(PROJECT_ROOT, args.output_mask)
    
    process_svg_folder(svg_dir, output_img_dir, output_mask_dir, args.width, args.height, augment=not args.no_augment, num_variants=args.num_variants)
