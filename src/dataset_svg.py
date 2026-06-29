import os
import glob
import cv2
import numpy as np
import torch
import copy
import random
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Tuple
from io import BytesIO
from PIL import Image

try:
    import cairosvg
except ImportError:
    print("Installing cairosvg...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg"])
    import cairosvg

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

# Label mapping
LABEL_BACKGROUND = 0
LABEL_FILL = 1
LABEL_SATIN = 2


def parse_svg_metadata(svg_path: str) -> Tuple[ET.Element, Dict[str, str]]:
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
    import re
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


class EmbroideryDatasetSVG(Dataset):
    def __init__(self, svg_dir, transform=None, crops_per_image=1, augment_color=True, target_size=512):
        self.svg_paths = sorted(glob.glob(f"{svg_dir}/*.svg"))
        self.transform = transform
        self.augment_color = augment_color
        self.target_size = target_size
        
        # HACK CHÍ MẠNG: Nhân bản danh sách để 1 ảnh gốc được load nhiều lần trong 1 Epoch
        self.svg_paths = self.svg_paths * crops_per_image

    def __len__(self):
        return len(self.svg_paths)

    def __getitem__(self, idx):
        svg_path = self.svg_paths[idx]
        
        # 1. Parse SVG and metadata
        root, metadata = parse_svg_metadata(svg_path)
        
        if not metadata:
            # Fallback if no metadata found
            raise ValueError(f"No metadata found in {svg_path}")
        
        # 2. Get SVG dimensions
        svg_width, svg_height = get_svg_dimensions(svg_path)
        
        # 3. Apply color augmentation if enabled
        if self.augment_color:
            # Use random seed for each call to get different augmentations
            seed = random.randint(0, 2**32 - 1)
            augment_svg_colors(root, metadata, seed=seed)
        
        # 4. Render augmented SVG to PNG
        svg_bytes = ET.tostring(root, encoding='unicode')
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes.encode('utf-8'),
                                     output_width=self.target_size,
                                     output_height=self.target_size,
                                     background_color=None,
                                     unsafe=True)
        
        # Load RGBA image
        img = Image.open(BytesIO(png_bytes))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_array = np.array(img)
        
        # Extract alpha channel as input
        alpha_channel = img_array[:, :, 3].astype(np.float32)
        
        # 5. Create label mask from original SVG (not augmented)
        mask = create_label_mask(svg_path, self.target_size, self.target_size, metadata)
        mask_binary = mask.astype(np.float32)
        
        # 6. Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=alpha_channel, mask=mask_binary)
            image_tensor = augmented['image'] / 255.0  # Normalize to [0, 1]
            mask_tensor = augmented['mask'].long()     # LongTensor for CrossEntropy
        else:
            # Fallback if no transform
            image_tensor = torch.tensor(alpha_channel / 255.0).unsqueeze(0)
            mask_tensor = torch.tensor(mask_binary).long()
        
        return image_tensor, mask_tensor
