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

SATIN_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FFA500", "#800080", "#FFC0CB", "#FFD700", "#C0C0C0", "#A52A2A"
]

FILL_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD",
    "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8B500", "#FF6F61",
    "#88B04B", "#F7CAC9", "#92A8D1", "#B565A7"
]

LABEL_BACKGROUND = 0
LABEL_FILL = 1
LABEL_SATIN = 2

_MASK_COLOR_FILL = "#FF0000"   
_MASK_COLOR_SATIN = "#00FF00"  

def parse_svg_metadata(svg_path: str) -> Tuple[ET.Element, Dict[str, str]]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    metadata = {}
    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            path_id = child.get("id")
            stitch_type = child.get("{http://www.inkscape.org/namespaces/inkscape}label", "fill")
            if path_id is not None:
                metadata[path_id] = stitch_type.strip().lower() if stitch_type else "fill"
    
    return root, metadata

def augment_svg_colors(root: ET.Element, metadata: Dict[str, str], seed: int = None) -> None:
    if seed is not None:
        random.seed(seed)
    
    used_satin_colors = set()
    used_fill_colors = set()
    
    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            path_id = child.get("id")
            stitch_type = metadata.get(path_id, "fill")
            stitch_type = stitch_type.strip().lower() if stitch_type else "fill"
            
            if stitch_type == "satin":
                available_satin = [c for c in SATIN_COLORS if c not in used_satin_colors]
                new_color = random.choice(available_satin) if available_satin else random.choice(SATIN_COLORS)
                used_satin_colors.add(new_color)
            else:
                available_fill = [c for c in FILL_COLORS if c not in used_fill_colors]
                new_color = random.choice(available_fill) if available_fill else random.choice(FILL_COLORS)
                used_fill_colors.add(new_color)
            
            child.set("fill", new_color)
            style = child.get("style")
            if style:
                import re
                new_style = re.sub(r'fill\s*:\s*#[0-9A-Fa-f]{6}', f'fill:{new_color}', style)
                child.set("style", new_style)

def create_label_mask(svg_path: str, width: int, height: int, metadata: Dict[str, str],
                       supersample_factor: int = 1) -> np.ndarray:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            path_id = child.get("id")
            stitch_type = metadata.get(path_id, "fill")
            stitch_type = stitch_type.strip().lower() if stitch_type else "fill"
            id_color = _MASK_COLOR_SATIN if stitch_type == "satin" else _MASK_COLOR_FILL
            child.set("fill", id_color)
            child.attrib.pop("style", None)
            child.set("fill-opacity", "1")
            child.set("stroke", "none")

    render_width = width * supersample_factor
    render_height = height * supersample_factor

    svg_bytes = ET.tostring(root, encoding='unicode')
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes.encode('utf-8'),
                                 output_width=render_width,
                                 output_height=render_height,
                                 background_color=None,
                                 unsafe=True)

    img = Image.open(BytesIO(png_bytes))
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img_array = np.array(img)

    alpha = img_array[:, :, 3]
    r = img_array[:, :, 0].astype(np.int16)
    g = img_array[:, :, 1].astype(np.int16)
    is_visible = alpha >= 128

    mask_render = np.zeros((render_height, render_width), dtype=np.uint8)
    mask_render[is_visible & (g > r)] = LABEL_SATIN
    mask_render[is_visible & (r >= g)] = LABEL_FILL

    if supersample_factor == 1:
        return mask_render

    mask = cv2.resize(mask_render, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask

def render_svg_to_rgba(root: ET.Element, width: int, height: int,
                             supersample_factor: int = 1) -> np.ndarray:
    """Render 1 cây SVG ra thẳng mảng RGBA (4 kênh)."""
    render_width = width * supersample_factor
    render_height = height * supersample_factor

    svg_bytes = ET.tostring(root, encoding='unicode')
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes.encode('utf-8'),
                                 output_width=render_width,
                                 output_height=render_height,
                                 background_color=None, # Giữ nguyên độ trong suốt
                                 unsafe=True)

    img = Image.open(BytesIO(png_bytes))
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    img_array = np.array(img) # Shape: (H, W, 4)

    if supersample_factor == 1:
        return img_array

    # Downsample cả 4 kênh bằng INTER_AREA
    rgba_image = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_AREA)
    return rgba_image

def get_svg_dimensions(svg_path: str) -> Tuple[int, int]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    viewbox = root.get("viewBox")
    if viewbox:
        try:
            vb_x, vb_y, vb_w, vb_h = map(float, viewbox.split())
            return int(vb_w), int(vb_h)
        except (ValueError, IndexError):
            pass
    import re
    width_str = root.get("width")
    height_str = root.get("height")
    if width_str and height_str:
        width_match = re.search(r'([\d.]+)', width_str)
        height_match = re.search(r'([\d.]+)', height_str)
        if width_match and height_match:
            width = int(float(width_match.group(1)))
            height = int(float(height_match.group(1)))
            return width, height
    return 512, 512

class EmbroideryDatasetSVG(Dataset):
    def __init__(self, svg_dir, transform=None, crops_per_image=1, augment_color=True,
                 target_size=512, cache_in_memory=True, supersample_factor=1):
        self.svg_paths = sorted(glob.glob(f"{svg_dir}/*.svg"))
        self.transform = transform
        self.augment_color = augment_color
        self.target_size = target_size
        self.cache_in_memory = cache_in_memory
        self.supersample_factor = supersample_factor
        self._cache: Dict[str, Tuple[Dict[str, str], np.ndarray]] = {}
        self.svg_paths = self.svg_paths * crops_per_image

    def __len__(self):
        return len(self.svg_paths)

    def _get_metadata_and_mask(self, svg_path: str) -> Tuple[Dict[str, str], np.ndarray]:
        if self.cache_in_memory and svg_path in self._cache:
            return self._cache[svg_path]
        _, metadata = parse_svg_metadata(svg_path)
        if not metadata:
            raise ValueError(f"No metadata found in {svg_path}")
        mask = create_label_mask(svg_path, self.target_size, self.target_size, metadata,
                                  supersample_factor=self.supersample_factor)
        if self.cache_in_memory:
            self._cache[svg_path] = (metadata, mask)
        return metadata, mask

    def __getitem__(self, idx):
        svg_path = self.svg_paths[idx]
        metadata, mask = self._get_metadata_and_mask(svg_path)
        mask_binary = mask.astype(np.float32)

        root, _ = parse_svg_metadata(svg_path)
        svg_width, svg_height = get_svg_dimensions(svg_path)

        if self.augment_color:
            seed = random.randint(0, 2**32 - 1)
            augment_svg_colors(root, metadata, seed=seed)

        # Trích xuất ảnh RGBA 4 kênh
        rgba_image = render_svg_to_rgba(
            root, self.target_size, self.target_size,
            supersample_factor=self.supersample_factor
        )

        # Trích xuất bản RGB để trực quan hóa
        rgb_image = rgba_image[:, :, :3].copy()

        # Đưa ảnh 4 kênh vào Augmentation
        if self.transform is not None:
            augmented = self.transform(image=rgba_image, mask=mask_binary)
            image_tensor = augmented['image'].float() / 255.0  
            mask_tensor = augmented['mask'].long()     
        else:
            image_tensor = torch.tensor(rgba_image).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.tensor(mask_binary).long()

        return image_tensor, mask_tensor, rgb_image