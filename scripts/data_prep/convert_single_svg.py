#!/usr/bin/env python3
"""
Single Image PNG to SVG Converter with Manual Boolean Cutout
Convert a single PNG image to SVG with true cutout (zero overlap).
"""

import os
import sys
import cv2
import numpy as np
import vtracer
import fal_client
import xml.etree.ElementTree as ET
import re
import argparse
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    import numpy as np
except ImportError:
    print("Installing shapely and numpy...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shapely", "numpy"])
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def enhance_with_fal(img_path, original_alpha=None):
    """
    Sử dụng fal.ai NAFNet model để làm mượt và khử nhiễu ảnh.
    Giữ lại alpha channel gốc nếu có.
    """
    try:
        # Encode image as data URL
        image_data_url = fal_client.encode_file(img_path)
        
        # Call NAFNet deblur model
        result = fal_client.run(
            "fal-ai/nafnet/deblur",
            arguments={
                "image_url": image_data_url
            }
        )
        
        # Download processed image
        import requests
        processed_url = result["image"]["url"]
        response = requests.get(processed_url)
        
        # Convert to numpy array
        import io
        from PIL import Image
        img_pil = Image.open(io.BytesIO(response.content))
        img_array = np.array(img_pil)
        
        # Convert RGB to BGR for OpenCV compatibility
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
        
        # If original has alpha, restore it
        if original_alpha is not None:
            if img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2BGRA)
                img_array[:, :, 3] = original_alpha
        
        return img_array
    except Exception as e:
        print(f"  Warning: Fal.ai enhancement failed: {e}. Using original image.")
        return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


def clean_rgba_image(img):
    """
    Làm sạch kênh Alpha (trong suốt) bằng cách khử nhiễu,
    giữ nguyên format RGBA để vtracer có thể tạo SVG với nền trong suốt.
    Đồng thời làm mượt cả RGB channels để giảm nhọn nhô từ ảnh gốc.
    """
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
    else:
        alpha = None
    
    # Process RGB channels
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            rgb = img[:, :, :3]
        else:
            rgb = img
            alpha = None
        
        # Bilateral filter to smooth while preserving edges
        rgb_smooth = cv2.bilateralFilter(rgb, 9, 75, 75)
    else:
        rgb_smooth = img
    
    # Process alpha channel with threshold (binary)
    if alpha is not None:
        _, alpha_binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        
        # Morphology operations to clean alpha
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_clean = cv2.morphologyEx(alpha_binary, cv2.MORPH_OPEN, kernel)
        alpha_clean = cv2.morphologyEx(alpha_clean, cv2.MORPH_CLOSE, kernel)
        
        # Combine
        result = cv2.cvtColor(rgb_smooth, cv2.COLOR_RGB2RGBA)
        result[:, :, 3] = alpha_clean
    else:
        result = rgb_smooth
    
    return result


class SVGElement:
    """Represents an SVG element with high-precision geometry."""

    def __init__(self, element: ET.Element, document_order: int):
        self.element = element
        self.document_order = document_order  # Later = on top
        self.tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        self.id = element.get("id", f"element_{document_order}")
        self.geometry = None
        self.original_geometry = None
        self.style = self._parse_style()
        self.transform = self._parse_transform()

    def _parse_style(self) -> Dict[str, str]:
        style_dict = {}
        style_str = self.element.get("style", "")
        if style_str:
            for item in style_str.split(";"):
                if ":" in item:
                    key, value = item.split(":", 1)
                    style_dict[key.strip()] = value.strip()

        for attr in ["fill", "stroke", "stroke-width", "opacity"]:
            if attr in self.element.attrib:
                style_dict[attr] = self.element.attrib[attr]

        return style_dict

    def _parse_transform(self) -> Optional[np.ndarray]:
        transform_str = self.element.get("transform", "")
        if not transform_str:
            return None

        try:
            translate_match = re.search(r"translate\(([-\d.,\s]+)\)", transform_str)
            if translate_match:
                values = [
                    float(x) for x in translate_match.group(1).replace(",", " ").split()
                ]
                if len(values) >= 2:
                    return np.array([[1, 0, values[0]], [0, 1, values[1]], [0, 0, 1]])
                elif len(values) == 1:
                    return np.array([[1, 0, values[0]], [0, 1, 0], [0, 0, 1]])

            matrix_match = re.search(r"matrix\(([-\d.,\s]+)\)", transform_str)
            if matrix_match:
                values = [
                    float(x) for x in matrix_match.group(1).replace(",", " ").split()
                ]
                if len(values) == 6:
                    return np.array(
                        [
                            [values[0], values[2], values[4]],
                            [values[1], values[3], values[5]],
                            [0, 0, 1],
                        ]
                    )
        except Exception as e:
            print(f"Warning: Error parsing transform '{transform_str}': {e}")

        return None


class HighPrecisionSVGParser:
    """High-precision SVG parser generating clean geometries."""

    def __init__(self):
        self.high_density_segments = 128

    def parse_file(self, filepath: str) -> Tuple[ET.Element, List[SVGElement]]:
        tree = ET.parse(filepath)
        root = tree.getroot()

        elements = []
        document_order = [0]
        self._extract_elements(root, elements, document_order)

        return root, elements

    def _extract_elements(
        self, parent: ET.Element, elements: List[SVGElement], document_order: List[int]
    ):
        for child in parent:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

            if tag in ["rect", "circle", "ellipse", "polygon", "polyline", "path"]:
                svg_elem = SVGElement(child, document_order[0])
                document_order[0] += 1

                svg_elem.geometry = self._element_to_high_precision_geometry(child)
                svg_elem.original_geometry = svg_elem.geometry

                if svg_elem.geometry and svg_elem.transform is not None:
                    svg_elem.geometry = self._apply_transform(
                        svg_elem.geometry, svg_elem.transform
                    )

                if svg_elem.geometry:
                    svg_elem.geometry = self._clean_polygon(svg_elem.geometry)

                if svg_elem.geometry and not svg_elem.geometry.is_empty:
                    elements.append(svg_elem)

            elif tag in ["g", "svg"]:
                self._extract_elements(child, elements, document_order)

    def _element_to_high_precision_geometry(
        self, element: ET.Element
    ) -> Optional[Polygon]:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        try:
            if tag == "rect":
                x = float(element.get("x", 0))
                y = float(element.get("y", 0))
                w = float(element.get("width", 0))
                h = float(element.get("height", 0))
                return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            elif tag == "circle":
                cx, cy, r = (
                    float(element.get("cx", 0)),
                    float(element.get("cy", 0)),
                    float(element.get("r", 0)),
                )
                angles = np.linspace(
                    0, 2 * np.pi, self.high_density_segments, endpoint=False
                )
                return Polygon(
                    [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
                )
            elif tag == "ellipse":
                cx, cy = float(element.get("cx", 0)), float(element.get("cy", 0))
                rx, ry = float(element.get("rx", 0)), float(element.get("ry", 0))
                angles = np.linspace(
                    0, 2 * np.pi, self.high_density_segments, endpoint=False
                )
                return Polygon(
                    [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in angles]
                )
            elif tag == "polygon":
                points_str = element.get("points", "")
                coords = [
                    tuple(map(float, p.split(",")))
                    for p in re.findall(r"[-\d.]+,[-\d.]+", points_str)
                ]
                return Polygon(coords) if len(coords) >= 3 else None
            elif tag == "path":
                d = element.get("d", "")
                if not d:
                    return None
                coords = self._parse_svg_path_high_precision(d)
                if len(coords) >= 3:
                    poly = Polygon(coords)
                    return poly.buffer(0) if not poly.is_valid else poly
        except Exception as e:
            print(f"Warning: Could not parse {tag}: {e}")
        return None

    def _parse_svg_path_high_precision(self, d: str) -> List[Tuple[float, float]]:
        coords = []
        current_pos = [0.0, 0.0]
        d = re.sub(r"\s+", " ", d.strip())
        commands = re.findall(r"[MmLlHhVvCcSsQqTtAaZz][^MmLlHhVvCcSsQqTtAaZz]*", d)

        for command in commands:
            cmd = command[0]
            params = [
                float(p)
                for p in re.findall(
                    r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", command[1:]
                )
            ]

            if cmd.upper() == "M":
                if len(params) >= 2:
                    current_pos = (
                        [params[0], params[1]]
                        if cmd.isupper()
                        else [current_pos[0] + params[0], current_pos[1] + params[1]]
                    )
                    coords.append(tuple(current_pos))
                    for i in range(2, len(params), 2):
                        if i + 1 < len(params):
                            current_pos = (
                                [params[i], params[i + 1]]
                                if cmd.isupper()
                                else [
                                    current_pos[0] + params[i],
                                    current_pos[1] + params[i + 1],
                                ]
                            )
                            coords.append(tuple(current_pos))
            elif cmd.upper() == "L":
                for i in range(0, len(params), 2):
                    if i + 1 < len(params):
                        current_pos = (
                            [params[i], params[i + 1]]
                            if cmd.isupper()
                            else [
                                current_pos[0] + params[i],
                                current_pos[1] + params[i + 1],
                            ]
                        )
                        coords.append(tuple(current_pos))
            elif cmd.upper() == "H":
                for param in params:
                    current_pos[0] = param if cmd.isupper() else current_pos[0] + param
                    coords.append(tuple(current_pos))
            elif cmd.upper() == "V":
                for param in params:
                    current_pos[1] = param if cmd.isupper() else current_pos[1] + param
                    coords.append(tuple(current_pos))
            elif cmd.upper() == "C":
                for i in range(0, len(params), 6):
                    if i + 5 < len(params):
                        if cmd.isupper():
                            cp1, cp2, ep = (
                                [params[i], params[i + 1]],
                                [params[i + 2], params[i + 3]],
                                [params[i + 4], params[i + 5]],
                            )
                        else:
                            cp1 = [
                                current_pos[0] + params[i],
                                current_pos[1] + params[i + 1],
                            ]
                            cp2 = [
                                current_pos[0] + params[i + 2],
                                current_pos[1] + params[i + 3],
                            ]
                            ep = [
                                current_pos[0] + params[i + 4],
                                current_pos[1] + params[i + 5],
                            ]

                        # Approximating Bezier curves cleanly
                        for step in range(1, 21):
                            t = step / 20.0
                            x = (
                                (1 - t) ** 3 * current_pos[0]
                                + 3 * (1 - t) ** 2 * t * cp1[0]
                                + 3 * (1 - t) * t**2 * cp2[0]
                                + t**3 * ep[0]
                            )
                            y = (
                                (1 - t) ** 3 * current_pos[1]
                                + 3 * (1 - t) ** 2 * t * cp1[1]
                                + 3 * (1 - t) * t**2 * cp2[1]
                                + t**3 * ep[1]
                            )
                            coords.append((x, y))
                        current_pos = ep
            elif cmd.upper() == "Z":
                if coords and coords[-1] != coords[0]:
                    coords.append(coords[0])
        return coords

    def _clean_polygon(self, polygon: Polygon) -> Optional[Polygon]:
        if not polygon or polygon.is_empty:
            return None
        try:
            cleaned = polygon.buffer(1e-9).buffer(
                -1e-9
            )  # Snaps self intersections safely
            if isinstance(cleaned, MultiPolygon):
                return (
                    max(cleaned.geoms, key=lambda p: p.area) if cleaned.geoms else None
                )
            return cleaned if cleaned.area > 1e-7 else None
        except:
            return None

    def _apply_transform(self, geometry: Polygon, transform: np.ndarray) -> Polygon:
        try:

            def tx_coords(coords):
                res = []
                for x, y in coords:
                    p = transform @ np.array([x, y, 1])
                    res.append((p[0], p[1]))
                return res

            ext = tx_coords(geometry.exterior.coords)
            ints = [tx_coords(interior.coords) for interior in geometry.interiors]
            poly = Polygon(ext, ints)
            return poly if poly.is_valid else geometry
        except:
            return geometry


class StrictCutoutProcessor:
    """Processes geometries sequentially from top to bottom to guarantee zero overlap."""

    def __init__(self, area_threshold=10.0):
        self.area_tolerance = 1e-5
        self.snap_buffer = (
            1e-8  # Micro-buffer to cleanly eliminate shared edge overlaps
        )
        self.area_threshold = area_threshold  # Filter polygons smaller than this (px²)

    def create_cutouts(self, elements: List[SVGElement]) -> List[SVGElement]:
        print("  Thực hiện Boolean Cutout (Strict Top-to-Bottom)...")

        # Sort elements: Highest document order (top visible layers) first!
        elements.sort(key=lambda x: x.document_order, reverse=True)

        processed_elements = []
        cumulative_upper_mask = None

        for idx, elem in enumerate(elements):
            if elem.geometry is None or elem.geometry.is_empty:
                continue

            current_geom = elem.geometry

            # Filter by area threshold
            if current_geom.area < self.area_threshold:
                continue

            if cumulative_upper_mask is None:
                # Topmost element stays completely whole (if above threshold)
                elem.geometry = current_geom
                processed_elements.append(elem)
                cumulative_upper_mask = current_geom
            else:
                # Check if upper layers cover this element completely/partially
                if current_geom.intersects(cumulative_upper_mask):
                    try:
                        # Clean execution of shape subtraction: Shape - Cumulative Mask
                        cutout_geom = current_geom.difference(
                            cumulative_upper_mask.buffer(self.snap_buffer)
                        )

                        # Clean micro-slivers from subtraction
                        if not cutout_geom.is_empty:
                            if isinstance(cutout_geom, MultiPolygon):
                                valid_parts = [
                                    p
                                    for p in cutout_geom.geoms
                                    if p.area > self.area_tolerance and p.area >= self.area_threshold
                                ]
                                if valid_parts:
                                    elem.geometry = (
                                        MultiPolygon(valid_parts)
                                        if len(valid_parts) > 1
                                        else valid_parts[0]
                                    )
                                    processed_elements.append(elem)
                            else:
                                if cutout_geom.area > self.area_tolerance and cutout_geom.area >= self.area_threshold:
                                    elem.geometry = cutout_geom
                                    processed_elements.append(elem)
                    except Exception as e:
                        print(
                            f"  Warning: Subtraction error on {elem.id}: {e}. Retaining original."
                        )
                        processed_elements.append(elem)
                else:
                    # No intersection with upper geometries: keep intact
                    processed_elements.append(elem)

                # Merge this layer's original boundary into the running upper mask
                try:
                    cumulative_upper_mask = unary_union(
                        [cumulative_upper_mask, current_geom]
                    ).buffer(0)
                except:
                    cumulative_upper_mask = unary_union(
                        [cumulative_upper_mask, current_geom]
                    )

        # Return elements back to their original drawing order (bottom to top for rendering)
        processed_elements.sort(key=lambda x: x.document_order)
        return processed_elements


def _polygon_to_path(polygon: Polygon) -> str:
    if polygon.is_empty:
        return ""
    coords = list(polygon.exterior.coords)
    if len(coords) < 3:
        return ""

    path_data = f"M {coords[0][0]:.5f},{coords[0][1]:.5f}"
    for coord in coords[1:]:
        path_data += f" L {coord[0]:.5f},{coord[1]:.5f}"
    path_data += " Z"

    for interior in polygon.interiors:
        hole_coords = list(interior.coords)
        if len(hole_coords) >= 3:
            path_data += f" M {hole_coords[0][0]:.5f},{hole_coords[0][1]:.5f}"
            for coord in hole_coords[1:]:
                path_data += f" L {coord[0]:.5f},{coord[1]:.5f}"
            path_data += " Z"
    return path_data


def convert_stack_to_cutout(svg_path: str, area_threshold=10.0):
    """Convert stacked SVG to cutout SVG by removing overlaps."""
    try:
        parser = HighPrecisionSVGParser()
        processor = StrictCutoutProcessor(area_threshold=area_threshold)
        
        root, elements = parser.parse_file(svg_path)
        
        if not elements:
            print("  No path elements found!")
            return False
        
        cutout_elements = processor.create_cutouts(elements)
        
        # Generate output SVG - preserve all original attributes including viewBox
        new_root = ET.Element("svg")
        for attr, value in root.attrib.items():
            new_root.set(attr, value)
        new_root.set("xmlns", "http://www.w3.org/2000/svg")
        
        # Only calculate viewBox if not present in original SVG
        if "viewBox" not in root.attrib:
            all_coords = []
            for element in cutout_elements:
                if element.geometry and not element.geometry.is_empty:
                    if isinstance(element.geometry, Polygon):
                        all_coords.extend(list(element.geometry.exterior.coords))
                    elif isinstance(element.geometry, MultiPolygon):
                        for poly in element.geometry.geoms:
                            all_coords.extend(list(poly.exterior.coords))
            
            if all_coords:
                min_x = min(c[0] for c in all_coords)
                min_y = min(c[1] for c in all_coords)
                max_x = max(c[0] for c in all_coords)
                max_y = max(c[1] for c in all_coords)
                
                # Set viewBox to encompass all paths
                new_root.set("viewBox", f"{min_x} {min_y} {max_x - min_x} {max_y - min_y}")
        
        elements_added = 0
        for element in cutout_elements:
            if element.geometry and not element.geometry.is_empty:
                if isinstance(element.geometry, Polygon):
                    path_data = _polygon_to_path(element.geometry)
                else:
                    path_data = " ".join(
                        [_polygon_to_path(p) for p in element.geometry.geoms]
                    )

                if path_data:
                    path_elem = ET.SubElement(new_root, "path")
                    path_elem.set("d", path_data)
                    path_elem.set("id", f"cutout_{element.id}")
                    for style_attr, style_value in element.style.items():
                        if style_attr in ["fill", "stroke", "stroke-width", "opacity"]:
                            path_elem.set(style_attr, style_value)
                    elements_added += 1
        
        tree = ET.ElementTree(new_root)
        ET.indent(tree, space="  ", level=0)
        tree.write(svg_path, encoding='unicode', xml_declaration=True)
        return True
        
    except Exception as e:
        print(f"Error converting to cutout: {e}")
        return False


def process_single_image(input_path: str, output_path: str = None, use_fal: bool = True):
    """Process a single image through the pipeline."""
    
    # Define folder paths
    dirty_dir = os.path.join(PROJECT_ROOT, "data", "svg", "dirty_png")
    clean_dir = os.path.join(PROJECT_ROOT, "data", "svg", "clean_png")
    svg_dir = os.path.join(PROJECT_ROOT, "data", "svg", "logo")
    
    # Ensure folders exist
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    
    # If input is just filename, prepend dirty_dir
    if not os.path.isabs(input_path) and not os.path.exists(input_path):
        input_path = os.path.join(dirty_dir, input_path)
    
    # Validate input
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # Get filename without extension
    input_file = Path(input_path)
    filename = input_file.stem
    
    # Determine paths
    clean_path = os.path.join(clean_dir, f"{filename}.png")
    
    if output_path is None:
        output_path = os.path.join(svg_dir, f"{filename}.svg")
    elif not os.path.isabs(output_path):
        output_path = os.path.join(svg_dir, output_path)
    
    print(f"Processing: {input_path}")
    print(f"Clean PNG: {clean_path}")
    print(f"Output SVG: {output_path}")
    
    # Load image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Cannot load image: {input_path}")
        return False
    
    # Extract alpha channel if exists
    original_alpha = None
    if len(img.shape) == 3 and img.shape[2] == 4:
        original_alpha = img[:, :, 3].copy()
    
    # Enhance with fal.ai if requested
    if use_fal:
        print("  Đang tăng chất lượng với fal.ai...")
        img = enhance_with_fal(input_path, original_alpha)
    
    # Clean RGBA image
    print("  Đang làm sạch ảnh...")
    clean_img = clean_rgba_image(img)
    
    # Save clean image to clean_png folder
    cv2.imwrite(clean_path, clean_img)
    
    # Vectorize with vtracer (có thể crash với Python 3.14)
    print("  Đang vector hóa...")
    try:
        vtracer.convert_image_to_svg_py(
            clean_path,
            output_path,
            colormode="color",
            hierarchical="stack",
            mode="spline",
            filter_speckle=2,
            color_precision=12,
            layer_difference=4,
            corner_threshold=20,
            length_threshold=4.0,
            max_iterations=25,
            splice_threshold=20,
            path_precision=12
        )
    except Exception as e:
        print(f"Error during vectorization: {e}")
        print("  Bỏ qua bước vectorization, chỉ lưu clean PNG.")
        # Copy clean PNG as fallback
        import shutil
        fallback_path = output_path.replace('.svg', '_fallback.png')
        shutil.copy(clean_path, fallback_path)
        print(f"  Đã lưu fallback: {fallback_path}")
        print("  Gợi ý: Downgrade Python sang 3.11 hoặc 3.12 để vtracer hoạt động ổn định.")
        return False
    
    # Apply manual cutout
    print("  Đang áp dụng Boolean Cutout...")
    convert_stack_to_cutout(output_path, area_threshold=10.0)  # Filter polygons < 10px²
    
    print(f"Hoàn thành!")
    print(f"  Clean PNG: {clean_path}")
    print(f"  SVG: {output_path}")
    return True


if __name__ == "__main__":
    # Hardcoded input - can be filename (looks in dirty_png) or full path
    INPUT_PATH = "50.png"  # Will look in data/svg/dirty_png/1.png
    OUTPUT_PATH = None  # Auto-generate as filename.svg in svg folder
    USE_FAL = True  # Set to False to skip fal.ai enhancement
    
    process_single_image(INPUT_PATH, OUTPUT_PATH, use_fal=USE_FAL)
