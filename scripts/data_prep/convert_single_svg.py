#!/usr/bin/env python3
"""
Single Image PNG to SVG Converter with Manual Boolean Cutout & Auto Labeling
Sử dụng thuật toán Skeletonize + Structural Heuristics
Đã fix lỗi phân biệt "Viền ngoài Satin" và "Chữ O lớn Fill"
"""

import os
import sys
import cv2
import numpy as np
import vtracer
import fal_client
import xml.etree.ElementTree as ET
import re
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Yêu cầu cài đặt: pip install shapely scikit-image
try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
except ImportError:
    print("Installing shapely...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shapely"])
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union

try:
    from skimage.morphology import skeletonize
except ImportError:
    print("Installing scikit-image...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    from skimage.morphology import skeletonize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Đăng ký Namespace cho Inkscape
ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"


def enhance_with_fal(img_path, original_alpha=None):
    try:
        image_data_url = fal_client.encode_file(img_path)
        result = fal_client.run("fal-ai/nafnet/deblur", arguments={"image_url": image_data_url})
        import requests, io
        from PIL import Image
        response = requests.get(result["image"]["url"])
        img_array = np.array(Image.open(io.BytesIO(response.content)))
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
        if original_alpha is not None:
            if img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2BGRA)
            img_array[:, :, 3] = original_alpha
        return img_array
    except Exception as e:
        print(f"  Warning: Fal.ai enhancement failed: {e}. Using original image.")
        return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

def clean_rgba_image(img):
    alpha = img[:, :, 3] if len(img.shape) == 3 and img.shape[2] == 4 else None
    if len(img.shape) == 3:
        rgb = img[:, :, :3] if img.shape[2] == 4 else img
        rgb_smooth = cv2.bilateralFilter(rgb, 9, 75, 75)
    else:
        rgb_smooth = img
    if alpha is not None:
        _, alpha_binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_clean = cv2.morphologyEx(alpha_binary, cv2.MORPH_OPEN, kernel)
        alpha_clean = cv2.morphologyEx(alpha_clean, cv2.MORPH_CLOSE, kernel)
        result = cv2.cvtColor(rgb_smooth, cv2.COLOR_RGB2RGBA)
        result[:, :, 3] = alpha_clean
    else:
        result = rgb_smooth
    return result


class SVGElement:
    def __init__(self, element: ET.Element, document_order: int):
        self.element = element
        self.document_order = document_order
        self.tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        self.id = element.get("id", f"element_{document_order}")
        self.geometry = None
        self.original_geometry = None
        self.style = self._parse_style()
        self.transform = self._parse_transform()
        self.label = "fill"

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
        if not transform_str: return None
        try:
            translate_match = re.search(r"translate\(([-\d.,\s]+)\)", transform_str)
            if translate_match:
                values = [float(x) for x in translate_match.group(1).replace(",", " ").split()]
                if len(values) >= 2: return np.array([[1, 0, values[0]], [0, 1, values[1]], [0, 0, 1]])
            matrix_match = re.search(r"matrix\(([-\d.,\s]+)\)", transform_str)
            if matrix_match:
                values = [float(x) for x in matrix_match.group(1).replace(",", " ").split()]
                if len(values) == 6:
                    return np.array([[values[0], values[2], values[4]], [values[1], values[3], values[5]], [0, 0, 1]])
        except:
            pass
        return None

class HighPrecisionSVGParser:
    def __init__(self):
        self.high_density_segments = 128

    def parse_file(self, filepath: str) -> Tuple[ET.Element, List[SVGElement]]:
        tree = ET.parse(filepath)
        root = tree.getroot()
        elements = []
        self._extract_elements(root, elements, [0])
        return root, elements

    def _extract_elements(self, parent: ET.Element, elements: List[SVGElement], document_order: List[int]):
        for child in parent:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag in ["rect", "circle", "ellipse", "polygon", "polyline", "path"]:
                svg_elem = SVGElement(child, document_order[0])
                document_order[0] += 1
                svg_elem.geometry = self._element_to_high_precision_geometry(child)
                svg_elem.original_geometry = svg_elem.geometry
                
                if svg_elem.geometry and svg_elem.transform is not None:
                    svg_elem.geometry = self._apply_transform(svg_elem.geometry, svg_elem.transform)
                if svg_elem.geometry:
                    svg_elem.geometry = self._clean_polygon(svg_elem.geometry)
                if svg_elem.geometry and not svg_elem.geometry.is_empty:
                    elements.append(svg_elem)
            elif tag in ["g", "svg"]:
                self._extract_elements(child, elements, document_order)

    def _element_to_high_precision_geometry(self, element: ET.Element) -> Optional[Polygon]:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        try:
            if tag == "path":
                d = element.get("d", "")
                if not d: return None
                coords = self._parse_svg_path_high_precision(d)
                if len(coords) >= 3:
                    poly = Polygon(coords)
                    return poly.buffer(0) if not poly.is_valid else poly
        except:
            pass
        return None

    def _parse_svg_path_high_precision(self, d: str) -> List[Tuple[float, float]]:
        coords = []
        current_pos = [0.0, 0.0]
        d = re.sub(r"\s+", " ", d.strip())
        commands = re.findall(r"[MmLlHhVvCcSsQqTtAaZz][^MmLlHhVvCcSsQqTtAaZz]*", d)

        for command in commands:
            cmd = command[0]
            params = [float(p) for p in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", command[1:])]
            if cmd.upper() == "M" or cmd.upper() == "L":
                for i in range(0, len(params), 2):
                    if i + 1 < len(params):
                        current_pos = [params[i], params[i + 1]] if cmd.isupper() else [current_pos[0] + params[i], current_pos[1] + params[i + 1]]
                        coords.append(tuple(current_pos))
            elif cmd.upper() == "C":
                for i in range(0, len(params), 6):
                    if i + 5 < len(params):
                        if cmd.isupper():
                            cp1, cp2, ep = [params[i], params[i + 1]], [params[i + 2], params[i + 3]], [params[i + 4], params[i + 5]]
                        else:
                            cp1 = [current_pos[0] + params[i], current_pos[1] + params[i + 1]]
                            cp2 = [current_pos[0] + params[i + 2], current_pos[1] + params[i + 3]]
                            ep = [current_pos[0] + params[i + 4], current_pos[1] + params[i + 5]]
                        for step in range(1, 21):
                            t = step / 20.0
                            x = ((1 - t)**3 * current_pos[0] + 3 * (1 - t)**2 * t * cp1[0] + 3 * (1 - t) * t**2 * cp2[0] + t**3 * ep[0])
                            y = ((1 - t)**3 * current_pos[1] + 3 * (1 - t)**2 * t * cp1[1] + 3 * (1 - t) * t**2 * cp2[1] + t**3 * ep[1])
                            coords.append((x, y))
                        current_pos = ep
            elif cmd.upper() == "Z":
                if coords and coords[-1] != coords[0]:
                    coords.append(coords[0])
        return coords

    def _clean_polygon(self, polygon: Polygon) -> Optional[Polygon]:
        if not polygon or polygon.is_empty: return None
        try:
            cleaned = polygon.buffer(1e-9).buffer(-1e-9)
            if isinstance(cleaned, MultiPolygon): return max(cleaned.geoms, key=lambda p: p.area) if cleaned.geoms else None
            return cleaned if cleaned.area > 1e-7 else None
        except: return None

    def _apply_transform(self, geometry: Polygon, transform: np.ndarray) -> Polygon:
        try:
            def tx_coords(coords): return [( (transform @ np.array([x, y, 1]))[0], (transform @ np.array([x, y, 1]))[1] ) for x, y in coords]
            return Polygon(tx_coords(geometry.exterior.coords), [tx_coords(interior.coords) for interior in geometry.interiors])
        except: return geometry

class StrictCutoutProcessor:
    def __init__(self, area_threshold=10.0):
        self.area_tolerance = 1e-5
        self.snap_buffer = 1e-8 
        self.area_threshold = area_threshold

    def create_cutouts(self, elements: List[SVGElement]) -> List[SVGElement]:
        print("  Thực hiện Boolean Cutout (Strict Top-to-Bottom)...")
        elements.sort(key=lambda x: x.document_order, reverse=True)
        processed_elements = []
        cumulative_upper_mask = None

        for elem in elements:
            if elem.geometry is None or elem.geometry.is_empty or elem.geometry.area < self.area_threshold:
                continue

            current_geom = elem.geometry
            if cumulative_upper_mask is None:
                elem.geometry = current_geom
                processed_elements.append(elem)
                cumulative_upper_mask = current_geom
            else:
                if current_geom.intersects(cumulative_upper_mask):
                    try:
                        cutout_geom = current_geom.difference(cumulative_upper_mask.buffer(self.snap_buffer))
                        if not cutout_geom.is_empty:
                            if isinstance(cutout_geom, MultiPolygon):
                                valid_parts = [p for p in cutout_geom.geoms if p.area > self.area_tolerance and p.area >= self.area_threshold]
                                if valid_parts:
                                    elem.geometry = MultiPolygon(valid_parts) if len(valid_parts) > 1 else valid_parts[0]
                                    processed_elements.append(elem)
                            elif cutout_geom.area > self.area_tolerance and cutout_geom.area >= self.area_threshold:
                                elem.geometry = cutout_geom
                                processed_elements.append(elem)
                    except:
                        processed_elements.append(elem)
                else:
                    processed_elements.append(elem)

                try:
                    cumulative_upper_mask = unary_union([cumulative_upper_mask, current_geom]).buffer(0)
                except:
                    pass

        processed_elements.sort(key=lambda x: x.document_order)
        return processed_elements


# ==========================================
# THUẬT TOÁN ĐA LUẬT ĐÃ FIX LỖI (OUTER BORDER vs BIG 'O')
# ==========================================
def auto_label_elements_advanced(
    elements: List[SVGElement], 
    canvas_w: float, 
    canvas_h: float, 
    physical_width_mm: float, 
    threshold_mm: float
):
    print(f"  Tự động gán nhãn AI (Heuristics + Skeleton) | Kích thước: {physical_width_mm}mm...")
    
    pixel_to_mm = physical_width_mm / max(canvas_w, 1.0)
    canvas_area = canvas_w * canvas_h
    
    mask_h, mask_w = int(canvas_h), int(canvas_w)
    if mask_h <= 0 or mask_w <= 0:
        mask_h, mask_w = 768, 768
        
    satin_count = 0
    fill_count = 0

    # --- BƯỚC 0: TÌM VÀ ĐỊNH VỊ OUTER BORDER THỰC SỰ ---
    # Thay vì mảng rỗng nào cũng là viền, chỉ duy nhất mảng bao trùm lớn nhất mới là viền.
    max_hull_area = 0
    outer_border_id = None
    for elem in elements:
        if elem.geometry and not elem.geometry.is_empty:
            ha = elem.geometry.convex_hull.area
            if ha > max_hull_area:
                max_hull_area = ha
                outer_border_id = elem.id

    for elem in elements:
        if elem.geometry is None or elem.geometry.is_empty:
            elem.label = "fill"
            continue
            
        geom = elem.geometry
        
        # 1. Các chỉ số hình học
        area_px = geom.area
        hull_area_px = geom.convex_hull.area
        solidity = area_px / hull_area_px if hull_area_px > 0 else 1.0
        
        min_x, min_y, max_x, max_y = geom.bounds
        w, h = max_x - min_x, max_y - min_y
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 100.0
        
        is_hollow = False
        if isinstance(geom, Polygon):
            is_hollow = len(geom.interiors) > 0
        elif isinstance(geom, MultiPolygon):
            is_hollow = any(len(p.interiors) > 0 for p in geom.geoms)

        # 2. Skeleton Median Thickness
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        
        def draw_polygon(poly, mask_arr):
            ext_coords = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask_arr, [ext_coords], 255)
            for interior in poly.interiors:
                int_coords = np.array(interior.coords, dtype=np.int32)
                cv2.fillPoly(mask_arr, [int_coords], 0)

        if isinstance(geom, Polygon):
            draw_polygon(geom, mask)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                draw_polygon(poly, mask)
                
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        bool_mask = mask > 0
        
        thickness_mm = 0.0
        try:
            skeleton = skeletonize(bool_mask)
            skeleton_coords = np.where(skeleton)
            skeleton_distances = dist[skeleton_coords]
            
            if len(skeleton_distances) > 0:
                median_dist = np.median(skeleton_distances)
                thickness_mm = (median_dist * 2.0) * pixel_to_mm
            else:
                thickness_mm = (np.max(dist) * 2.0) * pixel_to_mm
        except:
            thickness_mm = (np.max(dist) * 2.0) * pixel_to_mm

        # --- 3. BỘ LUẬT PHÂN LOẠI (RULE TREE) ---
        
        # Cờ đánh dấu: Có phải là mảng lớn nhất bao trùm cả logo không?
        is_outermost = (elem.id == outer_border_id) and ((hull_area_px / canvas_area) > 0.4)
        
        # Rule 1: Outer Border THỰC SỰ
        # Nếu là mảng ngoài cùng, có lỗ rỗng, và độ dày KHÔNG LỚN HƠN 8.0mm (Giới hạn viền vá lớn nhất)
        if is_outermost and is_hollow and thickness_mm <= 8.0:
            elem.label = "satin"
            
        # Rule 2: Các chữ O, hình tròn rỗng bên trong (Hoặc viền ngoài nhưng siêu khổng lồ)
        elif is_hollow:
            if thickness_mm >= threshold_mm:
                elem.label = "fill"
            else:
                elem.label = "satin"
                
        # Rule 3: Chữ cái, nét ngoằn ngoèo không có lỗ rỗng (Ví dụ chữ S, C, M...)
        # Solidity thấp chứng tỏ nó không phải là mảng đặc
        elif solidity < 0.65 and thickness_mm < (threshold_mm * 1.5):
            elem.label = "satin"
            
        # Rule 4: Dải nét dài và hẹp
        # Dù hơi dày (ví dụ 3-4mm) nhưng kéo dài (Aspect Ratio > 4.0) -> Satin
        elif aspect_ratio > 4.0 and thickness_mm <= 4.0:
            elem.label = "satin"
            
        # Rule 5: Khối đặc lớn (Compact Solid Blob)
        elif solidity > 0.85 and thickness_mm >= threshold_mm:
            elem.label = "fill"
            
        # Fallback dựa vào Median Thickness
        else:
            if thickness_mm < threshold_mm:
                elem.label = "satin"
            else:
                elem.label = "fill"
                
        if elem.label == "satin": satin_count += 1
        else: fill_count += 1
        
    print(f"    -> Đã gán {satin_count} Satin, {fill_count} Fill.")


def _polygon_to_path(polygon: Polygon) -> str:
    if polygon.is_empty: return ""
    coords = list(polygon.exterior.coords)
    if len(coords) < 3: return ""
    path_data = f"M {coords[0][0]:.5f},{coords[0][1]:.5f}"
    for coord in coords[1:]: path_data += f" L {coord[0]:.5f},{coord[1]:.5f}"
    path_data += " Z"
    for interior in polygon.interiors:
        hole_coords = list(interior.coords)
        if len(hole_coords) >= 3:
            path_data += f" M {hole_coords[0][0]:.5f},{hole_coords[0][1]:.5f}"
            for coord in hole_coords[1:]: path_data += f" L {coord[0]:.5f},{coord[1]:.5f}"
            path_data += " Z"
    return path_data


def convert_stack_to_cutout(svg_path: str, area_threshold=10.0, physical_width_mm=80.0, threshold_mm=2.0):
    try:
        parser = HighPrecisionSVGParser()
        processor = StrictCutoutProcessor(area_threshold=area_threshold)
        
        root, elements = parser.parse_file(svg_path)
        if not elements: return False
        
        cutout_elements = processor.create_cutouts(elements)
        
        canvas_w, canvas_h = 768.0, 768.0 
        if "viewBox" in root.attrib:
            vb = root.attrib["viewBox"].split()
            if len(vb) == 4:
                canvas_w, canvas_h = float(vb[2]), float(vb[3])
        elif "width" in root.attrib and "height" in root.attrib:
            canvas_w = float(re.sub(r'[^\d.]', '', root.attrib["width"]))
            canvas_h = float(re.sub(r'[^\d.]', '', root.attrib["height"]))
            
        auto_label_elements_advanced(
            cutout_elements, 
            canvas_w=canvas_w, 
            canvas_h=canvas_h, 
            physical_width_mm=physical_width_mm, 
            threshold_mm=threshold_mm
        )
        
        new_root = ET.Element("svg")
        for attr, value in root.attrib.items():
            new_root.set(attr, value)
        new_root.set("xmlns", "http://www.w3.org/2000/svg")
        new_root.set(f"{{{INKSCAPE_NS}}}version", "1.0")
        
        for element in cutout_elements:
            if element.geometry and not element.geometry.is_empty:
                path_data = _polygon_to_path(element.geometry) if isinstance(element.geometry, Polygon) else " ".join([_polygon_to_path(p) for p in element.geometry.geoms])
                if path_data:
                    path_elem = ET.SubElement(new_root, "path")
                    path_elem.set("d", path_data)
                    path_elem.set("id", f"cutout_{element.id}")
                    
                    path_elem.set(f"{{{INKSCAPE_NS}}}label", element.label)
                    
                    for style_attr, style_value in element.style.items():
                        if style_attr in ["fill", "stroke", "stroke-width", "opacity"]:
                            path_elem.set(style_attr, style_value)
                            
        tree = ET.ElementTree(new_root)
        ET.indent(tree, space="  ", level=0)
        tree.write(svg_path, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        print(f"Error converting to cutout: {e}")
        return False


def process_single_image(input_path: str, output_path: str = None, use_fal: bool = True):
    dirty_dir = os.path.join(PROJECT_ROOT, "data", "svg", "dirty_png")
    clean_dir = os.path.join(PROJECT_ROOT, "data", "svg", "clean_png")
    svg_dir = os.path.join(PROJECT_ROOT, "data", "svg", "logo")
    
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    
    if not os.path.isabs(input_path) and not os.path.exists(input_path):
        input_path = os.path.join(dirty_dir, input_path)
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
        
    filename = Path(input_path).stem
    clean_path = os.path.join(clean_dir, f"{filename}.png")
    output_path = os.path.join(svg_dir, f"{filename}.svg") if output_path is None else (output_path if os.path.isabs(output_path) else os.path.join(svg_dir, output_path))
    
    print(f"\nProcessing: {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None: return False
    
    original_alpha = img[:, :, 3].copy() if len(img.shape) == 3 and img.shape[2] == 4 else None
    
    if use_fal:
        print("  Đang tăng chất lượng với fal.ai...")
        img = enhance_with_fal(input_path, original_alpha)
        
    print("  Đang làm sạch ảnh...")
    cv2.imwrite(clean_path, clean_rgba_image(img))
    
    print("  Đang vector hóa...")
    try:
        vtracer.convert_image_to_svg_py(
            clean_path, output_path, colormode="color", hierarchical="stack", mode="spline",
            filter_speckle=2, color_precision=12, layer_difference=4, corner_threshold=20,
            length_threshold=4.0, max_iterations=25, splice_threshold=20, path_precision=12
        )
    except Exception as e:
        print(f"Error during vectorization: {e}. Bỏ qua bước vectorization.")
        return False
    
    LOGO_PHYSICAL_WIDTH = 80.0 
    THICKNESS_THRESHOLD = 2.0  
    
    convert_stack_to_cutout(
        output_path, 
        area_threshold=10.0, 
        physical_width_mm=LOGO_PHYSICAL_WIDTH, 
        threshold_mm=THICKNESS_THRESHOLD
    )
    
    print(f"Hoàn thành xuất sắc!")
    print(f"  SVG File: {output_path}")
    return True


if __name__ == "__main__":
    INPUT_PATH = "133.png"
    OUTPUT_PATH = None
    USE_FAL = True
    
    process_single_image(INPUT_PATH, OUTPUT_PATH, use_fal=USE_FAL)