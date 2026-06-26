import os
import glob
import cv2
import numpy as np
import vtracer
import fal_client
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re
from typing import List, Tuple, Dict, Optional, Union

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    import numpy as np
except ImportError:
    print("Installing shapely and numpy...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shapely", "numpy"])
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    import numpy as np

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
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Nếu có alpha channel gốc, dán lại vào ảnh đã xử lý
        if original_alpha is not None:
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.merge([img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2], original_alpha])
        
        return img_array
    except Exception as e:
        print(f"Lỗi khi xử lý với fal.ai: {e}")
        return None

def clean_rgba_image(img):
    """
    Làm sạch kênh Alpha (trong suốt) bằng cách khử nhiễu,
    giữ nguyên format RGBA để vtracer có thể tạo SVG với nền trong suốt.
    Đồng thời làm mượt cả RGB channels để giảm nhọn nhô từ ảnh gốc.
    """
    if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        alpha = img[:, :, 3]
        rgb = img[:, :, :3]
        
        # Làm mượt RGB channels bằng bilateral filter để giữ cạnh nhưng giảm nhiễu
        rgb_smooth = cv2.bilateralFilter(rgb, 9, 75, 75)
        
        # Làm SẮC LẸM alpha channel bằng Threshold thay vì Blur
        # vtracer cần alpha binary (0 hoặc 255) để tránh tạo hàng ngàn vector path
        _, alpha_binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        
        # Khử nhiễu alpha channel bằng morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_clean = cv2.morphologyEx(alpha_binary, cv2.MORPH_OPEN, kernel)
        alpha_clean = cv2.morphologyEx(alpha_clean, cv2.MORPH_CLOSE, kernel)
        
        # Cập nhật alpha channel đã làm sạch và RGB đã làm mượt
        result = np.zeros_like(img)
        result[:, :, :3] = rgb_smooth
        result[:, :, 3] = alpha_clean
        
        return result
    
    # Nếu ảnh không có alpha, thêm alpha channel (đen = trong suốt)
    if len(img.shape) == 3 and img.shape[2] == 3:  # BGR
        # Làm mượt RGB channels
        rgb_smooth = cv2.bilateralFilter(img, 9, 75, 75)
        alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        result = cv2.merge([rgb_smooth[:, :, 0], rgb_smooth[:, :, 1], rgb_smooth[:, :, 2], alpha])
        return result
    
    return img

def merge_svg_paths(svg_path):
    """
    Merge các path có đường biên tương tự trong SVG để chúng share border.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Tìm tất cả path elements
        paths = root.findall('.//{http://www.w3.org/2000/svg}path')
        
        # Group paths by color
        paths_by_color = {}
        for path in paths:
            fill = path.get('fill', 'none')
            stroke = path.get('stroke', 'none')
            key = (fill, stroke)
            if key not in paths_by_color:
                paths_by_color[key] = []
            paths_by_color[key].append(path)
        
        # Save modified SVG
        tree.write(svg_path, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        print(f"Lỗi khi merge SVG paths: {e}")
        return False


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

    def __init__(self):
        self.area_tolerance = 1e-5
        self.snap_buffer = (
            1e-8  # Micro-buffer to cleanly eliminate shared edge overlaps
        )

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

            if cumulative_upper_mask is None:
                # Topmost element stays completely whole
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
                                    if p.area > self.area_tolerance
                                ]
                                if valid_parts:
                                    elem.geometry = (
                                        MultiPolygon(valid_parts)
                                        if len(valid_parts) > 1
                                        else valid_parts[0]
                                    )
                                    processed_elements.append(elem)
                            else:
                                if cutout_geom.area > self.area_tolerance:
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


def convert_stack_to_cutout(svg_path: str):
    """Convert stacked SVG to cutout SVG by removing overlaps."""
    try:
        parser = HighPrecisionSVGParser()
        processor = StrictCutoutProcessor()
        
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
                path_data = _geometry_to_svg_path(element.geometry)
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


def _geometry_to_svg_path(geometry: Union[Polygon, MultiPolygon]) -> str:
    """Convert Shapely geometry to SVG path."""
    if isinstance(geometry, Polygon):
        return _polygon_to_path(geometry)
    elif isinstance(geometry, MultiPolygon):
        paths = [_polygon_to_path(poly) for poly in geometry.geoms]
        return " ".join(paths)
    return ""


def _polygon_to_path(polygon: Polygon) -> str:
    """Convert Polygon to SVG path."""
    if polygon.is_empty:
        return ""
    
    coords = list(polygon.exterior.coords)
    if len(coords) < 3:
        return ""
    
    path_data = f"M {coords[0][0]:.6f},{coords[0][1]:.6f}"
    for coord in coords[1:]:
        path_data += f" L {coord[0]:.6f},{coord[1]:.6f}"
    path_data += " Z"
    
    for interior in polygon.interiors:
        hole_coords = list(interior.coords)
        if len(hole_coords) >= 3:
            path_data += f" M {hole_coords[0][0]:.6f},{hole_coords[0][1]:.6f}"
            for coord in hole_coords[1:]:
                path_data += f" L {coord[0]:.6f},{coord[1]:.6f}"
            path_data += " Z"
    
    return path_data

def process_pipeline(input_dir, clean_dir, svg_dir):
    """
    Pipeline: Đọc PNG -> Làm sạch -> Lưu PNG tạm -> Vector hóa (Color Mode)
    """
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    # Quét tất cả các định dạng ảnh phổ biến
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG'): 
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong thư mục: {input_dir}")
        return

    print(f"Bắt đầu xử lý {len(image_paths)} ảnh...")

    for img_path in tqdm(image_paths, desc="Vector hóa"):
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)

        clean_path = os.path.join(clean_dir, f"{name}.png")
        svg_path = os.path.join(svg_dir, f"{name}.svg")

        # 1. Đọc ảnh & Làm sạch
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        
        # Trích xuất alpha channel gốc trước khi xử lý với fal.ai
        original_alpha = None
        if len(img.shape) == 3 and img.shape[2] == 4:
            original_alpha = img[:, :, 3].copy()
        
        # 1.1 Tăng chất lượng ảnh với fal.ai NAFNet
        enhanced_img = enhance_with_fal(img_path, original_alpha)
        if enhanced_img is not None:
            img = enhanced_img
            print(f"  Đã tăng chất lượng với fal.ai: {filename}")
        
        clean_img = clean_rgba_image(img)
        cv2.imwrite(clean_path, clean_img)

        # 2. Vector hóa (Chế độ color để giữ màu logo và transparency)
        vtracer.convert_image_to_svg_py(
            clean_path,
            svg_path,
            colormode="color",        # Giữ nguyên màu
            hierarchical="stack",   # Tách vùng màu riêng biệt thay vì xếp chồng
            mode="spline",            # Đường cong Bezier mượt
            filter_speckle=2,         # Giảm ngưỡng để giữ nhiều chi tiết màu hơn
            color_precision=12,       # Tăng độ chính xác màu để giảm quantization
            layer_difference=4,       # Giảm thêm để gộp các vùng màu tương tự lại với nhau
            corner_threshold=20,      # Giảm ngưỡng góc để đường cong mượt hơn
            length_threshold=4.0,     # Tăng ngưỡng độ dài đường cong để loại bỏ các đoạn ngắn
            max_iterations=25,        # Tăng iterations để hội tụ tốt hơn
            splice_threshold=20,      # Giảm ngưỡng nối đường để đường cong mượt hơn
            path_precision=12         # Tăng precision đường cong
        )
        
        # 3. Convert stacked SVG to cutout (remove overlaps using boolean operations)
        convert_stack_to_cutout(svg_path)

    print(f"\nHoàn thành! Các SVG đã được đục lỗ (True Cutout) nằm tại: {svg_dir}")

if __name__ == "__main__":
    # Lấy thư mục hiện tại là data_prep/
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Đi ngược lên 1 cấp (..) rồi vào thư mục data/
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 
    
    # Trỏ đến đúng các thư mục bạn mong muốn
    INPUT_DIR = os.path.join(PROJECT_ROOT, "data_prep", "data", "dirty_png")
    CLEAN_DIR = os.path.join(PROJECT_ROOT, "data_prep", "data", "clean_png")
    SVG_DIR = os.path.join(PROJECT_ROOT, "data_prep", "data", "svg")

    print(f"DEBUG: PATH INPUT = {INPUT_DIR}")
    
    process_pipeline(INPUT_DIR, CLEAN_DIR, SVG_DIR)