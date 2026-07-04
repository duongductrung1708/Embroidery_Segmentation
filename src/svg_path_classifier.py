#!/usr/bin/env python3
"""
SVG Path Classifier for Embroidery Segmentation
================================================

Phân loại tự động path SVG thành:
- Background (0)
- Fill (1)
- Satin (2)

Chỉ dựa trên hình học (Đã chuẩn hóa 100% về Millimet - mm).

QUY TẮC CỨNG (dùng để sinh nhãn train tự động):
1. Path bám theo viền ngoài cùng của toàn bộ logo (dày <= 8mm) -> SATIN
2. Path dạng khung/viền mỏng (ring, có lỗ ở giữa, solidity thấp) -> SATIN
3. Width <= 2.0mm -> SATIN
4. Width > 2.0mm -> FILL
"""

import re
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cairosvg
import cv2
import numpy as np
import svgpathtools
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage.morphology import skeletonize

# ---------------------------------------------------------------------------
# Constants (Hệ Millimet - mm)
# ---------------------------------------------------------------------------
LABEL_BACKGROUND = 0
LABEL_FILL = 1
LABEL_SATIN = 2

# Nếu SVG không có đơn vị thực, giả sử logo rộng 80mm (8cm)
DEFAULT_REAL_WIDTH_MM = 80.0

# Ngưỡng phân loại theo độ rộng
SATIN_WIDTH_THRESHOLD_MM = 2.0

# Giới hạn độ dày tối đa cho viền ngoài cùng để được gọi là SATIN
OUTER_BORDER_MAX_THICKNESS_MM = 8.0

# Path được coi là "khung/viền" (ring, donut) nếu độ đặc thấp hơn ngưỡng này
RING_SOLIDITY_THRESHOLD = 0.35

# Path được coi là bám viền ngoài nếu >= tỉ lệ này diện tích của nó nằm trong dải biên ngoài
OUTER_BAND_OVERLAP_THRESHOLD = 0.6

# Độ dày dải biên ngoài, tính theo % kích thước lớn nhất (bbox) của logo
OUTER_BAND_THICKNESS_RATIO = 0.015


class SVGPath:
    """Container cho thông tin 1 path SVG."""

    def __init__(self, path_id: str, element: ET.Element, path_data: str):
        self.id = path_id
        self.element = element
        self.path_data = path_data
        self.label: Optional[str] = None          # 'SATIN' hoặc 'FILL'
        self.width_mm: Optional[float] = None     # Đã đổi thành mm
        self.is_outer_boundary: bool = False
        self.is_ring_shape: bool = False
        self.geometry = None                      


# ---------------------------------------------------------------------------
# Load SVG & quy đổi đơn vị về mm
# ---------------------------------------------------------------------------
def load_svg(svg_path: str) -> Tuple[ET.Element, Dict[str, SVGPath], float]:
    """Load SVG, trả về (root, dict path_id -> SVGPath, pixel_per_mm)."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    pixel_per_mm = calculate_pixel_per_mm(root)

    paths: Dict[str, SVGPath] = {}
    for idx, child in enumerate(root.iter()):
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            path_id = child.get("id", f"path_{idx}")
            path_data = child.get("d", "")
            if path_data:
                paths[path_id] = SVGPath(path_id, child, path_data)

    return root, paths, pixel_per_mm


def calculate_pixel_per_mm(root: ET.Element) -> float:
    """Tính pixel/mm từ viewBox + width (nếu có đơn vị thực)."""
    viewbox = root.get("viewBox")
    if viewbox:
        try:
            parts = viewbox.split()
            vb_width = float(parts[2]) if len(parts) == 4 else 512.0
        except (ValueError, IndexError):
            vb_width = 512.0
    else:
        vb_width = 512.0

    width_str = root.get("width", str(vb_width))
    value, unit = parse_dimension(width_str)

    if unit:
        width_mm = convert_to_mm(value, unit)
        return vb_width / width_mm

    # Không có đơn vị thực -> giả sử logo rộng 80mm
    return vb_width / DEFAULT_REAL_WIDTH_MM


def parse_dimension(dimension_str: str) -> Tuple[float, Optional[str]]:
    """Tách 1 chuỗi kích thước ('100px', '10cm'...) thành (giá trị, đơn vị)."""
    dimension_str = dimension_str.strip()
    match = re.match(r'^([\d.]+)\s*([a-zA-Z]*)$', dimension_str)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower() if match.group(2) else None
        return value, unit
    match = re.match(r'^([\d.]+)', dimension_str)
    if match:
        return float(match.group(1)), None
    return 100.0, None


def convert_to_mm(value: float, unit: str) -> float:
    """Quy đổi giá trị thẳng sang mm."""
    unit = unit.lower()
    conversions = {
        'cm': 10.0,
        'mm': 1.0,
        'm': 1000.0,
        'in': 25.4,
        'ft': 304.8,
        'px': 0.264583333,  # 96 DPI
        'pt': 0.352777778,
        'pc': 4.233333333,
    }
    return value * conversions.get(unit, 1.0)


def get_svg_dimensions(svg_path: str) -> Tuple[int, int]:
    """Lấy width/height (pixel) của SVG."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    viewbox = root.get("viewBox")
    if viewbox:
        try:
            parts = viewbox.split()
            if len(parts) == 4:
                return int(float(parts[2])), int(float(parts[3]))
        except (ValueError, IndexError):
            pass

    width_str = root.get("width")
    height_str = root.get("height")
    if width_str and height_str:
        wm = re.search(r'([\d.]+)', width_str)
        hm = re.search(r'([\d.]+)', height_str)
        if wm and hm:
            return int(float(wm.group(1))), int(float(hm.group(1)))

    return 512, 512


# ---------------------------------------------------------------------------
# Hình học: dựng polygon chính xác cho path có nhiều subpath / có lỗ
# ---------------------------------------------------------------------------
def _sample_subpaths(path_data: str, num_samples: int = 80) -> List[List[Tuple[float, float]]]:
    try:
        path = svgpathtools.parse_path(path_data)
    except Exception as e:
        print(f"Loi parse path: {e}")
        return []

    try:
        subpaths = path.continuous_subpaths()
    except Exception:
        subpaths = [path]

    point_lists: List[List[Tuple[float, float]]] = []
    for sp in subpaths:
        pts: List[Tuple[float, float]] = []
        for seg in sp:
            for t in np.linspace(0, 1, num_samples, endpoint=False):
                p = seg.point(t)
                pts.append((p.real, p.imag))
        if len(pts) >= 3:
            point_lists.append(pts)
    return point_lists


def render_path_to_polygon(path_element: ET.Element) -> Optional[MultiPolygon]:
    data = path_element.get("d", "")
    if not data:
        return None

    raw_polys = []
    for pts in _sample_subpaths(data):
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        raw_polys.append(poly)

    if not raw_polys:
        return None

    if len(raw_polys) == 1:
        p = raw_polys[0]
        if p.geom_type == "Polygon":
            return MultiPolygon([p])
        if p.geom_type == "MultiPolygon":
            return p
        return None

    n = len(raw_polys)
    depth = [0] * n
    parent = [-1] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                inside = raw_polys[i].contains(raw_polys[j].representative_point())
            except Exception:
                inside = False
            if inside:
                depth[j] += 1
                if parent[j] == -1 or raw_polys[i].area < raw_polys[parent[j]].area:
                    parent[j] = i

    exteriors_idx = [i for i in range(n) if depth[i] % 2 == 0]
    result_polys = []

    for ext_i in exteriors_idx:
        ext_poly = raw_polys[ext_i]
        if ext_poly.geom_type != "Polygon":
            if not ext_poly.is_empty:
                result_polys.append(ext_poly)
            continue

        holes = [
            list(raw_polys[j].exterior.coords)
            for j in range(n)
            if depth[j] % 2 == 1 and parent[j] == ext_i and raw_polys[j].geom_type == "Polygon"
        ]
        try:
            poly = Polygon(ext_poly.exterior.coords, holes)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                result_polys.append(poly)
        except Exception:
            result_polys.append(ext_poly)

    if not result_polys:
        return None

    flat = []
    for p in result_polys:
        if p.geom_type == "Polygon":
            flat.append(p)
        elif p.geom_type == "MultiPolygon":
            flat.extend(list(p.geoms))
    return MultiPolygon(flat) if flat else None


def _iter_polygons(geom):
    if geom is None:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms


# ---------------------------------------------------------------------------
# Rule 1: phát hiện path bám viền ngoài cùng của logo
# ---------------------------------------------------------------------------
def find_outer_boundary_paths(paths: Dict[str, SVGPath], svg_width: int, svg_height: int) -> None:
    geoms = {}
    for pid, p in paths.items():
        g = render_path_to_polygon(p.element)
        if g is not None and not g.is_empty:
            p.geometry = g
            geoms[pid] = g

    if not geoms:
        return

    silhouette = unary_union(list(geoms.values()))
    if silhouette.is_empty:
        return

    minx, miny, maxx, maxy = silhouette.bounds
    max_dim = max(maxx - minx, maxy - miny)
    if max_dim <= 0:
        return

    band_thickness = max(1.0, max_dim * OUTER_BAND_THICKNESS_RATIO)
    eroded = silhouette.buffer(-band_thickness)
    outer_band = silhouette.difference(eroded) if not eroded.is_empty else silhouette

    for pid, g in geoms.items():
        if g.area <= 0:
            continue
        overlap_ratio = g.intersection(outer_band).area / g.area
        if overlap_ratio >= OUTER_BAND_OVERLAP_THRESHOLD:
            paths[pid].is_outer_boundary = True


# ---------------------------------------------------------------------------
# Rule 2: phát hiện path dạng khung/viền mỏng (ring) bao quanh path khác
# ---------------------------------------------------------------------------
def compute_ring_shapes(paths: Dict[str, SVGPath]) -> None:
    for p in paths.values():
        g = p.geometry
        if g is None or g.is_empty:
            continue

        has_hole = any(len(poly.interiors) > 0 for poly in _iter_polygons(g))
        if not has_hole:
            continue

        hull_area = g.convex_hull.area
        if hull_area <= 0:
            continue

        solidity = g.area / hull_area
        if solidity <= RING_SOLIDITY_THRESHOLD:
            p.is_ring_shape = True


# ---------------------------------------------------------------------------
# Đo độ rộng vật lý của path ra Millimet
# ---------------------------------------------------------------------------
def render_path_to_mask(path_element: ET.Element, svg_width: int, svg_height: int,
                         supersample_factor: int = 1) -> Optional[np.ndarray]:
    try:
        data = path_element.get("d", "")
        if not data:
            return None

        svg_template = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {svg_width} {svg_height}" '
            f'width="{svg_width}" height="{svg_height}">'
            f'<path d="{data}" fill="black" stroke="none"/></svg>'
        )

        render_width = svg_width * supersample_factor
        render_height = svg_height * supersample_factor

        png_bytes = cairosvg.svg2png(
            bytestring=svg_template.encode('utf-8'),
            output_width=render_width,
            output_height=render_height,
            background_color='white',
            unsafe=True,
        )

        img = Image.open(BytesIO(png_bytes))
        if img.mode != 'L':
            img = img.convert('L')
        mask = np.array(img)

        return (mask < 128).astype(np.uint8)

    except Exception as e:
        print(f"Error rendering path to mask: {e}")
        return None


def estimate_path_width_mm(path_obj: SVGPath, svg_width: int, svg_height: int,
                            pixel_per_mm: float, supersample_factor: int = 2) -> float:
    try:
        mask = render_path_to_mask(path_obj.element, svg_width, svg_height, supersample_factor)
        if mask is None or mask.sum() == 0:
            return 0.0

        distance_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        skeleton = skeletonize(mask.astype(bool))

        if skeleton.sum() == 0:
            width_pixels = 2 * distance_transform.max()
        else:
            skeleton_distances = distance_transform[skeleton]
            if len(skeleton_distances) > 0:
                width_pixels = 2 * np.percentile(skeleton_distances, 75)
            else:
                width_pixels = 2 * distance_transform.max()

        width_mm = width_pixels / (supersample_factor * pixel_per_mm)
        return max(0.0, width_mm)

    except Exception as e:
        print(f"Error estimating path width for {path_obj.id}: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Rule cứng tổng hợp (Theo chuẩn mm)
# ---------------------------------------------------------------------------
def classify_path(path_obj: SVGPath, width_mm: float) -> str:
    # Rule 1: Viền ngoài cùng
    if path_obj.is_outer_boundary:
        if width_mm <= OUTER_BORDER_MAX_THICKNESS_MM:
            return 'SATIN'
        return 'FILL'  # Nếu viền ngoài cùng mà quá to (ví dụ mảng nền lớn) thì ép về FILL
        
    # Rule 2: Khung rỗng
    if path_obj.is_ring_shape:
        return 'SATIN'
        
    # Rule 3: Độ dày
    if width_mm <= SATIN_WIDTH_THRESHOLD_MM:
        return 'SATIN'
        
    return 'FILL'


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------
def render_mask(svg_path: str, output_width: int, output_height: int,
                 supersample_factor: int = 2) -> np.ndarray:
    root, paths, pixel_per_mm = load_svg(svg_path)

    if not paths:
        return np.zeros((output_height, output_width), dtype=np.uint8)

    svg_width, svg_height = get_svg_dimensions(svg_path)

    find_outer_boundary_paths(paths, svg_width, svg_height)
    compute_ring_shapes(paths)

    for pid, path_obj in paths.items():
        width_mm = estimate_path_width_mm(path_obj, svg_width, svg_height,
                                           pixel_per_mm, supersample_factor)
        path_obj.width_mm = width_mm
        path_obj.label = classify_path(path_obj, width_mm)

        print(f"Path {pid}: width={width_mm:.3f}mm, "
              f"boundary={path_obj.is_outer_boundary}, ring={path_obj.is_ring_shape}, "
              f"label={path_obj.label}")

    return render_classified_mask(paths, output_width, output_height,
                                   svg_width, svg_height, supersample_factor)


def render_classified_mask(paths: Dict[str, SVGPath], output_width: int, output_height: int,
                            svg_width: int, svg_height: int,
                            supersample_factor: int = 1) -> np.ndarray:
    mask = np.zeros((output_height, output_width), dtype=np.uint8)

    for path_obj in paths.values():
        if path_obj.label == 'FILL':
            label_value = LABEL_FILL
        elif path_obj.label == 'SATIN':
            label_value = LABEL_SATIN
        else:
            continue

        path_mask = render_path_to_mask(path_obj.element, svg_width, svg_height, supersample_factor)
        if path_mask is not None:
            if path_mask.shape != (output_height, output_width):
                path_mask = cv2.resize(path_mask, (output_width, output_height),
                                        interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, path_mask * label_value)

    return mask


def classify_svg_file(svg_path: str, output_mask_path: Optional[str] = None,
                       output_width: int = 512, output_height: int = 512) -> np.ndarray:
    mask = render_mask(svg_path, output_width, output_height)

    if output_mask_path:
        colored_mask = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        colored_mask[mask == LABEL_FILL] = [0, 255, 255]    # Cyan = Fill
        colored_mask[mask == LABEL_SATIN] = [255, 0, 255]   # Magenta = Satin
        cv2.imwrite(output_mask_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        print(f"Mask saved to {output_mask_path}")

    return mask


# ---------------------------------------------------------------------------
# Tích hợp với pipeline train (dataset_svg.py)
# ---------------------------------------------------------------------------
def _classify_with_convert_rule(path_obj: SVGPath, outer_border_id: Optional[str],
                                canvas_area: float, svg_width: int, svg_height: int,
                                physical_width_mm: float,
                                threshold_mm: float) -> str:
    """
    Rule tương thích với scripts/data_prep/convert_single_svg.py
    """
    geom = path_obj.geometry
    if geom is None or geom.is_empty:
        return "fill"

    area_px = geom.area
    hull_area_px = geom.convex_hull.area
    solidity = area_px / hull_area_px if hull_area_px > 0 else 1.0

    min_x, min_y, max_x, max_y = geom.bounds
    w, h = max_x - min_x, max_y - min_y
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 100.0

    is_hollow = any(len(poly.interiors) > 0 for poly in _iter_polygons(geom))

    path_mask = render_path_to_mask(path_obj.element, svg_width, svg_height,
                                    supersample_factor=1)
    thickness_mm = 0.0
    if path_mask is not None and path_mask.sum() > 0:
        pixel_to_mm = physical_width_mm / max(float(svg_width), 1.0)
        dist = cv2.distanceTransform(path_mask.astype(np.uint8), cv2.DIST_L2, 5)
        try:
            skeleton = skeletonize(path_mask.astype(bool))
            skeleton_distances = dist[np.where(skeleton)]
            if len(skeleton_distances) > 0:
                median_dist = np.median(skeleton_distances)
                thickness_mm = (median_dist * 2.0) * pixel_to_mm
            else:
                thickness_mm = (np.max(dist) * 2.0) * pixel_to_mm
        except Exception:
            thickness_mm = (np.max(dist) * 2.0) * pixel_to_mm

    is_outermost = (
        path_obj.id == outer_border_id
        and canvas_area > 0
        and (hull_area_px / canvas_area) > 0.4
    )

    if is_outermost and is_hollow and thickness_mm <= OUTER_BORDER_MAX_THICKNESS_MM:
        return "satin"
    if is_hollow:
        return "fill" if thickness_mm >= threshold_mm else "satin"
    if solidity < 0.65 and thickness_mm < (threshold_mm * 1.5):
        return "satin"
    if aspect_ratio > 4.0 and thickness_mm <= 4.0:
        return "satin"
    if solidity > 0.85 and thickness_mm >= threshold_mm:
        return "fill"
    return "satin" if thickness_mm < threshold_mm else "fill"


def build_convert_rule_metadata(svg_path: str, physical_width_mm: float = 80.0,
                                threshold_mm: float = 2.0) -> Dict[str, str]:
    root, paths, _ = load_svg(svg_path)
    if not paths:
        return {}

    svg_width, svg_height = get_svg_dimensions(svg_path)
    canvas_area = float(svg_width * svg_height)

    max_hull_area = 0.0
    outer_border_id = None
    for pid, path_obj in paths.items():
        geom = render_path_to_polygon(path_obj.element)
        if geom is None or geom.is_empty:
            continue
        path_obj.geometry = geom
        hull_area = geom.convex_hull.area
        if hull_area > max_hull_area:
            max_hull_area = hull_area
            outer_border_id = pid

    metadata: Dict[str, str] = {}
    for pid, path_obj in paths.items():
        metadata[pid] = _classify_with_convert_rule(
            path_obj,
            outer_border_id=outer_border_id,
            canvas_area=canvas_area,
            svg_width=svg_width,
            svg_height=svg_height,
            physical_width_mm=physical_width_mm,
            threshold_mm=threshold_mm,
        )

    return metadata


def build_geometric_metadata(svg_path: str, supersample_factor: int = 2) -> Dict[str, str]:
    root, paths, pixel_per_mm = load_svg(svg_path)
    if not paths:
        return {}

    svg_width, svg_height = get_svg_dimensions(svg_path)

    find_outer_boundary_paths(paths, svg_width, svg_height)
    compute_ring_shapes(paths)

    metadata: Dict[str, str] = {}
    for pid, path_obj in paths.items():
        width_mm = estimate_path_width_mm(path_obj, svg_width, svg_height,
                                           pixel_per_mm, supersample_factor)
        path_obj.width_mm = width_mm
        path_obj.label = classify_path(path_obj, width_mm)
        metadata[pid] = path_obj.label.lower()  

    return metadata


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python svg_path_classifier.py <svg_path> [output_mask_path]")
        sys.exit(1)

    svg_path_arg = sys.argv[1]
    output_path_arg = sys.argv[2] if len(sys.argv) > 2 else None

    mask = classify_svg_file(svg_path_arg, output_path_arg)

    print(f"\nMask shape: {mask.shape}")
    print(f"Background pixels: {(mask == LABEL_BACKGROUND).sum()}")
    print(f"Fill pixels: {(mask == LABEL_FILL).sum()}")
    print(f"Satin pixels: {(mask == LABEL_SATIN).sum()}")


def inspect_svg_scale(svg_path: str) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    width_attr = root.get("width")
    height_attr = root.get("height")
    viewbox_attr = root.get("viewBox")

    print(f"File: {svg_path}")
    print(f"  width attr  : {width_attr!r}")
    print(f"  height attr : {height_attr!r}")
    print(f"  viewBox attr: {viewbox_attr!r}")

    value, unit = parse_dimension(width_attr) if width_attr else (None, None)
    print(f"  parsed width value/unit: {value} / {unit}")

    pixel_per_mm = calculate_pixel_per_mm(root)
    print(f"  => pixel_per_mm = {pixel_per_mm:.4f}")

    if unit is None:
        print(f"  [CANH BAO] Khong tim thay don vi thuc (cm/mm/in) tren "
              f"thuoc tinh width -> dang FALLBACK ve gia dinh logo rong "
              f"{DEFAULT_REAL_WIDTH_MM}mm. Neu kich thuoc that su khac, "
              f"moi phep do width phia sau se SAI theo ti le tuong ung.")