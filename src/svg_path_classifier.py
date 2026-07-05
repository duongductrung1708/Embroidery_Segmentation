"""
SVG Path Classifier for Embroidery Segmentation
================================================

Phân loại tự động path SVG thành:
- Background (0)
- Fill (1)
- Satin (2)

Chỉ dựa trên hình học (KHÔNG dùng màu, stroke-width, hay tên layer).

LƯU Ý: file này CHỈ CÒN 1 bộ rule duy nhất -- rule mm, dùng chung với bước
convert PNG -> SVG (convert_single_svg.py) và với dataset_svg.py lúc train.
Bản cũ từng có thêm 1 bộ rule cm riêng biệt (build_geometric_metadata,
classify_path, estimate_path_width_cm...) -- đã XOÁ HẲN vì gây nhầm lẫn:
CLI debug dùng rule cm trong khi train thật dùng rule mm, ra 2 kết quả
khác nhau cho cùng 1 file. Giờ CLI và train dùng chung đúng 1 hàm.

QUY TẮC (build_convert_rule_metadata / _classify_with_convert_rule):
Đơn vị: MILIMET (mm), quy đổi theo giả định physical_width_mm (mặc định
80mm) là chiều rộng thực tế của TOÀN BỘ canvas SVG.

1. Path là viền ngoài cùng thực sự (hull area > 40% canvas, có lỗ rỗng)
   VÀ độ dày <= 8.0mm -> SATIN
2. Path có lỗ rỗng (hollow) còn lại:
   - độ dày >= threshold_mm (mặc định 2.0mm) -> FILL
   - mỏng hơn -> SATIN
3. Path không lỗ, solidity < 0.65 (nét ngoằn ngoèo kiểu chữ S, C, M) và
   độ dày < threshold_mm * 1.5 -> SATIN
4. Path dài/hẹp (aspect ratio > 4.0) và độ dày <= 4.0mm -> SATIN
5. Path đặc, solidity > 0.85, độ dày >= threshold_mm -> FILL
6. Fallback: độ dày < threshold_mm -> SATIN, ngược lại -> FILL

QUAN TRỌNG: physical_width_mm=80.0 là một GIẢ ĐỊNH CỐ ĐỊNH áp cho MỌI file
-- chỉ đúng nếu toàn bộ dataset thực sự được sinh ra từ cùng 1 khuôn/kích
thước 80mm. Nếu không, mọi ngưỡng mm phía trên (2mm, 8mm, 4mm) đều sai
lệch theo đúng tỉ lệ đó.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_BACKGROUND = 0
LABEL_FILL = 1
LABEL_SATIN = 2

DEFAULT_PHYSICAL_WIDTH_MM = 80.0
DEFAULT_THRESHOLD_MM = 2.0

# Giới hạn độ dày tối đa để 1 viền ngoài cùng (outermost, có lỗ) được coi
# là SATIN. Vượt ngưỡng này (ví dụ mảng nền lớn tình cờ chạm mép) -> FILL.
OUTER_BORDER_MAX_THICKNESS_MM = 8.0

_PREVIEW_COLOR_FILL = "#FF0000"
_PREVIEW_COLOR_SATIN = "#00FF00"


class SVGPath:
    """Container cho thông tin 1 path SVG."""

    def __init__(self, path_id: str, element: ET.Element, path_data: str):
        self.id = path_id
        self.element = element
        self.path_data = path_data
        self.geometry = None            # shapely geometry đã xử lý hole
        self.label: Optional[str] = None  # 'satin' hoặc 'fill'


# ---------------------------------------------------------------------------
# Load SVG & kích thước
# ---------------------------------------------------------------------------
def load_svg(svg_path: str) -> Tuple[ET.Element, Dict[str, SVGPath]]:
    """Load SVG, trả về (root, dict path_id -> SVGPath)."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    paths: Dict[str, SVGPath] = {}
    for idx, child in enumerate(root.iter()):
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            path_id = child.get("id", f"path_{idx}")
            path_data = child.get("d", "")
            if path_data:
                paths[path_id] = SVGPath(path_id, child, path_data)

    return root, paths


def get_svg_dimensions(svg_path: str) -> Tuple[int, int]:
    """Lấy width/height (pixel/đơn vị SVG gốc) từ viewBox hoặc width/height."""
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
    """Tách 'd' thành các subpath rời rạc (mỗi lệnh M mới bắt đầu 1 subpath)
    và sample điểm cho từng subpath."""
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
    """
    Dựng geometry chính xác cho 1 path SVG có thể gồm nhiều subpath
    (ví dụ chữ 'O', 'A' có lỗ, hoặc viền lồng nhau).

    1. Sample từng subpath thành 1 Polygon riêng (chưa có hole).
    2. Tính độ sâu lồng nhau (nesting depth): 1 subpath nằm trong bao nhiêu
       subpath khác.
    3. Subpath depth chẵn (0, 2, 4...) là phần đặc (exterior).
       Subpath depth lẻ (1, 3, 5...) là lỗ (hole) của subpath cha gần nhất.
    4. Ghép thành MultiPolygon (exterior kèm hole tương ứng).
    """
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
                inside = raw_polys[i].contains(raw_polys[j])
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
    """Duyệt qua từng Polygon con của 1 geometry (Polygon hoặc MultiPolygon)."""
    if geom is None:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms


def rasterize_geometry_to_mask(geom, svg_width: int, svg_height: int,
                                supersample_factor: int = 1) -> Optional[np.ndarray]:
    """
    Rasterize TRỰC TIẾP từ shapely geometry (đã xử lý hole đúng qua nesting
    depth ở render_path_to_polygon) thành binary mask.

    Cố tình KHÔNG dùng render_path_to_mask (render lại path 'd' gốc qua
    cairosvg) để đo độ dày, vì cairosvg áp dụng fill-rule 'nonzero' theo
    đúng chiều vẽ (winding) thực tế của path: nếu 2 subpath (viền ngoài +
    lỗ trong) được vẽ CÙNG chiều thay vì ngược chiều chuẩn, cairosvg sẽ
    render thành hình ĐẶC (không có lỗ) dù is_hollow (dựa trên shapely,
    không phụ thuộc chiều vẽ) đã xác định đúng là có lỗ. Hai phần bị lệch
    pha nhau khiến thickness_mm đo được là của hình đặc, không phải hình
    vành khăn -- làm ngưỡng threshold_mm phía sau sai hoàn toàn cho đúng
    loại hình (viền tròn, chữ có lỗ như O/D/B/Q/R) mà rule 1 & 2 quan tâm
    nhất. Rasterize thẳng từ geometry tránh được vấn đề này hoàn toàn.
    """
    if geom is None or geom.is_empty:
        return None

    w = max(1, int(round(svg_width * supersample_factor)))
    h = max(1, int(round(svg_height * supersample_factor)))
    mask = np.zeros((h, w), dtype=np.uint8)

    for poly in _iter_polygons(geom):
        try:
            ext = (np.array(poly.exterior.coords) * supersample_factor).astype(np.int32)
            cv2.fillPoly(mask, [ext], 1)
            for interior in poly.interiors:
                hole = (np.array(interior.coords) * supersample_factor).astype(np.int32)
                cv2.fillPoly(mask, [hole], 0)
        except Exception as e:
            print(f"Error rasterizing polygon: {e}")

    return mask


# ---------------------------------------------------------------------------
# Rule mm -- DUY NHẤT bộ rule đang thực sự dùng để train (qua dataset_svg.py)
# ---------------------------------------------------------------------------
def _classify_with_convert_rule(path_obj: SVGPath, outer_border_id: Optional[str],
                                 canvas_area: float, svg_width: int, svg_height: int,
                                 physical_width_mm: float,
                                 threshold_mm: float) -> Tuple[str, Dict]:
    """
    Rule tương thích với bước convert PNG -> SVG (convert_single_svg.py).
    Trả về (label, details) -- details dùng để in diagnostic khi verbose=True.
    """
    geom = path_obj.geometry
    if geom is None or geom.is_empty:
        return "fill", {
            "thickness_mm": 0.0, "solidity": 1.0, "aspect_ratio": 0.0,
            "is_hollow": False, "is_outermost": False,
        }

    area_px = geom.area
    hull_area_px = geom.convex_hull.area
    solidity = area_px / hull_area_px if hull_area_px > 0 else 1.0

    min_x, min_y, max_x, max_y = geom.bounds
    w, h = max_x - min_x, max_y - min_y
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 100.0

    is_hollow = any(len(poly.interiors) > 0 for poly in _iter_polygons(geom))

    path_mask = rasterize_geometry_to_mask(geom, svg_width, svg_height,
                                            supersample_factor=1)
    thickness_mm = 0.0
    if path_mask is not None and path_mask.sum() > 0:
        pixel_to_mm = physical_width_mm / max(float(svg_width), 1.0)
        dist = cv2.distanceTransform(path_mask.astype(np.uint8), cv2.DIST_L2, 5)
        try:
            from skimage.morphology import skeletonize
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
        label = "satin"
    elif is_hollow:
        label = "fill" if thickness_mm >= threshold_mm else "satin"
    elif solidity < 0.65 and thickness_mm < (threshold_mm * 1.5):
        label = "satin"
    elif aspect_ratio > 4.0 and thickness_mm <= 4.0:
        label = "satin"
    elif solidity > 0.85 and thickness_mm >= threshold_mm:
        label = "fill"
    else:
        label = "satin" if thickness_mm < threshold_mm else "fill"

    details = {
        "thickness_mm": thickness_mm,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "is_hollow": is_hollow,
        "is_outermost": is_outermost,
    }
    return label, details


def build_convert_rule_metadata(svg_path: str, physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                                 threshold_mm: float = DEFAULT_THRESHOLD_MM,
                                 verbose: bool = False) -> Dict[str, str]:
    """
    Trả về {path_id: "satin"/"fill"} theo rule mm -- dùng cho:
    1. Fallback trong dataset_svg.build_hybrid_metadata() lúc train khi
       path chưa có inkscape:label.
    2. CLI debug (python svg_path_classifier.py file.svg) -- CÙNG một hàm,
       nên kết quả CLI và kết quả train luôn khớp nhau.
    """
    root, paths = load_svg(svg_path)
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

    if verbose:
        print(f"File: {svg_path}")
        print(f"  Canvas: {svg_width}x{svg_height} (đơn vị SVG gốc), "
              f"giả định rộng {physical_width_mm}mm thực tế")
        print(f"  Outer border (hull lớn nhất): {outer_border_id}")

    metadata: Dict[str, str] = {}
    for pid, path_obj in paths.items():
        label, details = _classify_with_convert_rule(
            path_obj,
            outer_border_id=outer_border_id,
            canvas_area=canvas_area,
            svg_width=svg_width,
            svg_height=svg_height,
            physical_width_mm=physical_width_mm,
            threshold_mm=threshold_mm,
        )
        path_obj.label = label
        metadata[pid] = label

        if verbose:
            print(f"  Path {pid}: thickness={details['thickness_mm']:.3f}mm, "
                  f"hollow={details['is_hollow']}, outermost={details['is_outermost']}, "
                  f"solidity={details['solidity']:.2f}, aspect={details['aspect_ratio']:.2f}, "
                  f"label={label}")

    return metadata


# ---------------------------------------------------------------------------
# Render mask 3 lớp (0=BG, 1=Fill, 2=Satin) để xem trước / debug
# ---------------------------------------------------------------------------
def render_mask(svg_path: str, output_width: int = 512, output_height: int = 512,
                 physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                 threshold_mm: float = DEFAULT_THRESHOLD_MM,
                 supersample_factor: int = 2, verbose: bool = True) -> np.ndarray:
    """Chạy build_convert_rule_metadata rồi render mask minh hoạ (giống hệt
    cách dataset_svg.create_label_mask render mask thật lúc train)."""
    metadata = build_convert_rule_metadata(svg_path, physical_width_mm=physical_width_mm,
                                            threshold_mm=threshold_mm, verbose=verbose)
    if not metadata:
        return np.zeros((output_height, output_width), dtype=np.uint8)

    tree = ET.parse(svg_path)
    root = tree.getroot()

    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            pid = child.get("id")
            label = metadata.get(pid, "fill")
            color = _PREVIEW_COLOR_SATIN if label == "satin" else _PREVIEW_COLOR_FILL
            child.set("fill", color)
            child.attrib.pop("style", None)
            child.set("fill-opacity", "1")
            child.set("stroke", "none")

    render_width = output_width * supersample_factor
    render_height = output_height * supersample_factor

    svg_bytes = ET.tostring(root, encoding='unicode')
    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes.encode('utf-8'),
        output_width=render_width,
        output_height=render_height,
        background_color=None,
        unsafe=True,
    )

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

    return cv2.resize(mask_render, (output_width, output_height), interpolation=cv2.INTER_NEAREST)


def classify_svg_file(svg_path: str, output_mask_path: Optional[str] = None,
                       output_width: int = 512, output_height: int = 512,
                       physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                       threshold_mm: float = DEFAULT_THRESHOLD_MM) -> np.ndarray:
    """Phân loại 1 file SVG, tuỳ chọn lưu ảnh mask minh hoạ."""
    mask = render_mask(svg_path, output_width, output_height,
                        physical_width_mm=physical_width_mm, threshold_mm=threshold_mm,
                        verbose=True)

    if output_mask_path:
        colored_mask = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        colored_mask[mask == LABEL_FILL] = [0, 255, 255]    # Cyan = Fill
        colored_mask[mask == LABEL_SATIN] = [255, 0, 255]   # Magenta = Satin
        cv2.imwrite(output_mask_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        print(f"Mask saved to {output_mask_path}")

    return mask


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python svg_path_classifier.py <svg_path> [output_mask_path] "
              "[physical_width_mm] [threshold_mm]")
        print(f"  (mac dinh physical_width_mm={DEFAULT_PHYSICAL_WIDTH_MM}, "
              f"threshold_mm={DEFAULT_THRESHOLD_MM} -- giong het luc train)")
        sys.exit(1)

    svg_path_arg = sys.argv[1]
    output_path_arg = sys.argv[2] if len(sys.argv) > 2 else None
    physical_width_mm_arg = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_PHYSICAL_WIDTH_MM
    threshold_mm_arg = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_THRESHOLD_MM

    mask = classify_svg_file(svg_path_arg, output_path_arg,
                              physical_width_mm=physical_width_mm_arg,
                              threshold_mm=threshold_mm_arg)

    print(f"\nMask shape: {mask.shape}")
    print(f"Background pixels: {(mask == LABEL_BACKGROUND).sum()}")
    print(f"Fill pixels: {(mask == LABEL_FILL).sum()}")
    print(f"Satin pixels: {(mask == LABEL_SATIN).sum()}")