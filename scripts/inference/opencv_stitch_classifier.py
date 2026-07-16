#!/usr/bin/env python3
"""
OpenCV Stitch Classifier - Satin vs Fill (rule-based, KHÔNG train model)
=========================================================================

BẢN CẬP NHẬT: THUẬT TOÁN INK RATIO CHỐNG "NỀN GIẢ VIỀN"
- Tôn trọng tuyệt đối tỷ lệ Threshold = 1/6 chiều rộng vật lý (Bỏ trần 12mm).
- [MỚI]: Tính toán `ink_ratio` (Tỷ lệ lượng mực). Nếu một mảng chiếm > 35% 
  tổng lượng mực của cả logo VÀ khá đặc (Solidity > 0.45) -> Chắc chắn là NỀN (Fill),
  bất chấp nó bị đục lỗ làm sai lệch độ dày.
- Khắc phục triệt để hiệu ứng Domino ép sai các nét mảnh bên trong vòng tròn.
"""

import argparse
import glob
import os
import sys
import re
import concurrent.futures
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import cairosvg
except ImportError:
    cairosvg = None

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_BACKGROUND = 0
LABEL_FILL = 1
LABEL_SATIN = 2

DEFAULT_PHYSICAL_WIDTH_MM = 80.0
MIN_PHYSICAL_STITCH_MM = 0.4
MIN_NOISE_AREA_MM2 = 0.2
CONTEXT_CONTAINMENT_RATIO = 0.5
TOUCH_DILATION_PHYSICAL_MM = 0.5 

_PREVIEW_COLOR_FILL = (0, 255, 255)   # Cyan (Hiện màu Vàng)
_PREVIEW_COLOR_SATIN = (255, 0, 255)  # Magenta (Hiện màu Hồng)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_SVG_CANVAS_W = 4200
DEFAULT_SVG_CANVAS_H = 4800
SVG_CLASSIFY_SCALE = 1400.0 / DEFAULT_SVG_CANVAS_W
_PNG_SAVE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]

INKSCAPE_NS = "{http://www.inkscape.org/namespaces/inkscape}"


class Shape:
    def __init__(self, shape_id: int, contour: np.ndarray, holes: List[np.ndarray]):
        self.id = shape_id
        self.contour = contour
        self.holes = holes
        self.label: Optional[str] = None
        self.global_id: Optional[int] = None


def parse_svg_physical_width(root: ET.Element, default_mm: float = 80.0) -> float:
    w_str = root.get("width")
    if w_str:
        match = re.match(r"^([\d.]+)([a-zA-Z%]*)$", w_str.strip())
        if match:
            val = float(match.group(1))
            unit = match.group(2).lower()
            if unit == "mm": return val
            if unit == "cm": return val * 10.0
            if unit == "in": return val * 25.4
            if unit in ("px", "pt", ""): return val * (25.4 / 96.0)

    vb = root.get("viewBox")
    if vb:
        parts = vb.replace(",", " ").split()
        if len(parts) == 4:
            return float(parts[2]) * (25.4 / 96.0)
            
    return default_mm


def normalize_to_canvas(img: np.ndarray, target_w: int = 4200, target_h: int = 4800) -> np.ndarray:
    h_orig, w_orig = img.shape[:2]
    if h_orig == target_h and w_orig == target_w:
        if len(img.shape) == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3: return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    scale = min(target_w / w_orig, target_h / h_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if len(resized_img.shape) == 2: resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGRA)
    elif resized_img.shape[2] == 3:
        rgba = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        rgba[:, :, :3] = resized_img
        rgba[:, :, 3] = 255
        resized_img = rgba

    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    return canvas


def extract_shapes(binary_mask: np.ndarray) -> List[Shape]:
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None: return []
    hierarchy = hierarchy[0]

    shapes: List[Shape] = []
    for i, h in enumerate(hierarchy):
        if h[3] != -1: continue
        holes = [contours[j] for j, hj in enumerate(hierarchy) if hj[3] == i]
        if cv2.contourArea(contours[i]) < 4: continue
        shapes.append(Shape(shape_id=len(shapes), contour=contours[i], holes=holes))
    return shapes


def shape_to_mask(shape: Shape, canvas_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(canvas_shape, dtype=np.uint8)
    cv2.drawContours(mask, [shape.contour], -1, 1, thickness=cv2.FILLED)
    for hole in shape.holes:
        cv2.drawContours(mask, [hole], -1, 0, thickness=cv2.FILLED)
    return mask


def thickness_mm_from_mask(mask: np.ndarray, pixel_to_mm: float) -> Tuple[float, float]:
    if mask.sum() == 0: return 0.0, 0.0

    h, w = mask.shape
    MAX_EVAL_SIZE = 1500
    scale_factor = 1.0

    if max(h, w) > MAX_EVAL_SIZE:
        scale_factor = MAX_EVAL_SIZE / max(h, w)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        eval_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        eval_mask = mask

    dist = cv2.distanceTransform(eval_mask.astype(np.uint8), cv2.DIST_L2, 5)
    actual_pixel_to_mm = pixel_to_mm / scale_factor
    max_thickness_mm = (np.max(dist) * 2.0) * actual_pixel_to_mm
    
    median_thickness_mm = max_thickness_mm
    if skeletonize is not None:
        try:
            skeleton = skeletonize(eval_mask > 0)
            skel_dist = dist[skeleton]
            if len(skel_dist) > 0:
                median_thickness_mm = (np.median(skel_dist) * 2.0) * actual_pixel_to_mm
        except:
            pass

    return median_thickness_mm, max_thickness_mm


def _classify_shape(shape: Shape, mask: np.ndarray, is_outer_candidate: bool,
                     canvas_area: float, pixel_to_mm: float,
                     threshold_mm: float, total_fg_area_px: float) -> Tuple[str, Dict]:

    hull = cv2.convexHull(shape.contour)
    hull_area_px = cv2.contourArea(hull)
    true_area_px = float(np.count_nonzero(mask))
    
    solidity = true_area_px / hull_area_px if hull_area_px > 0 else 1.0
    ink_ratio = true_area_px / total_fg_area_px if total_fg_area_px > 0 else 1.0

    rect = cv2.minAreaRect(shape.contour)
    short_side, long_side = sorted([rect[1][0], rect[1][1]])
    aspect_ratio = long_side / short_side if short_side > 0 else 100.0

    is_hollow = len(shape.holes) > 0
    thickness_median_mm, thickness_max_mm = thickness_mm_from_mask(mask, pixel_to_mm)
    thickness_mm = thickness_median_mm

    is_outermost = (is_outer_candidate and canvas_area > 0 and (hull_area_px / canvas_area) > 0.4)

    details = {
        "thickness_mm": thickness_median_mm,
        "thickness_max_mm": thickness_max_mm,
        "solidity": solidity,
        "ink_ratio": ink_ratio,
        "aspect_ratio": aspect_ratio,
        "is_hollow": is_hollow,
        "is_outermost": is_outermost
    }

    if thickness_mm < MIN_PHYSICAL_STITCH_MM:
        area_mm2 = true_area_px * (pixel_to_mm ** 2)
        if area_mm2 < MIN_NOISE_AREA_MM2:
            return "noise", details

    if thickness_mm <= threshold_mm:
        # Nếu chiếm > 35% tổng lượng mực toàn logo VÀ khối lượng đặc khá cao (> 0.45)
        # Rất có khả năng đây là Nền Background lớn bị đục lỗ đánh lừa thuật toán. Ép về FILL!
        if ink_ratio > 0.35 and solidity > 0.45:
            label = "fill"
        else:
            label = "satin"
    else:
        label = "fill"

    return label, details


def classify_binary_mask(binary_mask: np.ndarray,
                          physical_width_mm: float,
                          threshold_mm: float,
                          total_fg_area_px: float,
                          verbose: bool = False) -> Tuple[List[Shape], np.ndarray]:
    h, w = binary_mask.shape[:2]
    canvas_area = float(w * h)
    pixel_to_mm = physical_width_mm / max(float(w), 1.0)

    shapes = extract_shapes(binary_mask)
    if not shapes: return [], np.zeros((h, w), dtype=np.uint8)

    max_hull_area = 0.0
    outer_border_id = -1
    for shape in shapes:
        hull_area = cv2.contourArea(cv2.convexHull(shape.contour))
        if hull_area > max_hull_area:
            max_hull_area = hull_area
            outer_border_id = shape.id

    label_mask = np.zeros((h, w), dtype=np.uint8)

    for shape in shapes:
        mask = shape_to_mask(shape, (h, w))
        
        hole_mask = np.zeros((h, w), dtype=np.uint8)
        for hole in shape.holes:
            cv2.drawContours(hole_mask, [hole], -1, 1, thickness=cv2.FILLED)
            
        label, details = _classify_shape(
            shape, mask,
            is_outer_candidate=(shape.id == outer_border_id),
            canvas_area=canvas_area,
            pixel_to_mm=pixel_to_mm,
            threshold_mm=threshold_mm,
            total_fg_area_px=total_fg_area_px
        )
        shape.label = label

        if label in ["satin", "fill"]:
            label_value = LABEL_SATIN if label == "satin" else LABEL_FILL
            final_draw_mask = (mask == 1) & (hole_mask == 0)
            label_mask[final_draw_mask] = label_value

        if verbose:
            print(f"  Shape {shape.id}: med={details['thickness_mm']:.3f}mm, "
                  f"sol={details['solidity']:.2f}, ink={details['ink_ratio']:.2f} -> {label.upper()}")

    return shapes, label_mask


# ---------------------------------------------------------------------------
# Context-aware Classification
# ---------------------------------------------------------------------------
class _GlobalShapeRecord:
    def __init__(self, shape: Shape, mask: np.ndarray, area_px: int):
        self.shape = shape
        self.mask = mask
        self.area_px = area_px


def _build_interior_mask(shape: Shape, shape_mask: np.ndarray,
                          canvas_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if len(shape.holes) == 0: return None
    outer_mask = np.zeros(canvas_shape, dtype=np.uint8)
    cv2.drawContours(outer_mask, [shape.contour], -1, 1, thickness=cv2.FILLED)
    interior_mask = outer_mask & (1 - shape_mask)
    if interior_mask.sum() == 0: return None
    return interior_mask


def _dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    if dilation_px <= 0: return mask
    kernel = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def refine_labels_by_context(records: List[_GlobalShapeRecord],
                              label_mask: np.ndarray,
                              canvas_shape: Tuple[int, int],
                              pixel_to_mm: float,
                              containment_ratio: float = CONTEXT_CONTAINMENT_RATIO,
                              verbose: bool = False) -> np.ndarray:
    
    touch_dilation_px = max(1, int(TOUCH_DILATION_PHYSICAL_MM / pixel_to_mm))
    
    outline_candidates = [r for r in records if r.shape.label == "satin" and len(r.shape.holes) > 0]
    outline_candidates = sorted(
        outline_candidates,
        key=lambda r: cv2.contourArea(cv2.convexHull(r.shape.contour)),
        reverse=True,
    )

    for outline in outline_candidates:
        if outline.shape.label != "satin": continue

        interior_mask = _build_interior_mask(outline.shape, outline.mask, canvas_shape)
        if interior_mask is None: continue

        outline_mask_dilated = _dilate_mask(outline.mask, touch_dilation_px)

        for other in records:
            if other is outline or other.shape.label != "satin" or other.area_px == 0: continue

            overlap_interior = int(np.count_nonzero(interior_mask & other.mask))
            interior_ratio = overlap_interior / other.area_px
            if interior_ratio < containment_ratio: continue

            touch_overlap = int(np.count_nonzero(outline_mask_dilated & other.mask))
            if touch_overlap == 0: continue

            if verbose:
                print(f"  [Context] Shape(id={other.shape.global_id}) dính biên với "
                      f"Outline(id={outline.shape.global_id}) -> ép SATIN thành FILL")
            other.shape.label = "fill"
            label_mask[other.mask == 1] = LABEL_FILL

    return label_mask


# ---------------------------------------------------------------------------
# SVG Processing Utilities
# ---------------------------------------------------------------------------
def _tag(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

def _is_in_defs_or_clip(el: ET.Element, parent_map: dict) -> bool:
    node = el
    while node in parent_map:
        node = parent_map[node]
        if _tag(node) in ("defs", "clipPath", "mask"):
            return True
    return False

def get_declared_label(path_elem: ET.Element) -> Optional[str]:
    lbl = path_elem.attrib.get(f"{INKSCAPE_NS}label")
    if lbl:
        lbl = lbl.lower()
        if "satin" in lbl: return "satin"
        if "fill" in lbl: return "fill"
    return None

def _render_single_path_mask(path_elems: List[ET.Element], path_index: int,
                              canvas_w: int, canvas_h: int, root: ET.Element) -> np.ndarray:
    if cairosvg is None:
        raise ImportError("Can cai dat 'cairosvg' de rasterize SVG.")

    saved_attribs = [dict(el.attrib) for el in path_elems]
    try:
        for i, el in enumerate(path_elems):
            if i == path_index:
                el.set("fill", "#000000")
                el.set("fill-opacity", "1")
                el.set("stroke", "none")
                el.attrib.pop("style", None)
                el.attrib.pop("display", None)
                el.attrib.pop("clip-path", None)
            else:
                el.set("display", "none")

        png_bytes = cairosvg.svg2png(bytestring=ET.tostring(root), output_width=canvas_w,
                                      output_height=canvas_h, background_color=None, unsafe=True)
    finally:
        for el, attrib in zip(path_elems, saved_attribs):
            el.attrib.clear()
            el.attrib.update(attrib)

    img = np.array(Image.open(BytesIO(png_bytes)).convert("RGBA"))
    alpha = img[:, :, 3]
    return (alpha >= 128).astype(np.uint8)


def _render_labeled_paths_full_res(path_elems: List[ET.Element],
                                    path_labels: List[Optional[str]],
                                    root: ET.Element,
                                    canvas_w: int, canvas_h: int) -> Tuple[np.ndarray, np.ndarray]:
    saved_attribs = [dict(el.attrib) for el in path_elems]
    try:
        for el, label in zip(path_elems, path_labels):
            if label is None:
                el.set("display", "none")
            else:
                color = "#FF0000" if label == "satin" else "#0000FF"
                el.set("fill", color)
                el.set("fill-opacity", "1")
                el.set("stroke", "none")
                el.attrib.pop("style", None)
                el.attrib.pop("display", None)
                el.attrib.pop("clip-path", None)

        png_bytes = cairosvg.svg2png(bytestring=ET.tostring(root), output_width=canvas_w,
                                      output_height=canvas_h, background_color=None, unsafe=True)
    finally:
        for el, attrib in zip(path_elems, saved_attribs):
            el.attrib.clear()
            el.attrib.update(attrib)

    img = np.array(Image.open(BytesIO(png_bytes)).convert("RGBA"))
    alpha = img[:, :, 3]
    is_visible = alpha >= 128
    r, g, b = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    is_satin = is_visible & (r > 150) & (g < 80) & (b < 80)
    is_fill = is_visible & (b > 150) & (r < 80) & (g < 80)

    label_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    label_mask[is_fill] = LABEL_FILL
    label_mask[is_satin] = LABEL_SATIN
    return label_mask, is_visible


def fill_unlabeled_gaps(label_mask: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    unlabeled = foreground_mask & (label_mask == 0)
    if not unlabeled.any(): return label_mask

    labeled = foreground_mask & (label_mask != 0)
    if not labeled.any(): return label_mask

    from scipy.ndimage import distance_transform_edt
    _, indices = distance_transform_edt(~labeled, return_indices=True)

    filled = label_mask.copy()
    ys, xs = np.where(unlabeled)
    nearest_ys = indices[0][ys, xs]
    nearest_xs = indices[1][ys, xs]
    filled[ys, xs] = label_mask[nearest_ys, nearest_xs]
    return filled


def classify_multicolor_image(image_path: str,
                               physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                               color_tolerance: int = 15,
                               quantize: bool = True,
                               alpha_threshold: int = 10,
                               min_region_pixels: int = 5,
                               enable_context_refinement: bool = True,
                               verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

    img = normalize_to_canvas(img, target_w=4200, target_h=4800)
    img_bgr = img[:, :, :3]
    alpha = img[:, :, 3]
    is_background = alpha < alpha_threshold
    foreground_mask = ~is_background

    h, w = img_bgr.shape[:2]
    canvas_shape = (h, w)
    label_mask = np.zeros((h, w), dtype=np.uint8)
    
    ys, xs = np.where(foreground_mask)
    if len(xs) > 0:
        crop_w = xs.max() - xs.min() + 1
        logo_real_w_mm = physical_width_mm * (crop_w / w)
        dynamic_threshold_mm = logo_real_w_mm / 6.0
    else:
        dynamic_threshold_mm = physical_width_mm / 6.0

    if verbose:
        print(f">> [Auto-Size] Logo Real Width = {logo_real_w_mm:.2f}mm | Threshold = {dynamic_threshold_mm:.2f}mm")

    total_fg_area_px = float(np.count_nonzero(foreground_mask))

    def quantize_colors_fast(img_bgr: np.ndarray, valid_mask: np.ndarray, step: int = 48) -> np.ndarray:
        quantized = img_bgr.copy()
        rounded = (np.round(quantized[valid_mask] / step) * step).clip(0, 255).astype(np.uint8)
        quantized[valid_mask] = rounded
        return quantized

    working_img = quantize_colors_fast(img_bgr, foreground_mask, step=48) if quantize else img_bgr.copy()
    working_img[is_background] = (0, 0, 0)

    pixels = working_img[foreground_mask].reshape(-1, 3)
    if pixels.size == 0: return label_mask, working_img

    unique_colors = np.unique(pixels, axis=0)

    global_records: List[_GlobalShapeRecord] = []
    global_id_counter = 0

    for color in unique_colors:
        color_int = color.astype(np.int16)
        lower = np.clip(color_int - color_tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(color_int + color_tolerance, 0, 255).astype(np.uint8)
        color_mask = cv2.inRange(working_img, lower, upper)
        color_mask[is_background] = 0

        n_pixels = int((color_mask > 0).sum())
        if n_pixels < min_region_pixels: continue

        shapes, sub_label_mask = classify_binary_mask(
            color_mask, physical_width_mm=physical_width_mm,
            threshold_mm=dynamic_threshold_mm,
            total_fg_area_px=total_fg_area_px, verbose=verbose,
        )
        label_mask[sub_label_mask > 0] = sub_label_mask[sub_label_mask > 0]

        for shape in shapes:
            if shape.label not in ("satin", "fill"): continue
            shape.global_id = global_id_counter
            global_id_counter += 1
            shape_mask = shape_to_mask(shape, canvas_shape)
            area_px = int(np.count_nonzero(shape_mask))
            global_records.append(_GlobalShapeRecord(shape, shape_mask, area_px))

    if enable_context_refinement and global_records:
        pixel_to_mm_full = physical_width_mm / max(float(w), 1.0)
        label_mask = refine_labels_by_context(global_records, label_mask, canvas_shape, pixel_to_mm=pixel_to_mm_full, verbose=verbose)

    label_mask = fill_unlabeled_gaps(label_mask, foreground_mask)
    return label_mask, working_img


# ---------------------------------------------------------------------------
# Master Classification Pipeline
# ---------------------------------------------------------------------------
def classify_svg(svg_path: str,
                  fallback_physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                  canvas_w: int = DEFAULT_SVG_CANVAS_W,
                  canvas_h: int = DEFAULT_SVG_CANVAS_H,
                  classify_scale: float = SVG_CLASSIFY_SCALE,
                  min_region_pixels: int = 5,
                  enable_context_refinement: bool = True,
                  use_svg_labels: bool = False,
                  pre_parsed_root: Optional[ET.Element] = None,
                  verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    
    if cairosvg is None:
        raise ImportError("Can cai dat 'cairosvg' de doc va rasterize file SVG.")

    if pre_parsed_root is not None:
        root = pre_parsed_root
    else:
        tree = ET.parse(svg_path)
        root = tree.getroot()

    parent_map = {c: p for p in root.iter() for c in p}
    path_elems = [el for el in root.iter() if _tag(el) == "path" and not _is_in_defs_or_clip(el, parent_map)]

    full_png = cairosvg.svg2png(url=svg_path, output_width=canvas_w, output_height=canvas_h,
                                 background_color=None, unsafe=True)
    full_rgba = np.array(Image.open(BytesIO(full_png)).convert("RGBA"))
    alpha = full_rgba[:, :, 3]
    total_fg = alpha >= 128
    
    rendered_img_bgr = cv2.cvtColor(full_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    white_bg = np.ones_like(rendered_img_bgr) * 255
    rendered_img_bgr = np.where(total_fg[:, :, None], rendered_img_bgr, white_bg)

    label_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    if not total_fg.any() or not path_elems:
        return label_mask, rendered_img_bgr

    canvas_physical_mm = parse_svg_physical_width(root, fallback_physical_width_mm)
    ys, xs = np.where(total_fg)
    crop_w = xs.max() - xs.min() + 1
    logo_real_w_mm = canvas_physical_mm * (crop_w / canvas_w)
    dynamic_threshold_mm = logo_real_w_mm / 6.0

    if verbose:
        print(f">> [Auto-Size] Logo Real Width = {logo_real_w_mm:.2f}mm | Threshold = {dynamic_threshold_mm:.2f}mm")

    small_w = max(1, int(round(canvas_w * classify_scale)))
    small_h = max(1, int(round(canvas_h * classify_scale)))
    small_canvas_shape = (small_h, small_w)
    pixel_to_mm_small = canvas_physical_mm / max(float(small_w), 1.0)
    
    # Tính tổng lượng mực trên canvas nhỏ để làm chuẩn cho ink_ratio
    small_total_fg = cv2.resize(total_fg.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    small_total_fg_area_px = float(np.count_nonzero(small_total_fg))

    label_mask_small = np.zeros((small_h, small_w), dtype=np.uint8)
    global_records: List[_GlobalShapeRecord] = []
    shape_path_idx: List[int] = []
    global_id_counter = 0

    path_iter = enumerate(path_elems)
    if verbose:
        path_iter = tqdm(list(path_iter), desc="Rasterizing paths", unit="path", leave=False)

    path_labels: List[Optional[str]] = [None] * len(path_elems)

    for idx, path_elem in path_iter:
        if use_svg_labels:
            pre_label = get_declared_label(path_elem)
            if pre_label:
                path_labels[idx] = pre_label
                continue

        path_mask01 = _render_single_path_mask(path_elems, idx, small_w, small_h, root)
        n_pixels = int(path_mask01.sum())
        if n_pixels < min_region_pixels: continue

        binary_mask_255 = (path_mask01 * 255).astype(np.uint8)
        shapes, sub_label_mask = classify_binary_mask(
            binary_mask_255, physical_width_mm=canvas_physical_mm,
            threshold_mm=dynamic_threshold_mm, 
            total_fg_area_px=small_total_fg_area_px, verbose=verbose,
        )
        label_mask_small[sub_label_mask > 0] = sub_label_mask[sub_label_mask > 0]

        for shape in shapes:
            if shape.label not in ("satin", "fill"): continue
            shape.global_id = global_id_counter
            global_id_counter += 1
            shape_mask_small = shape_to_mask(shape, small_canvas_shape)
            area_px = int(np.count_nonzero(shape_mask_small))
            global_records.append(_GlobalShapeRecord(shape, shape_mask_small, area_px))
            shape_path_idx.append(idx)

    if enable_context_refinement and global_records:
        refine_labels_by_context(global_records, label_mask_small, small_canvas_shape, pixel_to_mm=pixel_to_mm_small, verbose=verbose)

    path_satin_area = [0] * len(path_elems)
    path_fill_area = [0] * len(path_elems)
    for rec, pidx in zip(global_records, shape_path_idx):
        if rec.shape.label == "satin": path_satin_area[pidx] += rec.area_px
        elif rec.shape.label == "fill": path_fill_area[pidx] += rec.area_px

    for idx in range(len(path_elems)):
        if path_labels[idx] is None:
            s_area, f_area = path_satin_area[idx], path_fill_area[idx]
            if s_area > 0 or f_area > 0:
                path_labels[idx] = "satin" if s_area >= f_area else "fill"

    label_mask, foreground_full = _render_labeled_paths_full_res(path_elems, path_labels, root, canvas_w, canvas_h)
    label_mask = fill_unlabeled_gaps(label_mask, foreground_full)

    return label_mask, rendered_img_bgr


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------
def save_preview(label_mask: np.ndarray, output_path: str) -> None:
    h, w = label_mask.shape[:2]
    preview = np.zeros((h, w, 3), dtype=np.uint8)
    preview[label_mask == LABEL_FILL] = _PREVIEW_COLOR_FILL
    preview[label_mask == LABEL_SATIN] = _PREVIEW_COLOR_SATIN
    cv2.imwrite(output_path, preview, _PNG_SAVE_PARAMS)

def save_quantized(working_img_bgr: np.ndarray, output_path: str) -> None:
    cv2.imwrite(output_path, working_img_bgr, _PNG_SAVE_PARAMS)

def process_one_image(image_path: str, preview_dir: str, quantized_dir: str,
                       physical_width_mm: float,
                       enable_context_refinement: bool, use_svg_labels: bool, verbose: bool,
                       classify_scale: float = SVG_CLASSIFY_SCALE) -> None:
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if image_path.lower().endswith(".svg"):
        mask, quantized_img = classify_svg(
            image_path, fallback_physical_width_mm=physical_width_mm,
            classify_scale=classify_scale,
            enable_context_refinement=enable_context_refinement,
            use_svg_labels=use_svg_labels, verbose=verbose,
        )
    else:
        mask, quantized_img = classify_multicolor_image(
            image_path, physical_width_mm=physical_width_mm,
            enable_context_refinement=enable_context_refinement, verbose=verbose,
        )

    os.makedirs(preview_dir, exist_ok=True)
    save_preview(mask, os.path.join(preview_dir, f"{base_name}_pred.png"))

    if quantized_dir:
        os.makedirs(quantized_dir, exist_ok=True)
        save_quantized(quantized_img, os.path.join(quantized_dir, f"{base_name}_quantized.png"))

    print(f"  -> {base_name}: Done")

def find_svgs_in_dir(folder: str) -> List[str]:
    return sorted(set(glob.glob(os.path.join(folder, "*.svg")) + glob.glob(os.path.join(folder, "*.SVG"))))

def find_images_in_dir(folder: str) -> List[str]:
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(files))

def _batch_worker(kwargs):
    image_path = kwargs['image_path']
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    try:
        if image_path.lower().endswith(".svg"):
            mask, quantized_img = classify_svg(
                image_path,
                fallback_physical_width_mm=kwargs['physical_width_mm'],
                classify_scale=kwargs.get('classify_scale', SVG_CLASSIFY_SCALE),
                enable_context_refinement=kwargs['enable_context'],
                use_svg_labels=kwargs['use_svg_labels'],
                verbose=False
            )
        else:
            mask, quantized_img = classify_multicolor_image(
                image_path,
                physical_width_mm=kwargs['physical_width_mm'],
                enable_context_refinement=kwargs['enable_context'],
                verbose=False
            )

        os.makedirs(kwargs['preview_dir'], exist_ok=True)
        preview_path = os.path.join(kwargs['preview_dir'], f"{base_name}_pred.png")

        h, w = mask.shape[:2]
        preview = np.zeros((h, w, 3), dtype=np.uint8)
        preview[mask == LABEL_FILL] = _PREVIEW_COLOR_FILL
        preview[mask == LABEL_SATIN] = _PREVIEW_COLOR_SATIN
        cv2.imwrite(preview_path, preview, _PNG_SAVE_PARAMS)

        if kwargs['quantized_dir']:
            os.makedirs(kwargs['quantized_dir'], exist_ok=True)
            quant_path = os.path.join(kwargs['quantized_dir'], f"{base_name}_quantized.png")
            cv2.imwrite(quant_path, quantized_img, _PNG_SAVE_PARAMS)

        return {"status": "ok", "base": base_name}
    except Exception as e:
        return {"status": "error", "base": base_name, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="OpenCV Stitch Classifier")
    parser.add_argument("input", help="1 file anh/SVG, HOAC 1 thu muc neu dung --batch")
    parser.add_argument("--out-dir", default="opencv_test", help="Thu muc goc luu preview")
    parser.add_argument("--quantized-dir", default=None, help="Thu muc rieng de luu anh quantized")
    parser.add_argument("--physical-width-mm", type=float, default=DEFAULT_PHYSICAL_WIDTH_MM)
    parser.add_argument("--classify-scale", type=float, default=SVG_CLASSIFY_SCALE)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--svg", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--use-svg-labels", action="store_true", help="Uu tien doc thuoc tinh inkscape:label neu co")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    quantized_dir = args.quantized_dir or os.path.join(out_dir, "quantized")
    enable_context = not args.no_context
    classify_scale = min(max(args.classify_scale, 1e-3), 1.0)

    if args.batch:
        image_files = find_svgs_in_dir(args.input) if args.svg else find_images_in_dir(args.input)
        if not image_files: sys.exit(1)

        n_workers = max(1, (os.cpu_count() or 2) - 1)
        tasks = []
        for img_path in image_files:
            tasks.append({
                'image_path': img_path,
                'preview_dir': out_dir,
                'quantized_dir': quantized_dir,
                'physical_width_mm': args.physical_width_mm,
                'classify_scale': classify_scale,
                'enable_context': enable_context,
                'use_svg_labels': args.use_svg_labels
            })

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_batch_worker, t): t for t in tasks}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                res = future.result()
                if res["status"] != "ok": tqdm.write(f"  [LOI] {res['base']}: {res['error']}")
    else:
        process_one_image(args.input, out_dir, quantized_dir,
                           args.physical_width_mm, enable_context, args.use_svg_labels, args.verbose, classify_scale)

if __name__ == "__main__":
    main()
