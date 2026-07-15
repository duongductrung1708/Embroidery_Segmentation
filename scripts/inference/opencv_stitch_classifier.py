#!/usr/bin/env python3
"""
OpenCV Stitch Classifier - Satin vs Fill (rule-based, KHÔNG train model)
=========================================================================

BẢN CẬP NHẬT: THÊM CONTEXT-AWARE CLASSIFICATION + BATCH MODE (SONG SONG)
- Chuẩn hóa Canvas về kích thước 4200x4800 chống méo hình.
- Đo đạc siêu tốc trên ảnh khổng lồ nhờ kỹ thuật Downscaling ma trận tạm thời.
- LỌC RÁC THÔNG MINH dựa trên Aspect Ratio và Solidity.
- CONTEXT-AWARE CLASSIFICATION: outline satin bao quanh + dính biên thực sự
  mới ép shape bên trong về fill (tránh chồng 2 lớp satin).
- TỐI ƯU HÓA: Chế độ --batch nay đã hỗ trợ xử lý đa tiến trình (Multiprocessing)
  kết hợp thanh tiến trình tqdm, tăng tốc độ chạy hàng loạt lên nhiều lần
  mà GIỮ NGUYÊN 100% logic cốt lõi.
- MỚI: DỰ ĐOÁN TRỰC TIẾP TRÊN FILE SVG (classify_svg) — rasterize từng <path>
  riêng lẻ (thay vì group theo màu từ ảnh PNG đã render) rồi tái sử dụng đúng
  logic hình học (thickness/solidity/aspect ratio) để phân loại satin/fill.

===============================================================================
PATCH (xem ghi chú "PATCH:" bên trong file): sửa lỗi cascade trong
refine_labels_by_context khi gặp NHIỀU LỚP outline satin lồng nhau
(vd hoạ tiết vòng tròn/registration-mark lồng nhau) khiến một số shape satin
đã được phân loại đúng bị ép sai thành fill qua nhiều bước domino.
===============================================================================

===============================================================================
TỐI ƯU TỐC ĐỘ (SPEEDUP PATCH):
- classify_svg gio chay theo 2 GIAI DOAN, nhung QUAN TRONG: giai doan cuoi
  van la RENDER VECTOR (khong resize pixel), nen bien hinh (edge) van MUOT
  y het nhu truoc, KHONG bi rang cua:
    1) "Fast pass" (chi de RA QUYET DINH satin/fill, khong dung de xuat anh):
       rasterize + phan loai hinh hoc (thickness/solidity/aspect
       ratio/context-refinement) tren 1 CANVAS THU NHO (mac dinh ~1/3 canh,
       vd 1400x1600 thay vi 4200x4800 => giam ~9 lan so pixel phai xu ly cho
       MOI path). Day la buoc ton thoi gian nhat truoc day vi cairosvg phai
       render full-canvas RIENG BIET cho tung <path> mot.
    2) Sau khi moi <path> da co 1 nhan quyet dinh (satin/fill, gop tu cac
       shape con cua no theo dien tich, SAU KHI context-refinement da chay),
       to lai mau cho tung <path> theo dung nhan do (do/xanh, giong het cach
       lam Ground-Truth) roi RENDER VECTOR TOAN BO SVG **1 LAN DUY NHAT** o
       DUNG do phan giai chuan (4200x4800). Anh mau nay duoc giai ma nguoc
       lai thanh label_mask - bien hinh la bien VECTOR that (co anti-alias
       tu cairosvg), KHONG phai pixel bi phong to len.
  => Tu N lan render full-res (N = so <path>) giam con ~N lan render o canvas
     NHO (chi de quyet dinh nhan) + 1 lan render full-res DUY NHAT (de xuat
     anh) - nhanh hon rat nhieu ma chat luong bien hinh GIU NGUYEN nhu ban
     goc, khong con hien tuong rang cua do upscale pixel.
- Bo `copy.deepcopy(svg_root)` cho MOI path: thay vao do chi mutate truc tiep
  attrib cua cac <path> element roi RESTORE lai sau khi rasterize xong. Tranh
  chi phi deepcopy toan bo cay XML lap lai hang tram lan.
- Bo buoc `skimage.morphology.skeletonize` trong thickness_mm_from_mask: gia
  tri median_thickness_mm nay TRUOC GIO chi dung de LOG/hien thi (quyet dinh
  phan loai luon dung thickness_max_mm - xem _classify_shape), nen bo di de
  tiet kiem CPU ma khong doi ket qua phan loai.
- save_preview / save_quantized: giam muc nen PNG (IMWRITE_PNG_COMPRESSION)
  de ghi file nhanh hon dang ke tren anh do phan giai lon.
===============================================================================
"""

import argparse
import glob
import os
import sys
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_BACKGROUND = 0
LABEL_FILL = 1
LABEL_SATIN = 2

DEFAULT_PHYSICAL_WIDTH_MM = 80.0

# NGUONG CHUNG DUY NHAT (khong tuning theo tung anh/tung shape nua):
# thickness_mm <= threshold_mm  -> SATIN (mac dinh)
# thickness_mm >  threshold_mm  -> FILL  (mac dinh)
# Chi co 2 truong hop DAC BIET duoc xu ly rieng (xem _classify_shape):
#   1) manh rac/nhieu qua nho (duoi MIN_PHYSICAL_STITCH_MM va dien tich qua nho)
#   2) context-aware refinement o buoc sau (outline satin bao ngoai ep shape con vao fill)
DEFAULT_THRESHOLD_MM = 12.0

MIN_PHYSICAL_STITCH_MM = 0.4
MIN_NOISE_AREA_MM2 = 0.2
CONTEXT_CONTAINMENT_RATIO = 0.5
CONTEXT_TOUCH_DILATION_PX = 3

_PREVIEW_COLOR_FILL = (0, 255, 255)   # Cyan  (hien thi mau VANG that)
_PREVIEW_COLOR_SATIN = (255, 0, 255)  # Magenta

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

DEFAULT_SVG_CANVAS_W = 4200
DEFAULT_SVG_CANVAS_H = 4800

# Canvas dung rieng cho buoc PHAN LOAI HINH HOC (fast pass) cua classify_svg.
# ~1400x1600 vua khop voi MAX_EVAL_SIZE (1500) da dung san trong
# thickness_mm_from_mask, nen khong bi downscale thua 2 lan.
SVG_CLASSIFY_SCALE = 1400.0 / DEFAULT_SVG_CANVAS_W

# Muc nen PNG khi ghi file (0 = khong nen/nhanh nhat, 9 = nen nhieu nhat/cham
# nhat). Dung muc thap de uu tien toc do ghi file tren anh do phan giai lon.
_PNG_SAVE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


class Shape:
    def __init__(self, shape_id: int, contour: np.ndarray, holes: List[np.ndarray]):
        self.id = shape_id
        self.contour = contour
        self.holes = holes
        self.label: Optional[str] = None
        self.global_id: Optional[int] = None


def normalize_to_canvas(img: np.ndarray, target_w: int = 4200, target_h: int = 4800) -> np.ndarray:
    h_orig, w_orig = img.shape[:2]

    if h_orig == target_h and w_orig == target_w:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    scale = min(target_w / w_orig, target_h / h_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if len(resized_img.shape) == 2:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGRA)
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


def load_binary_mask(image_path: str, invert: bool = False, thresh: int = 127) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    if invert:
        binary = 255 - binary
    return binary


def extract_shapes(binary_mask: np.ndarray) -> List[Shape]:
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    shapes: List[Shape] = []
    for i, h in enumerate(hierarchy):
        if h[3] != -1:
            continue
        holes = [contours[j] for j, hj in enumerate(hierarchy) if hj[3] == i]
        if cv2.contourArea(contours[i]) < 4:
            continue
        shapes.append(Shape(shape_id=len(shapes), contour=contours[i], holes=holes))

    return shapes


def shape_to_mask(shape: Shape, canvas_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(canvas_shape, dtype=np.uint8)
    cv2.drawContours(mask, [shape.contour], -1, 1, thickness=cv2.FILLED)
    for hole in shape.holes:
        cv2.drawContours(mask, [hole], -1, 0, thickness=cv2.FILLED)
    return mask


def thickness_mm_from_mask(mask: np.ndarray, pixel_to_mm: float) -> Tuple[float, float]:
    if mask.sum() == 0:
        return 0.0, 0.0

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

    # TOI UU: khong con dung skimage.morphology.skeletonize de tinh median
    # thickness nua (rat ton CPU tren mask lon). Gia tri nay TRUOC GIO chi
    # phuc vu MUC DICH LOG/hien thi - quyet dinh phan loai luon dung
    # thickness_max_mm (xem _classify_shape) nen viec bo skeletonize KHONG
    # lam thay doi ket qua phan loai cuoi cung.
    median_thickness_mm = max_thickness_mm

    return median_thickness_mm, max_thickness_mm


def _classify_shape(shape: Shape, mask: np.ndarray, is_outer_candidate: bool,
                     canvas_area: float, pixel_to_mm: float,
                     threshold_mm: float) -> Tuple[str, Dict]:

    hull = cv2.convexHull(shape.contour)
    hull_area_px = cv2.contourArea(hull)

    true_area_px = float(np.count_nonzero(mask))
    solidity = true_area_px / hull_area_px if hull_area_px > 0 else 1.0

    rect = cv2.minAreaRect(shape.contour)
    short_side, long_side = sorted([rect[1][0], rect[1][1]])
    aspect_ratio = long_side / short_side if short_side > 0 else 100.0

    is_hollow = len(shape.holes) > 0
    thickness_median_mm, thickness_max_mm = thickness_mm_from_mask(mask, pixel_to_mm)
    thickness_mm = thickness_max_mm

    is_outermost = (
        is_outer_candidate
        and canvas_area > 0
        and (hull_area_px / canvas_area) > 0.4
    )

    details = {
        "thickness_mm": thickness_median_mm,
        "thickness_max_mm": thickness_max_mm,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "is_hollow": is_hollow,
        "is_outermost": is_outermost
    }

    # -------------------------------------------------------------------
    # DAC BIET DUY NHAT con giu lai: loc RAC/NHIEU that su (manh vun sieu
    # mong ma may khong the thieu duoc VA dien tich khong dang ke). Neu du
    # dien tich thi KHONG con coi la rac nua - se roi xuong dung quy tac
    # chung ben duoi (thickness cang nho thi cang chac chan la satin).
    # -------------------------------------------------------------------
    if thickness_mm < MIN_PHYSICAL_STITCH_MM:
        area_mm2 = true_area_px * (pixel_to_mm ** 2)
        if area_mm2 < MIN_NOISE_AREA_MM2:
            return "noise", details

    # -------------------------------------------------------------------
    # QUY TAC CHUNG DUY NHAT, AP DUNG DONG NHAT CHO MOI SHAPE - khong con
    # phan biet theo solidity / aspect-ratio / is_hollow / is_outermost nua:
    #   thickness_mm <= threshold_mm (mac dinh 20mm) -> SATIN
    #   thickness_mm >  threshold_mm                 -> FILL
    # -------------------------------------------------------------------
    label = "satin" if thickness_mm <= threshold_mm else "fill"

    return label, details


def classify_binary_mask(binary_mask: np.ndarray,
                          physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                          threshold_mm: float = DEFAULT_THRESHOLD_MM,
                          verbose: bool = False) -> Tuple[List[Shape], np.ndarray]:
    h, w = binary_mask.shape[:2]
    canvas_area = float(w * h)
    pixel_to_mm = physical_width_mm / max(float(w), 1.0)

    shapes = extract_shapes(binary_mask)
    if not shapes:
        return [], np.zeros((h, w), dtype=np.uint8)

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
        label, details = _classify_shape(
            shape, mask,
            is_outer_candidate=(shape.id == outer_border_id),
            canvas_area=canvas_area,
            pixel_to_mm=pixel_to_mm,
            threshold_mm=threshold_mm
        )
        shape.label = label

        if label in ["satin", "fill"]:
            label_value = LABEL_SATIN if label == "satin" else LABEL_FILL
            label_mask[mask == 1] = label_value

        if verbose:
            print(f"  Shape {shape.id}: med={details['thickness_mm']:.3f}mm, "
                  f"max={details['thickness_max_mm']:.3f}mm, "
                  f"aspect={details['aspect_ratio']:.1f}, solidity={details['solidity']:.2f} -> {label.upper()}")

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
    if len(shape.holes) == 0:
        return None
    outer_mask = np.zeros(canvas_shape, dtype=np.uint8)
    cv2.drawContours(outer_mask, [shape.contour], -1, 1, thickness=cv2.FILLED)
    interior_mask = outer_mask & (1 - shape_mask)
    if interior_mask.sum() == 0:
        return None
    return interior_mask


def _dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    if dilation_px <= 0:
        return mask
    kernel = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def refine_labels_by_context(records: List[_GlobalShapeRecord],
                              label_mask: np.ndarray,
                              canvas_shape: Tuple[int, int],
                              containment_ratio: float = CONTEXT_CONTAINMENT_RATIO,
                              touch_dilation_px: int = CONTEXT_TOUCH_DILATION_PX,
                              verbose: bool = False) -> np.ndarray:
    outline_candidates = [r for r in records
                           if r.shape.label == "satin" and len(r.shape.holes) > 0]

    # PATCH: Xu ly outline TU LON DEN NHO (ngoai vao trong). Voi cac hoa tiet co
    # NHIEU LOP outline long nhau (vd vong tron/registration-mark dong tam), 1
    # outline nho hon vua la "outline" cho lop ben trong no, vua co the la
    # "other" bi 1 outline lon hon ep thanh fill. Neu xu ly theo dung thu tu
    # xuat hien trong SVG (khong sap xep), viec ep nhan co the CASCADE sai: 1
    # outline da bi ep thanh fill van tiep tuc duoc dung de ep cac shape khac o
    # vong lap sau, gay sai day chuyen (domino). Sap xep ngoai vao trong + bo
    # qua outline da bi ep thanh fill (xem check ben duoi) dam bao outline lon
    # nhat luon duoc uu tien xu ly cac lop ben trong no truoc, chi 1 lan.
    outline_candidates = sorted(
        outline_candidates,
        key=lambda r: cv2.contourArea(cv2.convexHull(r.shape.contour)),
        reverse=True,
    )

    for outline in outline_candidates:
        if outline.shape.label != "satin":
            # PATCH: Outline nay da bi 1 outline khac (lon hon, xu ly truoc do
            # trong cung 1 pass) ep thanh fill -> no khong con dieu kien la 1
            # outline satin nua, bo qua de tranh ep sai day chuyen (cascade)
            # len cac shape con nam trong no.
            if verbose:
                print(f"  [Context] Outline(global_id={outline.shape.global_id}) da bi ep "
                      f"thanh FILL boi 1 outline lon hon -> BO QUA, khong dung no de "
                      f"ep tiep cac shape ben trong")
            continue

        interior_mask = _build_interior_mask(outline.shape, outline.mask, canvas_shape)
        if interior_mask is None:
            continue

        outline_mask_dilated = _dilate_mask(outline.mask, touch_dilation_px)

        for other in records:
            if other is outline:
                continue
            if other.shape.label != "satin":
                continue
            if other.area_px == 0:
                continue

            overlap_interior = int(np.count_nonzero(interior_mask & other.mask))
            interior_ratio = overlap_interior / other.area_px
            if interior_ratio < containment_ratio:
                continue

            touch_overlap = int(np.count_nonzero(outline_mask_dilated & other.mask))
            if touch_overlap == 0:
                if verbose:
                    print(f"  [Context] Shape(global_id={other.shape.global_id}) nam trong "
                          f"outline Shape(global_id={outline.shape.global_id}) nhung KHONG "
                          f"dinh bien -> giu nguyen SATIN")
                continue

            if verbose:
                print(f"  [Context] Shape(global_id={other.shape.global_id}) dinh bien voi "
                      f"outline Shape(global_id={outline.shape.global_id}) "
                      f"(interior_ratio={interior_ratio:.2f}) -> ep SATIN thanh FILL")
            other.shape.label = "fill"
            label_mask[other.mask == 1] = LABEL_FILL

    return label_mask


# ---------------------------------------------------------------------------
# Hau xu ly
# ---------------------------------------------------------------------------
def fill_unlabeled_gaps(label_mask: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    unlabeled = foreground_mask & (label_mask == 0)
    if not unlabeled.any():
        return label_mask

    labeled = foreground_mask & (label_mask != 0)
    if not labeled.any():
        return label_mask

    from scipy.ndimage import distance_transform_edt
    _, indices = distance_transform_edt(~labeled, return_indices=True)

    filled = label_mask.copy()
    ys, xs = np.where(unlabeled)
    nearest_ys = indices[0][ys, xs]
    nearest_xs = indices[1][ys, xs]
    filled[ys, xs] = label_mask[nearest_ys, nearest_xs]

    return filled


def quantize_colors_fast(img_bgr: np.ndarray, valid_mask: np.ndarray, step: int = 48) -> np.ndarray:
    quantized = img_bgr.copy()
    rounded = (np.round(quantized[valid_mask] / step) * step).clip(0, 255).astype(np.uint8)
    quantized[valid_mask] = rounded
    return quantized


def classify_multicolor_image(image_path: str,
                               physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                               threshold_mm: float = DEFAULT_THRESHOLD_MM,
                               color_tolerance: int = 15,
                               quantize: bool = True,
                               alpha_threshold: int = 10,
                               min_region_pixels: int = 5,
                               enable_context_refinement: bool = True,
                               verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

    if verbose:
        print(">> Dang dong bo va ep Canvas ve khung chuan (4200x4800)...")
    img = normalize_to_canvas(img, target_w=4200, target_h=4800)

    img_bgr = img[:, :, :3]
    alpha = img[:, :, 3]
    is_background = alpha < alpha_threshold

    h, w = img_bgr.shape[:2]
    canvas_shape = (h, w)
    label_mask = np.zeros((h, w), dtype=np.uint8)

    working_img = quantize_colors_fast(img_bgr, ~is_background, step=48) if quantize else img_bgr.copy()
    working_img[is_background] = (0, 0, 0)

    pixels = working_img[~is_background].reshape(-1, 3)
    if pixels.size == 0:
        return label_mask, working_img

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
        if n_pixels < min_region_pixels:
            continue

        if verbose:
            print(f"Xu ly mau BGR={tuple(int(c) for c in color)}  ({n_pixels}px) ...")

        shapes, sub_label_mask = classify_binary_mask(
            color_mask, physical_width_mm=physical_width_mm,
            threshold_mm=threshold_mm, verbose=verbose,
        )
        label_mask[sub_label_mask > 0] = sub_label_mask[sub_label_mask > 0]

        for shape in shapes:
            if shape.label not in ("satin", "fill"):
                continue
            shape.global_id = global_id_counter
            global_id_counter += 1
            shape_mask = shape_to_mask(shape, canvas_shape)
            area_px = int(np.count_nonzero(shape_mask))
            global_records.append(_GlobalShapeRecord(shape, shape_mask, area_px))

    if enable_context_refinement and global_records:
        if verbose:
            print(">> Dang chay Context-aware refinement (outline satin -> fill ben trong)...")
        label_mask = refine_labels_by_context(
            global_records, label_mask, canvas_shape, verbose=verbose
        )

    foreground_mask = ~is_background
    label_mask = fill_unlabeled_gaps(label_mask, foreground_mask)

    return label_mask, working_img


# ---------------------------------------------------------------------------
# Du doan truc tiep tren file SVG (khong can rasterize thanh 1 anh PNG gop
# mau roi group theo mau nua - moi <path> duoc rasterize rieng va cham diem
# hinh hoc doc lap, tai su dung dung logic classify_binary_mask).
#
# TOI UU: buoc rasterize+phan loai tung path chay tren 1 CANVAS THU NHO
# (SVG_CLASSIFY_SCALE) de giam chi phi cairosvg + numpy nhieu lan, sau do
# label_mask thu nho duoc phong to (INTER_NEAREST) ve dung canvas chuan.
# ---------------------------------------------------------------------------
def _tag(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def _render_single_path_mask(path_elems: List[ET.Element], path_index: int,
                              canvas_w: int, canvas_h: int, root: ET.Element) -> np.ndarray:
    """Rasterize DUY NHAT 1 <path> (theo thu tu path_elems), an het cac path
    con lai, tra ve mask nhi phan (0/1) cua rieng path do.

    TOI UU: KHONG deepcopy toan bo cay XML nua (rat ton CPU/RAM neu goi lap
    lai hang tram lan). Thay vao do, mutate truc tiep attrib cua tung
    <path> element, render, roi RESTORE lai attrib goc ngay sau do (an toan
    vi ham nay luon chay TUAN TU trong 1 tien trinh, khong co race-condition).
    """
    if cairosvg is None:
        raise ImportError("Can cai dat 'cairosvg' de rasterize SVG (pip install cairosvg).")

    saved_attribs = [dict(el.attrib) for el in path_elems]
    try:
        for i, el in enumerate(path_elems):
            if i == path_index:
                el.set("fill", "#000000")
                el.set("fill-opacity", "1")
                el.set("stroke", "none")
                el.attrib.pop("style", None)
                el.attrib.pop("display", None)
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
    """Sau khi MOI <path> da co 1 nhan quyet dinh (satin/fill/None), to lai
    mau cho tung path theo dung nhan (do=satin, xanh duong=fill; None = an
    di) roi RENDER VECTOR TOAN BO SVG 1 LAN DUY NHAT o day phan giai chuan.
    Day chinh la ky thuat GIONG HET voi render_gt_label_mask (dung de tao
    Ground-Truth) - bien hinh la bien VECTOR that (co anti-alias), KHONG
    phai ket qua resize/phong to tu 1 mask pixel nho => KHONG bi rang cua.

    Tra ve (label_mask, foreground_mask) o dung canvas_w x canvas_h.
    """
    saved_attribs = [dict(el.attrib) for el in path_elems]
    try:
        for el, label in zip(path_elems, path_labels):
            if label is None:
                # Path nay khong co shape hop le nao (toan bo bi loc la rac/
                # nhieu o buoc fast-pass) -> an di, khong to mau, khong
                # duoc tinh vao label_mask cuoi cung.
                el.set("display", "none")
            else:
                color = "#FF0000" if label == "satin" else "#0000FF"
                el.set("fill", color)
                el.set("fill-opacity", "1")
                el.set("stroke", "none")
                el.attrib.pop("style", None)
                el.attrib.pop("display", None)

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


def classify_svg(svg_path: str,
                  physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                  threshold_mm: float = DEFAULT_THRESHOLD_MM,
                  canvas_w: int = DEFAULT_SVG_CANVAS_W,
                  canvas_h: int = DEFAULT_SVG_CANVAS_H,
                  classify_scale: float = SVG_CLASSIFY_SCALE,
                  min_region_pixels: int = 5,
                  enable_context_refinement: bool = True,
                  pre_parsed_root: Optional[ET.Element] = None,
                  verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Du doan satin/fill TRUC TIEP tu file SVG (thay vi tu anh PNG da render + gop mau).

    Moi <path> trong SVG duoc rasterize RIENG LE thanh 1 mask nhi phan, sau do
    duoc dua qua dung pipeline hinh hoc hien co (classify_binary_mask:
    contour/hull/solidity/aspect-ratio/distance-transform) de quyet dinh satin
    hay fill. Context-aware refinement (outline satin bao quanh ep shape con
    vao fill) van duoc ap dung tren toan bo cac shape gom duoc tu tat ca cac path.

    TOI UU TOC DO: buoc rasterize + phan loai hinh hoc (fast pass) chay tren
    1 CANVAS THU NHO (kich thuoc = canvas_w/h * classify_scale, mac dinh ~1/3
    canh) - giam dang ke so pixel phai xu ly cho MOI path (voi scale mac
    dinh, giam khoang 9 lan). Fast pass nay CHI DUNG DE RA QUYET DINH nhan
    satin/fill cho tung <path> (gop tu cac shape con cua no theo dien tich),
    KHONG dung truc tiep de xuat anh. Sau khi co nhan cho moi path, anh
    label_mask CUOI CUNG duoc tao bang cach RENDER VECTOR TOAN BO SVG 1 LAN
    DUY NHAT o dung canvas chuan (xem _render_labeled_paths_full_res) - nen
    bien hinh van MUOT nhu ban goc (co anti-alias tu cairosvg), KHONG bi
    rang cua nhu khi resize/phong to 1 mask pixel nho len.

    Tra ve (label_mask, rendered_img_bgr) - rendered_img_bgr la ban render mau
    day du cua SVG (dung thay cho "quantized image" trong ket qua truoc day,
    chi de tham khao/hien thi), luon o DUNG canvas_w x canvas_h.
    """
    if cairosvg is None:
        raise ImportError("Can cai dat 'cairosvg' de doc va rasterize file SVG (pip install cairosvg).")

    if pre_parsed_root is not None:
        root = pre_parsed_root
    else:
        tree = ET.parse(svg_path)
        root = tree.getroot()

    path_elems = [el for el in root.iter() if _tag(el) == "path"]

    # Anh mau day du (hien thi) - render 1 LAN DUY NHAT o full-res, doc thang
    # tu file (nhanh hon serialize root -> string).
    full_png = cairosvg.svg2png(url=svg_path, output_width=canvas_w, output_height=canvas_h,
                                 background_color="#FFFFFF", unsafe=True)
    full_rgb = np.array(Image.open(BytesIO(full_png)).convert("RGB"))
    rendered_img_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)

    label_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    if not path_elems:
        return label_mask, rendered_img_bgr

    # ---- GIAI DOAN 1 (fast pass): phan loai hinh hoc tren CANVAS THU NHO,
    # chi de RA QUYET DINH nhan cho tung path - khong dung de xuat anh. ----
    small_w = max(1, int(round(canvas_w * classify_scale)))
    small_h = max(1, int(round(canvas_h * classify_scale)))
    small_canvas_shape = (small_h, small_w)

    label_mask_small = np.zeros((small_h, small_w), dtype=np.uint8)  # buffer bat buoc cho refine_labels_by_context, khong dung de xuat anh
    global_records: List[_GlobalShapeRecord] = []
    shape_path_idx: List[int] = []
    global_id_counter = 0

    path_iter = enumerate(path_elems)
    if verbose:
        path_iter = tqdm(list(path_iter), desc="Rasterizing paths (fast pass)", unit="path", leave=False)

    for idx, _ in path_iter:
        path_mask01 = _render_single_path_mask(path_elems, idx, small_w, small_h, root)
        n_pixels = int(path_mask01.sum())
        if n_pixels < min_region_pixels:
            continue

        binary_mask_255 = (path_mask01 * 255).astype(np.uint8)
        shapes, sub_label_mask = classify_binary_mask(
            binary_mask_255, physical_width_mm=physical_width_mm,
            threshold_mm=threshold_mm, verbose=verbose,
        )
        label_mask_small[sub_label_mask > 0] = sub_label_mask[sub_label_mask > 0]

        for shape in shapes:
            if shape.label not in ("satin", "fill"):
                continue
            shape.global_id = global_id_counter
            global_id_counter += 1
            shape_mask_small = shape_to_mask(shape, small_canvas_shape)
            area_px = int(np.count_nonzero(shape_mask_small))
            global_records.append(_GlobalShapeRecord(shape, shape_mask_small, area_px))
            shape_path_idx.append(idx)

        if verbose:
            print(f"Path #{idx}: {n_pixels}px -> {len(shapes)} shape(s)")

    if enable_context_refinement and global_records:
        if verbose:
            print(">> Dang chay Context-aware refinement (outline satin -> fill ben trong)...")
        refine_labels_by_context(
            global_records, label_mask_small, small_canvas_shape, verbose=verbose
        )

    # Gop nhan tu cap SHAPE len cap PATH: 1 <path> co the tao ra nhieu shape
    # (vd path co nhieu subpath) - nhan cuoi cung cua ca path la nhan chiem
    # DIEN TICH LON HON (satin vs fill), tinh SAU KHI context-refinement da
    # chay xong (vi no co the doi shape.label tu satin -> fill).
    path_satin_area = [0] * len(path_elems)
    path_fill_area = [0] * len(path_elems)
    for rec, pidx in zip(global_records, shape_path_idx):
        if rec.shape.label == "satin":
            path_satin_area[pidx] += rec.area_px
        elif rec.shape.label == "fill":
            path_fill_area[pidx] += rec.area_px

    path_labels: List[Optional[str]] = []
    for idx in range(len(path_elems)):
        s_area, f_area = path_satin_area[idx], path_fill_area[idx]
        if s_area == 0 and f_area == 0:
            path_labels.append(None)  # ca path chi toan rac/nhieu -> an di
        else:
            path_labels.append("satin" if s_area >= f_area else "fill")

    # ---- GIAI DOAN 2: RENDER VECTOR TOAN BO SVG 1 LAN DUY NHAT o day phan
    # giai chuan, to mau tung path theo dung nhan vua quyet dinh - cho ra
    # label_mask CO BIEN VECTOR MUOT (giong het chat luong Ground-Truth,
    # khong bi rang cua). ----
    label_mask, foreground_full = _render_labeled_paths_full_res(
        path_elems, path_labels, root, canvas_w, canvas_h
    )
    label_mask = fill_unlabeled_gaps(label_mask, foreground_full)

    return label_mask, rendered_img_bgr


# ---------------------------------------------------------------------------
# Xuat anh
# ---------------------------------------------------------------------------
def save_preview(label_mask: np.ndarray, output_path: str) -> None:
    h, w = label_mask.shape[:2]
    preview = np.zeros((h, w, 3), dtype=np.uint8)
    preview[label_mask == LABEL_FILL] = _PREVIEW_COLOR_FILL
    preview[label_mask == LABEL_SATIN] = _PREVIEW_COLOR_SATIN
    cv2.imwrite(output_path, preview, _PNG_SAVE_PARAMS)


def save_quantized(working_img_bgr: np.ndarray, output_path: str) -> None:
    cv2.imwrite(output_path, working_img_bgr, _PNG_SAVE_PARAMS)


# ---------------------------------------------------------------------------
# Xu ly 1 anh: luon tach rieng preview va quantized vao 2 thu muc khac nhau
# ---------------------------------------------------------------------------
def process_one_image(image_path: str, preview_dir: str, quantized_dir: str,
                       physical_width_mm: float, threshold_mm: float,
                       enable_context_refinement: bool, verbose: bool,
                       classify_scale: float = SVG_CLASSIFY_SCALE) -> None:
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if image_path.lower().endswith(".svg"):
        mask, quantized_img = classify_svg(
            image_path, physical_width_mm=physical_width_mm, threshold_mm=threshold_mm,
            classify_scale=classify_scale,
            enable_context_refinement=enable_context_refinement, verbose=verbose,
        )
    else:
        mask, quantized_img = classify_multicolor_image(
            image_path, physical_width_mm=physical_width_mm, threshold_mm=threshold_mm,
            enable_context_refinement=enable_context_refinement, verbose=verbose,
        )

    os.makedirs(preview_dir, exist_ok=True)
    save_preview(mask, os.path.join(preview_dir, f"{base_name}_pred.png"))

    if quantized_dir:
        os.makedirs(quantized_dir, exist_ok=True)
        save_quantized(quantized_img, os.path.join(quantized_dir, f"{base_name}_quantized.png"))

    n_bg = int((mask == LABEL_BACKGROUND).sum())
    n_fill = int((mask == LABEL_FILL).sum())
    n_satin = int((mask == LABEL_SATIN).sum())
    print(f"  -> {base_name}: bg={n_bg}  fill={n_fill}  satin={n_satin}")


def find_images_in_dir(folder: str) -> List[str]:
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(files))


def find_svgs_in_dir(folder: str) -> List[str]:
    return sorted(set(glob.glob(os.path.join(folder, "*.svg")) +
                       glob.glob(os.path.join(folder, "*.SVG"))))


# ---------------------------------------------------------------------------
# Worker cho xử lý đa luồng (Batch Processing)
# ---------------------------------------------------------------------------
def _batch_worker(kwargs):
    image_path = kwargs['image_path']
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    try:
        if image_path.lower().endswith(".svg"):
            mask, quantized_img = classify_svg(
                image_path,
                physical_width_mm=kwargs['physical_width_mm'],
                threshold_mm=kwargs['threshold_mm'],
                classify_scale=kwargs.get('classify_scale', SVG_CLASSIFY_SCALE),
                enable_context_refinement=kwargs['enable_context'],
                verbose=False
            )
        else:
            mask, quantized_img = classify_multicolor_image(
                image_path,
                physical_width_mm=kwargs['physical_width_mm'],
                threshold_mm=kwargs['threshold_mm'],
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

        n_bg = int((mask == LABEL_BACKGROUND).sum())
        n_fill = int((mask == LABEL_FILL).sum())
        n_satin = int((mask == LABEL_SATIN).sum())

        return {"status": "ok", "base": base_name, "bg": n_bg, "fill": n_fill, "satin": n_satin}
    except Exception as e:
        return {"status": "error", "base": base_name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="OpenCV Stitch Classifier - Satin vs Fill (rule-based)")
    parser.add_argument("input", help="1 file anh/SVG, HOAC 1 thu muc neu dung --batch")
    parser.add_argument("--out-dir", default="opencv_test",
                         help="Thu muc goc luu preview (mac dinh: opencv_test)")
    parser.add_argument("--quantized-dir", default=None,
                         help="Thu muc rieng de luu anh quantized "
                              "(mac dinh: <out-dir>/quantized)")
    parser.add_argument("--physical-width-mm", type=float, default=DEFAULT_PHYSICAL_WIDTH_MM)
    parser.add_argument("--threshold-mm", type=float, default=DEFAULT_THRESHOLD_MM)
    parser.add_argument("--classify-scale", type=float, default=SVG_CLASSIFY_SCALE,
                         help="Ty le canvas dung cho buoc PHAN LOAI HINH HOC cua SVG "
                              "(0 < scale <= 1). Cang nho cang nhanh nhung cang mat "
                              "chi tiet hinh hoc nho. Mac dinh: %(default).3f")
    parser.add_argument("--batch", action="store_true",
                         help="Neu bat: 'input' la 1 THU MUC, quet toan bo anh ben trong")
    parser.add_argument("--svg", action="store_true",
                         help="Neu bat: xu ly file/thu muc SVG thay vi anh PNG/JPG "
                              "(du doan truc tiep tren <path> cua SVG)")
    parser.add_argument("--no-context", action="store_true",
                         help="Tat context-aware refinement (outline satin -> fill)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    quantized_dir = args.quantized_dir or os.path.join(out_dir, "quantized")
    enable_context = not args.no_context
    classify_scale = min(max(args.classify_scale, 1e-3), 1.0)

    if args.batch:
        image_files = find_svgs_in_dir(args.input) if args.svg else find_images_in_dir(args.input)
        if not image_files:
            print(f"Khong tim thay {'SVG' if args.svg else 'anh'} nao trong {args.input}")
            sys.exit(1)

        n_workers = max(1, (os.cpu_count() or 2) - 1)
        print(f"Tim thay {len(image_files)} file. Bat dau xu ly hang loat ({n_workers} tien trinh song song)...")

        tasks = []
        for img_path in image_files:
            tasks.append({
                'image_path': img_path,
                'preview_dir': out_dir,
                'quantized_dir': quantized_dir,
                'physical_width_mm': args.physical_width_mm,
                'threshold_mm': args.threshold_mm,
                'classify_scale': classify_scale,
                'enable_context': enable_context
            })

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_batch_worker, t): t for t in tasks}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Dang xu ly", unit="file"):
                res = future.result()
                if res["status"] == "ok":
                    tqdm.write(f"  -> {res['base']}: bg={res['bg']}  fill={res['fill']}  satin={res['satin']}")
                else:
                    tqdm.write(f"  [LOI] {res['base']}: {res['error']}")

        print(f"\nHoan tat. Preview -> {out_dir}/   Quantized -> {quantized_dir}/")
    else:
        process_one_image(args.input, out_dir, quantized_dir,
                           args.physical_width_mm, args.threshold_mm,
                           enable_context, args.verbose, classify_scale)


if __name__ == "__main__":
    main()