#!/usr/bin/env python3
"""
OpenCV Stitch Classifier - Satin vs Fill (rule-based, KHÔNG train model)
=========================================================================

BẢN CẬP NHẬT: THÊM CONTEXT-AWARE CLASSIFICATION
- Chuẩn hóa Canvas về kích thước 4200x4800 chống méo hình.
- Đo đạc siêu tốc trên ảnh khổng lồ nhờ kỹ thuật Downscaling ma trận tạm thời.
- LỌC RÁC THÔNG MINH: Bảo vệ tuyệt đối chữ I, nét mảnh và dấu chấm nhỏ dựa trên
  Aspect Ratio và Solidity, tiêu diệt triệt để các cụm nhiễu răng cưa.
- KHÔNG PHỤ THUỘC MÀU SẮC: Xóa bỏ cờ is_bg_color gây nhiễu, phân loại 100% dựa
  vào giới hạn vật lý của máy thêu.
- MỚI: CONTEXT-AWARE CLASSIFICATION (post-processing theo quan hệ giữa các shape)
  Sau khi phân loại độc lập từng shape, hệ thống xây dựng quan hệ "tiếp xúc / bao quanh"
  (adjacency) giữa các shape (kể cả khác màu). Nếu một shape SATIN đóng vai trò
  đường viền (outline) bao quanh một shape khác, shape bên trong sẽ bị ép về FILL,
  tránh chồng hai lớp satin lên nhau (đúng theo cách digitizing software xây dựng object).
"""

import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_BACKGROUND = 0
LABEL_FILL = 1
LABEL_SATIN = 2

DEFAULT_PHYSICAL_WIDTH_MM = 80.0
DEFAULT_THRESHOLD_MM = 2.0
OUTER_BORDER_MAX_THICKNESS_MM = 8.0

# Ngưỡng vật lý tối thiểu. Dưới mức này (vd: viền nhiễu) -> Có nguy cơ là Rác
MIN_PHYSICAL_STITCH_MM = 0.4

# Ngưỡng overlap để coi 1 shape là "nằm bên trong" vùng do outline bao quanh
# (dùng để xác định B có thuộc "lòng trong" của A hay không, KHÔNG dùng để quyết định đổi nhãn)
CONTEXT_CONTAINMENT_RATIO = 0.5

# Số pixel dilate để kiểm tra 2 shape có THỰC SỰ chạm/dính biên nhau hay không.
# Nếu giữa 2 satin có 1 lớp fill ngăn cách dày hơn mức này thì sẽ KHÔNG bị coi là dính nhau.
CONTEXT_TOUCH_DILATION_PX = 3

_PREVIEW_COLOR_FILL = (0, 255, 255)   # Cyan  (Màu Vàng trên ảnh Preview)
_PREVIEW_COLOR_SATIN = (255, 0, 255)  # Magenta (Màu Hồng trên ảnh Preview)


class Shape:
    """Container lưu trữ dữ liệu của 1 contour ngoài cùng và các lỗ bên trong nó."""
    def __init__(self, shape_id: int, contour: np.ndarray, holes: List[np.ndarray]):
        self.id = shape_id
        self.contour = contour
        self.holes = holes
        self.label: Optional[str] = None
        # global_id: id duy nhất trên toàn ảnh (khác các shape.id nội bộ theo từng màu)
        self.global_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Bước 0: Chuẩn hóa Canvas (Letterbox)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Bước 1 & 2 & 3: Đọc Mask, Tìm Contour, Tính độ dày
# ---------------------------------------------------------------------------
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
    median_thickness_mm = max_thickness_mm

    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(eval_mask.astype(bool))
        skeleton_distances = dist[np.where(skeleton)]
        if len(skeleton_distances) > 0:
            median_thickness_mm = (np.median(skeleton_distances) * 2.0) * actual_pixel_to_mm
    except ImportError:
        pass

    return median_thickness_mm, max_thickness_mm


# ---------------------------------------------------------------------------
# Bước 4: Áp dụng Quy Tắc (phân loại độc lập từng shape)
# ---------------------------------------------------------------------------
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

    label = None

    # =====================================================================
    # LUẬT 0: LỌC RÁC KỸ THUẬT SỐ & BẢO VỆ CHI TIẾT SIÊU MẢNH
    # =====================================================================
    if thickness_mm < MIN_PHYSICAL_STITCH_MM:
        # Cứu sống 1: Dấu chấm, dấu phẩy, hình vuông nhỏ (Đặc ruột)
        if solidity >= 0.75:
            area_mm2 = true_area_px * (pixel_to_mm ** 2)
            if area_mm2 >= 0.2:
                label = "satin"
            else:
                return "noise", details

        # Cứu sống 2: Chữ I, chữ l, nét gạch (Dài và hẹp)
        elif aspect_ratio > 2.5:
            label = "satin"

        # Không đặc, không dài -> Chắc chắn là rác răng cưa/góc nhọn
        else:
            return "noise", details

    # =====================================================================
    # 5 QUY TẮC PHÂN LOẠI CHÍNH TỪ DIGITIZER
    # =====================================================================
    if label is None:
        # 1. Viền ngoài cùng -> SATIN
        if is_outermost and is_hollow and thickness_mm <= OUTER_BORDER_MAX_THICKNESS_MM:
            label = "satin"

        # 2. Hình có lỗ (Chữ O, A, D, R...) -> SATIN nếu mỏng, FILL nếu dày
        elif is_hollow:
            label = "satin" if thickness_mm <= (threshold_mm * 1.5) else "fill"

        # 3. Nét chữ gạch/ngoằn ngoèo không lỗ (S, C, M, I, l...)
        elif not is_hollow and (solidity < 0.75 or aspect_ratio > 2.5):
            label = "satin" if thickness_mm <= (threshold_mm * 2.0) else "fill"

        # 4. Khối đặc ruột (Dấu chấm 'i', mảng nền đặc)
        elif solidity >= 0.75:
            label = "satin" if thickness_mm <= (threshold_mm * 1.5) else "fill"

        # 5. Fallback
        else:
            label = "satin" if thickness_mm <= (threshold_mm * 1.5) else "fill"

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
# MỚI: Context-aware Classification (post-processing theo quan hệ shape)
# ---------------------------------------------------------------------------
class _GlobalShapeRecord:
    """Gói 1 shape + mask toàn cục (kích thước bằng canvas) + label hiện tại."""
    def __init__(self, shape: Shape, mask: np.ndarray, area_px: int):
        self.shape = shape
        self.mask = mask          # mask nhị phân (0/1), kích thước = canvas
        self.area_px = area_px    # số pixel của shape (để tính tỉ lệ containment)


def _build_interior_mask(shape: Shape, shape_mask: np.ndarray,
                          canvas_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Trả về mask vùng "bên trong" của một shape có lỗ (hole), tức là
    outer_contour_area - shape_mask (chính là vùng lỗ, nơi có thể chứa shape khác).
    Nếu shape không có hole -> trả về None.
    """
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
    """
    Bước refine sau khi đã có kết quả phân loại ban đầu cho TẤT CẢ shape
    (kể cả từ các màu khác nhau trong ảnh multicolor).

    Quy tắc (ĐÃ SỬA THEO YÊU CẦU):
    Chỉ đổi nhãn khi 2 điều kiện sau đúng ĐỒNG THỜI:
      1) B nằm (phần lớn) trong vùng lỗ mà A (outline, satin, có hole) bao quanh
         -> xác định A đóng vai trò đường viền của B.
      2) A và B THỰC SỰ chạm/dính biên nhau (không có khoảng cách, không bị 1 lớp
         fill nào khác ngăn cách ở giữa) -> kiểm tra bằng cách dilate mask A vài
         pixel rồi giao với mask B.

    Nếu 2 satin bị ngăn cách bởi 1 vùng fill ở giữa (không dính nhau trực tiếp),
    thì KHÔNG có gì thay đổi, kể cả khi B nằm lọt bên trong A về mặt hình học.

    Khi cả 2 điều kiện đúng và B đang là "satin" -> ép B về "fill".
    Không đổi các shape đã là "fill" hoặc "noise".
    """
    outline_candidates = [r for r in records
                           if r.shape.label == "satin" and len(r.shape.holes) > 0]

    for outline in outline_candidates:
        interior_mask = _build_interior_mask(outline.shape, outline.mask, canvas_shape)
        if interior_mask is None:
            continue

        # Vùng biên của outline được nới rộng thêm vài pixel để bắt các điểm
        # thực sự tiếp xúc (kể cả khi có sai số răng cưa/anti-alias nhỏ).
        outline_mask_dilated = _dilate_mask(outline.mask, touch_dilation_px)

        for other in records:
            if other is outline:
                continue
            if other.shape.label != "satin":
                continue  # chỉ cần ép satin -> fill, các nhãn khác giữ nguyên
            if other.area_px == 0:
                continue

            # Điều kiện 1: B có nằm trong vùng do A bao quanh không?
            overlap_interior = int(np.count_nonzero(interior_mask & other.mask))
            interior_ratio = overlap_interior / other.area_px
            if interior_ratio < containment_ratio:
                continue

            # Điều kiện 2: A và B có THỰC SỰ dính biên nhau không?
            touch_overlap = int(np.count_nonzero(outline_mask_dilated & other.mask))
            if touch_overlap == 0:
                # Không dính nhau (có khoảng cách / bị fill khác ngăn cách) -> bỏ qua
                if verbose:
                    print(f"  [Context] Shape(global_id={other.shape.global_id}) nằm trong "
                          f"outline Shape(global_id={outline.shape.global_id}) nhưng KHÔNG "
                          f"dính biên -> giữ nguyên SATIN")
                continue

            if verbose:
                print(f"  [Context] Shape(global_id={other.shape.global_id}) dính biên trực tiếp "
                      f"với outline Shape(global_id={outline.shape.global_id}) "
                      f"(interior_ratio={interior_ratio:.2f}) -> ép SATIN thành FILL")
            other.shape.label = "fill"
            label_mask[other.mask == 1] = LABEL_FILL

    return label_mask


# ---------------------------------------------------------------------------
# Hậu xử lý
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

    # Thu thập TOÀN BỘ shape (mọi màu) để làm context refinement sau đó
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

        # Ghi lại từng shape với mask toàn cục (canvas size) để refine sau
        for shape in shapes:
            if shape.label not in ("satin", "fill"):
                continue
            shape.global_id = global_id_counter
            global_id_counter += 1
            shape_mask = shape_to_mask(shape, canvas_shape)
            area_px = int(np.count_nonzero(shape_mask))
            global_records.append(_GlobalShapeRecord(shape, shape_mask, area_px))

    # ---- BƯỚC MỚI: Context-aware refinement ----
    if enable_context_refinement and global_records:
        if verbose:
            print(">> Dang chay Context-aware refinement (outline satin -> fill ben trong)...")
        label_mask = refine_labels_by_context(
            global_records, label_mask, canvas_shape, verbose=verbose
        )

    # Lấp "khe hở đen" và hòa tan vùng rác kỹ thuật số
    foreground_mask = ~is_background
    label_mask = fill_unlabeled_gaps(label_mask, foreground_mask)

    return label_mask, working_img


# ---------------------------------------------------------------------------
# Xuất ảnh Preview & Khởi chạy CLI
# ---------------------------------------------------------------------------
def save_preview(label_mask: np.ndarray, output_path: str) -> None:
    h, w = label_mask.shape[:2]
    preview = np.zeros((h, w, 3), dtype=np.uint8)
    preview[label_mask == LABEL_FILL] = _PREVIEW_COLOR_FILL
    preview[label_mask == LABEL_SATIN] = _PREVIEW_COLOR_SATIN
    cv2.imwrite(output_path, preview)
    print(f"Preview saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Logo PNG TRONG SUỐT (Khuyên dùng):")
        print("    python opencv_stitch_classifier.py <image.png> [output_preview] [physical_width_mm] [threshold_mm] --logo [--no-context]")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    logo_flag = "--logo" in sys.argv
    no_context_flag = "--no-context" in sys.argv

    image_path_arg = args[0]
    output_path_arg = args[1] if len(args) > 1 else None
    physical_width_mm_arg = float(args[2]) if len(args) > 2 else DEFAULT_PHYSICAL_WIDTH_MM
    threshold_mm_arg = float(args[3]) if len(args) > 3 else DEFAULT_THRESHOLD_MM

    if logo_flag:
        mask, preview_colors = classify_multicolor_image(
            image_path_arg, physical_width_mm=physical_width_mm_arg,
            threshold_mm=threshold_mm_arg,
            enable_context_refinement=not no_context_flag,
            verbose=True,
        )
        if output_path_arg:
            save_preview(mask, output_path_arg)

    print(f"\nMask shape: {mask.shape}")
    print(f"Background pixels: {(mask == LABEL_BACKGROUND).sum()}")
    print(f"Fill pixels: {(mask == LABEL_FILL).sum()}")
    print(f"Satin pixels: {(mask == LABEL_SATIN).sum()}")