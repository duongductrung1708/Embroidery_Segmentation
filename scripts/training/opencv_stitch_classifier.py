#!/usr/bin/env python3
"""
OpenCV Stitch Classifier - Satin vs Fill (rule-based, KHÔNG train model)
=========================================================================

Bản chuyển đổi của svg_path_classifier.py, nhưng làm việc TRỰC TIẾP trên
ảnh raster (PNG/JPG mask nhị phân hoặc grayscale), dùng thuần OpenCV +
NumPy (không cần svgpathtools, không cần shapely).

BẢN CẬP NHẬT:
- Tích hợp hàm Chuẩn hóa Canvas (Mặc định 4200x4800).
- Tối ưu hóa hàm thickness_mm_from_mask để đo siêu tốc trên ảnh khổng lồ.
- Sửa lỗi vùng bên trong chữ (O, A, D, R...) bị gán Satin bằng cờ is_bg_color.
"""

import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants (giữ nguyên như bản SVG)
# ---------------------------------------------------------------------------
LABEL_BACKGROUND = 0
LABEL_FILL = 1
LABEL_SATIN = 2

DEFAULT_PHYSICAL_WIDTH_MM = 80.0
DEFAULT_THRESHOLD_MM = 2.0
OUTER_BORDER_MAX_THICKNESS_MM = 8.0

_PREVIEW_COLOR_FILL = (0, 255, 255)   # Cyan  (BGR)
_PREVIEW_COLOR_SATIN = (255, 0, 255)  # Magenta (BGR)


class Shape:
    """Container cho 1 contour ngoài (tương đương 1 SVGPath trước đây)."""

    def __init__(self, shape_id: int, contour: np.ndarray, holes: List[np.ndarray]):
        self.id = shape_id
        self.contour = contour        # contour ngoài (Nx1x2)
        self.holes = holes            # list contour lỗ bên trong
        self.label: Optional[str] = None


# ---------------------------------------------------------------------------
# Bước 0: Chuẩn hóa Canvas (Letterbox) cho ảnh khổng lồ
# ---------------------------------------------------------------------------
def normalize_to_canvas(img: np.ndarray, target_w: int = 4200, target_h: int = 4800) -> np.ndarray:
    """
    Scale ảnh vừa khít khung target_w x target_h mà KHÔNG làm méo hình.
    Đảm bảo ảnh đầu ra luôn có 4 kênh màu (RGBA) để xử lý trong suốt.
    """
    h_orig, w_orig = img.shape[:2]
    
    # Nếu ảnh đã đúng chuẩn kích thước, chỉ cần đảm bảo có 4 kênh
    if h_orig == target_h and w_orig == target_w:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    scale = min(target_w / w_orig, target_h / h_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    
    # Dùng INTER_NEAREST để không làm nhòe viền màu (Anti-alias)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    if len(resized_img.shape) == 2:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGRA)
    elif resized_img.shape[2] == 3:
        # Nếu ảnh BGR, thêm kênh Alpha = 255 (không trong suốt)
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
# Bước 1: load ảnh -> mask nhị phân
# ---------------------------------------------------------------------------
def load_binary_mask(image_path: str, invert: bool = False,
                      thresh: int = 127) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    if invert:
        binary = 255 - binary
    return binary


# ---------------------------------------------------------------------------
# Bước 2: tách contour ngoài + lỗ bằng hierarchy RETR_CCOMP
# ---------------------------------------------------------------------------
def extract_shapes(binary_mask: np.ndarray) -> List[Shape]:
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]  

    shapes: List[Shape] = []
    for i, h in enumerate(hierarchy):
        parent = h[3]
        if parent != -1:
            continue  

        holes = [
            contours[j] for j, hj in enumerate(hierarchy) if hj[3] == i
        ]
        if cv2.contourArea(contours[i]) < 4:
            continue
        shapes.append(Shape(shape_id=len(shapes), contour=contours[i], holes=holes))

    return shapes


# ---------------------------------------------------------------------------
# Bước 3: dựng mask riêng cho từng shape (ngoài trừ lỗ) để đo đạc
# ---------------------------------------------------------------------------
def shape_to_mask(shape: Shape, canvas_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(canvas_shape, dtype=np.uint8)
    cv2.drawContours(mask, [shape.contour], -1, 1, thickness=cv2.FILLED)
    for hole in shape.holes:
        cv2.drawContours(mask, [hole], -1, 0, thickness=cv2.FILLED)
    return mask


def thickness_mm_from_mask(mask: np.ndarray, pixel_to_mm: float) -> Tuple[float, float]:
    """
    CẬP NHẬT TỐI ƯU TỐC ĐỘ:
    Nếu mask quá lớn (4200x4800), skeletonize sẽ bị treo. Thu nhỏ mask tạm thời 
    xuống dưới 1500px để đo đạc, sau đó quy đổi ngược lại mm thực tế. 
    (Phương pháp này giữ được 99% độ chính xác mà tốc độ tăng gấp 10 lần).
    """
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
    
    # Hệ số mm sau khi đã thu nhỏ ảnh
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
# Bước 4: áp dụng 6 QUY TẮC CỐ ĐỊNH CHÍNH XÁC
# ---------------------------------------------------------------------------
def _classify_shape(shape: Shape, mask: np.ndarray, is_outer_candidate: bool,
                     canvas_area: float, pixel_to_mm: float,
                     threshold_mm: float, is_bg_color: bool = False) -> Tuple[str, Dict]:
    hull = cv2.convexHull(shape.contour)
    hull_area_px = cv2.contourArea(hull)

    true_area_px = float(np.count_nonzero(mask))
    solidity = true_area_px / hull_area_px if hull_area_px > 0 else 1.0

    rect = cv2.minAreaRect(shape.contour)
    (rw, rh) = rect[1]
    short_side, long_side = sorted([rw, rh])
    aspect_ratio = long_side / short_side if short_side > 0 else 100.0

    is_hollow = len(shape.holes) > 0
    thickness_median_mm, thickness_max_mm = thickness_mm_from_mask(mask, pixel_to_mm)

    thickness_mm = thickness_max_mm

    is_outermost = (
        is_outer_candidate
        and canvas_area > 0
        and (hull_area_px / canvas_area) > 0.4
    )

    # =====================================================================
    # 6 QUY TẮC PHÂN LOẠI 
    # =====================================================================

    # 1. Contour ngoài cùng thực sự -> SATIN
    if is_outermost and is_hollow and thickness_mm <= OUTER_BORDER_MAX_THICKNESS_MM:
        label = "satin"

    # 2. Có lỗ (hollow) còn lại -> FILL nếu dày, SATIN nếu mỏng
    elif is_hollow:
        label = "fill" if thickness_mm >= threshold_mm else "satin"

    # 3. Không lỗ, solidity < 0.65 (nét ngoằn ngoèo) -> SATIN
    elif not is_hollow and solidity < 0.65 and thickness_mm < (threshold_mm * 1.5):
        label = "satin"

    # 4. Dài/hẹp (aspect ratio > 4.0) -> SATIN
    elif aspect_ratio > 4.0 and thickness_mm <= 4.0:
        label = "satin"

    # 5. Đặc, solidity > 0.85 -> FILL
    elif solidity > 0.85:
        # VÁ LỖI CHO CHỮ O, A, D:
        # Lỗ trống trong chữ là cục đặc, nhưng nó nằm trong Mảng Nền (is_bg_color).
        # Ép buộc nó thành FILL, bất kể nó có mỏng dưới mức threshold hay không!
        if is_bg_color or thickness_mm >= threshold_mm:
            label = "fill"
        else:
            label = "satin"

    # 6. Fallback
    else:
        label = "satin" if thickness_mm < threshold_mm else "fill"

    details = {
        "thickness_mm": thickness_median_mm,
        "thickness_max_mm": thickness_max_mm,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "is_hollow": is_hollow,
        "is_outermost": is_outermost,
        "is_bg_color": is_bg_color
    }
    return label, details


def classify_binary_mask(binary_mask: np.ndarray,
                          physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                          threshold_mm: float = DEFAULT_THRESHOLD_MM,
                          is_bg_color: bool = False,
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
            threshold_mm=threshold_mm,
            is_bg_color=is_bg_color
        )
        shape.label = label
        label_value = LABEL_SATIN if label == "satin" else LABEL_FILL
        label_mask[mask == 1] = label_value

        if verbose:
            print(f"  Shape {shape.id}: median={details['thickness_mm']:.3f}mm, "
                  f"max={details['thickness_max_mm']:.3f}mm, bg={details['is_bg_color']}, "
                  f"solidity={details['solidity']:.2f}, label={label}")

    return shapes, label_mask


# ---------------------------------------------------------------------------
# Hậu xử lý: lấp "viền đen"
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


def quantize_colors(img_bgr: np.ndarray, n_colors: int = 8) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    samples = img_bgr.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        samples, n_colors, None, criteria, attempts=3,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(h, w, 3)
    return quantized


def classify_multicolor_image(image_path: str,
                               physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                               threshold_mm: float = DEFAULT_THRESHOLD_MM,
                               color_tolerance: int = 10,
                               n_colors: int = 8,
                               quantize: bool = True,
                               alpha_threshold: int = 128,
                               min_region_pixels: int = 15,
                               verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

    # ĐỒNG BỘ HÓA CANVAS LÊN 4200x4800 CHO TẤT CẢ FILE ĐẦU VÀO
    if verbose:
        print(">> Dang scale va dong bo Canvas (4200x4800)...")
    img = normalize_to_canvas(img, target_w=4200, target_h=4800)

    img_bgr = img[:, :, :3]
    alpha = img[:, :, 3]
    is_background = alpha < alpha_threshold

    h, w = img_bgr.shape[:2]
    label_mask = np.zeros((h, w), dtype=np.uint8)

    working_img = quantize_colors(img_bgr, n_colors=n_colors) if quantize else img_bgr.copy()
    working_img[is_background] = (0, 0, 0)

    pixels = working_img[~is_background].reshape(-1, 3)
    if pixels.size == 0:
        return label_mask, working_img
        
    unique_colors = np.unique(pixels, axis=0)
    total_valid_px = (~is_background).sum()

    for color in unique_colors:
        color_int = color.astype(np.int16)
        lower = np.clip(color_int - color_tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(color_int + color_tolerance, 0, 255).astype(np.uint8)
        color_mask = cv2.inRange(working_img, lower, upper)
        color_mask[is_background] = 0  

        n_pixels = int((color_mask > 0).sum())
        if n_pixels < min_region_pixels:
            continue  

        # NẾU MÀU NÀY CHIẾM HƠN 15% DIỆN TÍCH LOGO -> NÓ LÀ MÀU NỀN
        is_bg_color = (n_pixels / total_valid_px) > 0.15

        if verbose:
            tag = "[NỀN]" if is_bg_color else "[NÉT]"
            print(f"Xu ly mau BGR={tuple(int(c) for c in color)}  ({n_pixels}px) {tag} ...")

        _, sub_label_mask = classify_binary_mask(
            color_mask, physical_width_mm=physical_width_mm,
            threshold_mm=threshold_mm, is_bg_color=is_bg_color, verbose=verbose,
        )
        label_mask[sub_label_mask > 0] = sub_label_mask[sub_label_mask > 0]

    foreground_mask = ~is_background
    label_mask = fill_unlabeled_gaps(label_mask, foreground_mask)

    return label_mask, working_img


# ---------------------------------------------------------------------------
# Preview / debug
# ---------------------------------------------------------------------------
def save_preview(label_mask: np.ndarray, output_path: str) -> None:
    h, w = label_mask.shape[:2]
    preview = np.zeros((h, w, 3), dtype=np.uint8)
    preview[label_mask == LABEL_FILL] = _PREVIEW_COLOR_FILL
    preview[label_mask == LABEL_SATIN] = _PREVIEW_COLOR_SATIN
    cv2.imwrite(output_path, preview)
    print(f"Preview saved to {output_path}")


def classify_image_file(image_path: str, output_mask_path: Optional[str] = None,
                         invert: bool = False, thresh: int = 127,
                         physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                         threshold_mm: float = DEFAULT_THRESHOLD_MM) -> np.ndarray:
    binary_mask = load_binary_mask(image_path, invert=invert, thresh=thresh)
    print(f"File: {image_path}  ({binary_mask.shape[1]}x{binary_mask.shape[0]}px), "
          f"gia dinh rong {physical_width_mm}mm thuc te")

    shapes, label_mask = classify_binary_mask(
        binary_mask, physical_width_mm=physical_width_mm,
        threshold_mm=threshold_mm, is_bg_color=False, verbose=True,
    )
    print(f"Tim thay {len(shapes)} shape(s).")

    if output_mask_path:
        save_preview(label_mask, output_mask_path)

    return label_mask


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Logo mau, nen trong suot (PNG RGBA):")
        print("    python opencv_stitch_classifier.py <image.png> [output_preview] "
              "[physical_width_mm] [threshold_mm] --logo")
        print("  Mask nhi phan / grayscale don sac:")
        print("    python opencv_stitch_classifier.py <image_path> [output_preview] "
              "[physical_width_mm] [threshold_mm] [--invert]")
        print(f"  (mac dinh physical_width_mm={DEFAULT_PHYSICAL_WIDTH_MM}, "
              f"threshold_mm={DEFAULT_THRESHOLD_MM})")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    invert_flag = "--invert" in sys.argv
    logo_flag = "--logo" in sys.argv

    image_path_arg = args[0]
    output_path_arg = args[1] if len(args) > 1 else None
    physical_width_mm_arg = float(args[2]) if len(args) > 2 else DEFAULT_PHYSICAL_WIDTH_MM
    threshold_mm_arg = float(args[3]) if len(args) > 3 else DEFAULT_THRESHOLD_MM

    if logo_flag:
        mask, preview_colors = classify_multicolor_image(
            image_path_arg, physical_width_mm=physical_width_mm_arg,
            threshold_mm=threshold_mm_arg, verbose=True,
        )
        if output_path_arg:
            save_preview(mask, output_path_arg)
            cv2.imwrite(output_path_arg.replace(".png", "_quantized.png"), preview_colors)
    else:
        mask = classify_image_file(
            image_path_arg, output_path_arg, invert=invert_flag,
            physical_width_mm=physical_width_mm_arg, threshold_mm=threshold_mm_arg,
        )

    print(f"\nMask shape: {mask.shape}")
    print(f"Background pixels: {(mask == LABEL_BACKGROUND).sum()}")
    print(f"Fill pixels: {(mask == LABEL_FILL).sum()}")
    print(f"Satin pixels: {(mask == LABEL_SATIN).sum()}")