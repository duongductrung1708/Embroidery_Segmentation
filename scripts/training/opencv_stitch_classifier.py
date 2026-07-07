#!/usr/bin/env python3
"""
OpenCV Stitch Classifier - Satin vs Fill (rule-based, KHÔNG train model)
=========================================================================

Bản cập nhật:
- Fix lỗi nhận diện sai mảng nền lớn thành viền (Tính True Solidity).
- Fix lỗi mất viền đen trùng màu nền (Chỉ Quantize trên valid_pixels & hạ Alpha threshold).
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

_PREVIEW_COLOR_FILL = (0, 255, 255)   # Cyan 
_PREVIEW_COLOR_SATIN = (255, 0, 255)  # Magenta 


class Shape:
    def __init__(self, shape_id: int, contour: np.ndarray, holes: List[np.ndarray]):
        self.id = shape_id
        self.contour = contour        
        self.holes = holes            
        self.label: Optional[str] = None


def load_binary_mask(image_path: str, invert: bool = False,
                      thresh: int = 127) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    if invert:
        binary = 255 - binary
    return binary


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


def shape_to_mask(shape: Shape, canvas_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(canvas_shape, dtype=np.uint8)
    cv2.drawContours(mask, [shape.contour], -1, 1, thickness=cv2.FILLED)
    for hole in shape.holes:
        cv2.drawContours(mask, [hole], -1, 0, thickness=cv2.FILLED)
    return mask


def thickness_mm_from_mask(mask: np.ndarray, pixel_to_mm: float) -> float:
    if mask.sum() == 0:
        return 0.0

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask.astype(bool))
        skeleton_distances = dist[np.where(skeleton)]
        if len(skeleton_distances) > 0:
            median_dist = np.median(skeleton_distances)
            return (median_dist * 2.0) * pixel_to_mm
    except ImportError:
        pass  

    return (np.max(dist) * 2.0) * pixel_to_mm


# ---------------------------------------------------------------------------
# BỘ LUẬT ĐÃ SỬA: Dùng True Solidity để phân biệt Viền và Nền
# ---------------------------------------------------------------------------
def _classify_shape(shape: Shape, mask: np.ndarray, is_outer_candidate: bool,
                     canvas_area: float, pixel_to_mm: float,
                     threshold_mm: float) -> Tuple[str, Dict]:
    
    hull = cv2.convexHull(shape.contour)
    hull_area_px = cv2.contourArea(hull)
    
    # BẢN VÁ 1: Tính Solidity THỰC TẾ (trừ đi diện tích các lỗ thủng)
    true_area_px = np.count_nonzero(mask)
    solidity = true_area_px / hull_area_px if hull_area_px > 0 else 1.0

    rect = cv2.minAreaRect(shape.contour)
    (rw, rh) = rect[1]
    short_side, long_side = sorted([rw, rh])
    aspect_ratio = long_side / short_side if short_side > 0 else 100.0

    is_hollow = len(shape.holes) > 0
    thickness_mm = thickness_mm_from_mask(mask, pixel_to_mm)

    is_outermost = (
        is_outer_candidate
        and canvas_area > 0
        and (hull_area_px / canvas_area) > 0.4
        and solidity < 0.5  # ÉP BUỘC: Đã là viền ngoài thì ruột phải rỗng (solidity < 0.5)
    )

    if is_outermost and is_hollow and thickness_mm <= OUTER_BORDER_MAX_THICKNESS_MM:
        label = "satin"
    elif is_hollow:
        if solidity < 0.6:
            label = "fill"
        else:
            label = "fill" if thickness_mm >= threshold_mm else "satin"
    elif solidity < 0.65 and thickness_mm < (threshold_mm * 1.5):
        label = "satin"
    elif aspect_ratio > 4.0 and thickness_mm <= 4.0:
        label = "satin"
    elif solidity > 0.85:
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
            threshold_mm=threshold_mm,
        )
        shape.label = label
        label_value = LABEL_SATIN if label == "satin" else LABEL_FILL
        label_mask[mask == 1] = label_value

        if verbose:
            print(f"  Shape {shape.id}: thickness={details['thickness_mm']:.3f}mm, "
                  f"hollow={details['is_hollow']}, outermost={details['is_outermost']}, "
                  f"solidity={details['solidity']:.2f}, aspect={details['aspect_ratio']:.2f}, "
                  f"label={label}")

    return shapes, label_mask


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


# ---------------------------------------------------------------------------
# BẢN VÁ 2: Chỉ Quantize trên các pixel Hợp Lệ (Valid pixels)
# ---------------------------------------------------------------------------
def quantize_colors_on_valid_pixels(img_bgr: np.ndarray, valid_mask: np.ndarray, n_colors: int = 8) -> np.ndarray:
    """
    Chỉ chạy K-Means trên vùng logo. Không lãng phí cụm màu cho vùng trong suốt.
    Bảo vệ tối đa các viền đen mỏng không bị xóa sổ.
    """
    samples = img_bgr[valid_mask].astype(np.float32)
    if len(samples) == 0:
        return img_bgr.copy()
        
    n_clusters = min(n_colors, len(np.unique(samples, axis=0)))
    if n_clusters <= 1:
        return img_bgr.copy()
        
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        samples, n_clusters, None, criteria, attempts=3,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(np.uint8)
    
    quantized_img = img_bgr.copy()
    quantized_img[valid_mask] = centers[labels.flatten()]
    return quantized_img


def classify_multicolor_image(image_path: str,
                               physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                               threshold_mm: float = DEFAULT_THRESHOLD_MM,
                               color_tolerance: int = 10,
                               n_colors: int = 8,
                               quantize: bool = True,
                               alpha_threshold: int = 10,  # HẠ NGƯỠNG ĐỂ CỨU VIỀN ĐEN MỜ (ANTI-ALIAS)
                               min_region_pixels: int = 15,
                               verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(
            "Anh khong co kenh alpha (khong phai RGBA). "
            "Neu logo khong co nen trong suot, dung classify_image_file "
            "hoac tu truyen background_color."
        )

    img_bgr = img[:, :, :3]
    alpha = img[:, :, 3]
    is_background = alpha < alpha_threshold

    h, w = img_bgr.shape[:2]
    label_mask = np.zeros((h, w), dtype=np.uint8)

    # Dùng hàm Quantize mới, KHÔNG truyền vùng is_background vào tính toán
    working_img = quantize_colors_on_valid_pixels(img_bgr, ~is_background, n_colors=n_colors) if quantize else img_bgr.copy()
    working_img[is_background] = (0, 0, 0) 

    pixels = working_img[~is_background].reshape(-1, 3)
    if pixels.size == 0:
        return label_mask, working_img
    unique_colors = np.unique(pixels, axis=0)

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

        _, sub_label_mask = classify_binary_mask(
            color_mask, physical_width_mm=physical_width_mm,
            threshold_mm=threshold_mm, verbose=verbose,
        )
        label_mask[sub_label_mask > 0] = sub_label_mask[sub_label_mask > 0]

    foreground_mask = ~is_background
    label_mask = fill_unlabeled_gaps(label_mask, foreground_mask)

    return label_mask, working_img


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
        threshold_mm=threshold_mm, verbose=True,
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