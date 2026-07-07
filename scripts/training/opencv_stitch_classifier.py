"""
OpenCV Stitch Classifier - Satin vs Fill (rule-based, KHÔNG train model)
=========================================================================

Bản chuyển đổi của svg_path_classifier.py, nhưng làm việc TRỰC TIẾP trên
ảnh raster (PNG/JPG mask nhị phân hoặc grayscale), dùng thuần OpenCV +
NumPy (không cần svgpathtools, không cần shapely).

Ý tưởng:
- Mỗi "shape" (path) trong SVG bây giờ tương ứng với 1 contour ngoài
  (external contour) tìm được bằng cv2.findContours với mode RETR_CCOMP.
- RETR_CCOMP chỉ trả về hierarchy 2 CẤP:
    cấp 1 = contour ngoài (exterior),  parent = -1
    cấp 2 = lỗ (hole) của contour ngoài đó, parent = index contour ngoài
  -> đúng ý nghĩa "is_hollow" trong bản SVG (không cần tính nesting depth
     bằng shapely .contains() như bản cũ).
- Mọi phép đo (area, hull area -> solidity, aspect ratio, thickness mm)
  đều tính trên MASK nhị phân của từng contour (kèm lỗ), y hệt cách bản
  SVG rasterize geometry rồi đo, nên kết quả tương thích 1-1 với quy tắc
  gốc.

QUY TẮC PHÂN LOẠI (giữ nguyên y hệt bản SVG gốc, đơn vị mm quy đổi từ
physical_width_mm giả định cho TOÀN BỘ canvas):

1. Contour ngoài cùng thực sự (hull area > 40% canvas, có lỗ) và
   độ dày <= 8.0mm -> SATIN
2. Có lỗ (hollow) còn lại:
   - độ dày >= threshold_mm (mặc định 2.0mm) -> FILL
   - mỏng hơn -> SATIN
3. Không lỗ, solidity < 0.65 (nét ngoằn ngoèo kiểu chữ S, C, M) và
   độ dày < threshold_mm * 1.5 -> SATIN
4. Dài/hẹp (aspect ratio > 4.0) và độ dày <= 4.0mm -> SATIN
5. Đặc, solidity > 0.85, độ dày >= threshold_mm -> FILL
6. Fallback: độ dày < threshold_mm -> SATIN, ngược lại -> FILL

QUAN TRỌNG: physical_width_mm=80.0 là GIẢ ĐỊNH CỐ ĐỊNH cho MỌI ảnh input
(coi như bề rộng ảnh = 80mm thực tế). Nếu dataset của bạn không đồng nhất
kích thước thật, chỉnh lại tham số này (hoặc đo trực tiếp DPI/scale nếu có).

INPUT MONG ĐỢI: ảnh grayscale hoặc nhị phân, nét thêu là vùng SÁNG (hoặc
tối, dùng invert=True) trên nền khác màu. Nếu ảnh gốc là RGB nhiều màu
theo từng path riêng biệt (mỗi path 1 màu), dùng hàm
`classify_multicolor_image` bên dưới để tách từng vùng màu ra xử lý
riêng thay vì gộp chung 1 mask nhị phân.
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
# Bước 1: load ảnh -> mask nhị phân
# ---------------------------------------------------------------------------
def load_binary_mask(image_path: str, invert: bool = False,
                      thresh: int = 127) -> np.ndarray:
    """Đọc ảnh grayscale rồi threshold thành mask nhị phân 0/255.

    invert=True nếu nét thêu là vùng TỐI trên nền sáng.
    """
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
    """
    Dùng cv2.RETR_CCOMP: trả về hierarchy đúng 2 cấp (ngoài / lỗ), khớp
    thẳng với khái niệm is_hollow của bản SVG mà không cần tính nesting
    depth thủ công.

    hierarchy[i] = [Next, Previous, First_Child, Parent]
    - Parent == -1  -> đây là contour NGOÀI (exterior)
    - Parent != -1  -> đây là LỖ của contour ngoài tại index Parent
    """
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # bỏ chiều batch thừa của OpenCV

    shapes: List[Shape] = []
    for i, h in enumerate(hierarchy):
        parent = h[3]
        if parent != -1:
            continue  # đây là lỗ, sẽ được gán vào shape ngoài tương ứng

        holes = [
            contours[j] for j, hj in enumerate(hierarchy) if hj[3] == i
        ]
        # Bỏ qua contour quá nhỏ (nhiễu)
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


def thickness_mm_from_mask(mask: np.ndarray, pixel_to_mm: float) -> float:
    """distanceTransform + skeleton (median trên xương) -> độ dày mm.
    Giống hệt cách đo trong bản SVG, chỉ khác input là mask OpenCV thuần."""
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
        pass  # scikit-image không có -> fallback dùng max distance

    return (np.max(dist) * 2.0) * pixel_to_mm


# ---------------------------------------------------------------------------
# Bước 4: áp dụng quy tắc cố định (y hệt bản SVG)
# ---------------------------------------------------------------------------
def _classify_shape(shape: Shape, mask: np.ndarray, is_outer_candidate: bool,
                     canvas_area: float, pixel_to_mm: float,
                     threshold_mm: float) -> Tuple[str, Dict]:
    area_px = cv2.contourArea(shape.contour)
    hull = cv2.convexHull(shape.contour)
    hull_area_px = cv2.contourArea(hull)
    solidity = area_px / hull_area_px if hull_area_px > 0 else 1.0

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


def classify_binary_mask(binary_mask: np.ndarray,
                          physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                          threshold_mm: float = DEFAULT_THRESHOLD_MM,
                          verbose: bool = False) -> Tuple[List[Shape], np.ndarray]:
    """
    Phân loại toàn bộ shape trong 1 mask nhị phân.
    Trả về (list Shape đã gán .label, mask nhãn 3 lớp 0/1/2 cùng kích thước ảnh).
    """
    h, w = binary_mask.shape[:2]
    canvas_area = float(w * h)
    pixel_to_mm = physical_width_mm / max(float(w), 1.0)

    shapes = extract_shapes(binary_mask)
    if not shapes:
        return [], np.zeros((h, w), dtype=np.uint8)

    # Tìm outer_border_id = shape có hull area lớn nhất (giống bản SVG)
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


# ---------------------------------------------------------------------------
# Tiện ích: ảnh nhiều màu (mỗi path 1 màu riêng) -> tách từng màu rồi phân loại
# ---------------------------------------------------------------------------
def classify_multicolor_image(image_path: str,
                               physical_width_mm: float = DEFAULT_PHYSICAL_WIDTH_MM,
                               threshold_mm: float = DEFAULT_THRESHOLD_MM,
                               background_color: Tuple[int, int, int] = (255, 255, 255),
                               color_tolerance: int = 10,
                               verbose: bool = False) -> np.ndarray:
    """
    Dùng khi ảnh gốc là RGB, mỗi path/mảng thêu có 1 màu riêng biệt (không
    phải nhị phân). Với mỗi màu unique khác background, tách mask riêng
    rồi chạy classify_binary_mask, gộp kết quả vào 1 label_mask chung.
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

    h, w = img_bgr.shape[:2]
    label_mask = np.zeros((h, w), dtype=np.uint8)

    pixels = img_bgr.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    bg = np.array(background_color[::-1])  # BGR

    for color in unique_colors:
        if np.all(np.abs(color.astype(int) - bg.astype(int)) <= color_tolerance):
            continue  # bỏ qua nền

        color_mask = cv2.inRange(img_bgr, color - color_tolerance, color + color_tolerance)
        if color_mask.sum() == 0:
            continue

        if verbose:
            print(f"Xu ly mau BGR={tuple(color)} ...")

        _, sub_label_mask = classify_binary_mask(
            color_mask, physical_width_mm=physical_width_mm,
            threshold_mm=threshold_mm, verbose=verbose,
        )
        label_mask[sub_label_mask > 0] = sub_label_mask[sub_label_mask > 0]

    return label_mask


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
    """Entry point chính cho ảnh nhị phân/grayscale đơn giản (1 màu nét thêu)."""
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
        print("Usage: python opencv_stitch_classifier.py <image_path> [output_preview_path] "
              "[physical_width_mm] [threshold_mm] [--invert]")
        print(f"  (mac dinh physical_width_mm={DEFAULT_PHYSICAL_WIDTH_MM}, "
              f"threshold_mm={DEFAULT_THRESHOLD_MM})")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    invert_flag = "--invert" in sys.argv

    image_path_arg = args[0]
    output_path_arg = args[1] if len(args) > 1 else None
    physical_width_mm_arg = float(args[2]) if len(args) > 2 else DEFAULT_PHYSICAL_WIDTH_MM
    threshold_mm_arg = float(args[3]) if len(args) > 3 else DEFAULT_THRESHOLD_MM

    mask = classify_image_file(
        image_path_arg, output_path_arg, invert=invert_flag,
        physical_width_mm=physical_width_mm_arg, threshold_mm=threshold_mm_arg,
    )

    print(f"\nMask shape: {mask.shape}")
    print(f"Background pixels: {(mask == LABEL_BACKGROUND).sum()}")
    print(f"Fill pixels: {(mask == LABEL_FILL).sum()}")
    print(f"Satin pixels: {(mask == LABEL_SATIN).sum()}")