import os
import xml.etree.ElementTree as ET
import cairosvg
import cv2
import numpy as np
from PIL import Image
import io
import sys

# Đăng ký Namespace của Inkscape để Python đọc được nhãn
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("inkscape", INKSCAPE_NS)

# ---------------------------------------------------------------------------
# MỚI: Boundary tolerance
# ---------------------------------------------------------------------------
# Số pixel quanh MỌI đường biên (giữa satin/fill/background) sẽ được bỏ qua khi
# so sánh. Lý do: GT được render trực tiếp từ SVG (vector, qua cairosvg) trong
# khi ảnh dự đoán đến từ 1 pipeline rasterize khác (PNG gốc + resize nearest-
# neighbor trong normalize_to_canvas). Sai lệch vài pixel ở viền do 2 pipeline
# khác nhau là chuyện bình thường, KHÔNG phản ánh lỗi phân loại thật sự.
DEFAULT_BOUNDARY_TOLERANCE_PX = 8


def render_svg_to_gt_mask(svg_path: str, target_w: int = 4200, target_h: int = 4800) -> np.ndarray:
    """Đọc trực tiếp file SVG, tách nhãn Inkscape và dựng thành ma trận nhãn 0,1,2 trong bộ nhớ"""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # 1. Đổi màu tạm thời các path theo nhãn Inkscape để tránh bị nhòe màu khi render
    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            label = child.get(f"{{{INKSCAPE_NS}}}label", "").lower()

            if "satin" in label:
                color = "#00FF00"  # Xanh lục rực cho Satin
            elif "fill" in label:
                color = "#FF0000"  # Đỏ rực cho Fill
            else:
                color = "none"

            if color != "none":
                child.set("fill", color)
                child.set("stroke", "none")
                child.set("fill-opacity", "1")
                child.attrib.pop("style", None)

    # 2. Render SVG ra mảng byte RGBA trong RAM
    svg_bytes = ET.tostring(root, encoding='utf-8')
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=target_w, output_height=target_h)

    # 3. Chuyển mảng byte thành Ma trận nhãn (0: Nền, 1: Fill, 2: Satin)
    img_pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img_rgb = np.array(img_pil)

    gt_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    gt_mask[img_rgb[:, :, 0] > 128] = 1  # Màu Đỏ -> FILL
    gt_mask[img_rgb[:, :, 1] > 128] = 2  # Màu Xanh -> SATIN

    return gt_mask


def load_pred_mask_smart(pred_path: str) -> np.ndarray:
    """Đọc ảnh Dự đoán: Tự động hiểu cả ảnh mask số học lẫn ảnh màu Preview (Vàng/Hồng)"""
    pred_bgr = cv2.imread(pred_path, cv2.IMREAD_COLOR)
    if pred_bgr is None:
        return None

    h, w = pred_bgr.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    # KỊCH BẢN 1: Ảnh mask đen thui (toàn số 0, 1, 2)
    if pred_bgr.max() <= 2:
        return pred_bgr[:, :, 0]

    # KỊCH BẢN 2: Ảnh màu Preview của code phân loại
    # Màu Vàng (Cyan BGR) -> Gán Fill (1)
    fill_pixels = (pred_bgr[:, :, 0] < 50) & (pred_bgr[:, :, 1] > 200) & (pred_bgr[:, :, 2] > 200)
    # Màu Hồng (Magenta BGR) -> Gán Satin (2)
    satin_pixels = (pred_bgr[:, :, 0] > 200) & (pred_bgr[:, :, 1] < 50) & (pred_bgr[:, :, 2] > 200)

    pred_mask[fill_pixels] = 1
    pred_mask[satin_pixels] = 2

    return pred_mask


# ---------------------------------------------------------------------------
# MỚI: Xây vùng đệm quanh biên để loại nhiễu rasterization
# ---------------------------------------------------------------------------
def build_boundary_ignore_mask(gt_mask: np.ndarray, tolerance_px: int) -> np.ndarray:
    """
    Trả về mask (bool) các pixel nằm trong khoảng `tolerance_px` quanh BẤT KỲ
    đường biên nào giữa 2 nhãn khác nhau trong gt_mask (kể cả biên với nền).

    Các pixel này sẽ bị loại khỏi phép so sánh vì sai lệch tại đây thường do
    2 pipeline rasterize khác nhau (SVG vector render vs ảnh PNG resize), không
    phải lỗi phân loại thật sự của thuật toán.
    """
    if tolerance_px <= 0:
        return np.zeros_like(gt_mask, dtype=bool)

    edges = np.zeros_like(gt_mask, dtype=bool)
    edges |= (gt_mask != np.roll(gt_mask, 1, axis=0))
    edges |= (gt_mask != np.roll(gt_mask, -1, axis=0))
    edges |= (gt_mask != np.roll(gt_mask, 1, axis=1))
    edges |= (gt_mask != np.roll(gt_mask, -1, axis=1))

    edges_u8 = edges.astype(np.uint8)
    kernel = np.ones((tolerance_px * 2 + 1, tolerance_px * 2 + 1), np.uint8)
    dilated = cv2.dilate(edges_u8, kernel, iterations=1)

    return dilated.astype(bool)


def generate_visual_error_map(pred_mask: np.ndarray, gt_mask: np.ndarray, output_path: str,
                               boundary_tolerance_px: int = DEFAULT_BOUNDARY_TOLERANCE_PX):
    """
    So sánh ma trận dự đoán và ma trận chuẩn từ SVG, xuất ảnh trực quan hóa lỗi.

    MỚI: Phân biệt 2 loại lỗi:
      - "Lỗi biên" (boundary noise): sai lệch nằm trong vùng đệm quanh đường biên GT
        -> do khác pipeline rasterize, KHÔNG PHẢI lỗi phân loại thật -> tô CAM nhạt.
      - "Lỗi lõi" (core error): sai lệch nằm sâu trong vùng (ngoài vùng đệm biên)
        -> lỗi phân loại thật sự, cần sửa -> vẫn tô ĐỎ (sót satin) / VÀNG (dư satin).
    """
    h, w = gt_mask.shape
    error_visual = np.zeros((h, w, 3), dtype=np.uint8)

    ignore_mask = build_boundary_ignore_mask(gt_mask, boundary_tolerance_px)

    # Vùng đoán ĐÚNG (Match) -> Tô màu xám mờ để giữ dáng logo
    match_mask = (pred_mask == gt_mask) & (gt_mask != 0)
    error_visual[match_mask] = [70, 70, 70]

    missed_satin_all = (gt_mask == 2) & (pred_mask != 2)
    over_satin_all = (gt_mask != 2) & (pred_mask == 2)

    missed_satin_boundary = missed_satin_all & ignore_mask
    over_satin_boundary = over_satin_all & ignore_mask

    missed_satin_core = missed_satin_all & (~ignore_mask)
    over_satin_core = over_satin_all & (~ignore_mask)

    # Lỗi biên (nhiễu rasterization) -> tô CAM nhạt, để phân biệt với lỗi thật
    error_visual[missed_satin_boundary] = [0, 140, 255]   # Cam (BGR)
    error_visual[over_satin_boundary] = [0, 140, 255]

    # LỖI THIẾU SATIN THẬT (core) -> TÔ ĐỎ RỰC
    error_visual[missed_satin_core] = [0, 0, 255]

    # LỖI DƯ SATIN THẬT (core) -> TÔ VÀNG CHÓI
    error_visual[over_satin_core] = [0, 255, 255]

    cv2.imwrite(output_path, error_visual)

    return {
        "missed_total": int(np.count_nonzero(missed_satin_all)),
        "over_total": int(np.count_nonzero(over_satin_all)),
        "missed_boundary": int(np.count_nonzero(missed_satin_boundary)),
        "over_boundary": int(np.count_nonzero(over_satin_boundary)),
        "missed_core": int(np.count_nonzero(missed_satin_core)),
        "over_core": int(np.count_nonzero(over_satin_core)),
    }


def evaluate_directly_from_svg(svg_dir: str, pred_dir: str, error_map_dir: str,
                                boundary_tolerance_px: int = DEFAULT_BOUNDARY_TOLERANCE_PX):
    if not os.path.exists(error_map_dir):
        os.makedirs(error_map_dir)

    print(f"{'Tên File SVG':<30} | {'Sót Satin (Đỏ/Cam)':<22} | {'Dư Satin (Vàng/Cam)':<22} | {'Lỗi LÕI thật sự':<18}")
    print("-" * 100)

    total_checked = 0
    total_core_errors = 0
    total_boundary_errors = 0

    for filename in os.listdir(svg_dir):
        if not filename.endswith(".svg"):
            continue

        svg_path = os.path.join(svg_dir, filename)

        pred_filename = filename.replace(".svg", ".png")
        pred_path = os.path.join(pred_dir, pred_filename)

        if not os.path.exists(pred_path):
            print(f"[CẢNH BÁO] Không tìm thấy ảnh dự đoán kết quả cho: {filename} (định dạng mong đợi: {pred_filename})")
            continue

        try:
            gt_mask = render_svg_to_gt_mask(svg_path, target_w=4200, target_h=4800)

            pred_mask = load_pred_mask_smart(pred_path)
            if pred_mask is None:
                continue

            error_output_path = os.path.join(error_map_dir, f"error_{pred_filename}")
            stats = generate_visual_error_map(
                pred_mask, gt_mask, error_output_path,
                boundary_tolerance_px=boundary_tolerance_px
            )

            total_checked += 1
            core_errors = stats["missed_core"] + stats["over_core"]
            boundary_errors = stats["missed_boundary"] + stats["over_boundary"]
            total_core_errors += core_errors
            total_boundary_errors += boundary_errors

            missed_str = f"{stats['missed_total']} (cam:{stats['missed_boundary']})"
            over_str = f"{stats['over_total']} (cam:{stats['over_boundary']})"

            if core_errors > 0:
                print(f"{filename:<30} | {missed_str:<22} | {over_str:<22} | {core_errors:<18}")
            elif boundary_errors > 0:
                print(f"{filename:<30} | Chỉ lệch biên (rasterization), KHÔNG lỗi phân loại thật ({boundary_errors}px)")
            else:
                print(f"{filename:<30} | Khớp hoàn hảo 100%!")

        except Exception as e:
            print(f"Lỗi xử lý file {filename}: {e}")

    print(f"\n[INFO] Hoàn thành! Đã kiểm tra tổng cộng {total_checked} file.")
    print(f"[INFO] Tổng lỗi LÕI thật sự (cần sửa): {total_core_errors} px")
    print(f"[INFO] Tổng lỗi BIÊN (nhiễu rasterization, tolerance={boundary_tolerance_px}px, có thể bỏ qua): {total_boundary_errors} px")


if __name__ == "__main__":
    # Đã cấu hình theo thư mục dự án của bạn
    SVG_INKSCAPE_DIR = "data/opencv_test/svg/"
    PREDICTED_MASK_DIR = "data/opencv_test/predictions/"
    ERROR_MAP_OUTPUT_DIR = "data/opencv_test/error_maps/"

    # Có thể chỉnh boundary_tolerance_px qua CLI: python evaluate_svg_vs_pred.py 6
    tolerance = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BOUNDARY_TOLERANCE_PX

    evaluate_directly_from_svg(SVG_INKSCAPE_DIR, PREDICTED_MASK_DIR, ERROR_MAP_OUTPUT_DIR,
                                boundary_tolerance_px=tolerance)