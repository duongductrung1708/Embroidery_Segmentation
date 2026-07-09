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
# VÙNG DUNG SAI (BOUNDARY RELAXATION)
# ---------------------------------------------------------------------------
# Cho phép sai số 5 pixel (khoảng 0.1mm) ở các đường biên do khác biệt giữa 
# ảnh Vector (SVG) và ảnh Pixel (PNG). 
DEFAULT_TOLERANCE_PX = 5


def render_svg_to_gt_mask(svg_path: str, target_w: int = 4200, target_h: int = 4800) -> np.ndarray:
    """Đọc trực tiếp file SVG, tách nhãn Inkscape và dựng thành ma trận nhãn 0,1,2 trong bộ nhớ"""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Đổi màu tạm thời các path theo nhãn Inkscape
    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            label = child.get(f"{{{INKSCAPE_NS}}}label", "").lower()

            if "satin" in label:
                color = "#00FF00"  # Xanh lục
            elif "fill" in label:
                color = "#FF0000"  # Đỏ
            else:
                color = "none"

            if color != "none":
                child.set("fill", color)
                child.set("stroke", "none")
                child.set("fill-opacity", "1")
                child.attrib.pop("style", None)

    svg_bytes = ET.tostring(root, encoding='utf-8')
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=target_w, output_height=target_h)

    img_pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img_rgb = np.array(img_pil)

    gt_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    gt_mask[img_rgb[:, :, 0] > 128] = 1  # Đỏ -> FILL
    gt_mask[img_rgb[:, :, 1] > 128] = 2  # Xanh -> SATIN

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

    # KỊCH BẢN 2: Ảnh màu Preview 
    fill_pixels = (pred_bgr[:, :, 0] < 50) & (pred_bgr[:, :, 1] > 200) & (pred_bgr[:, :, 2] > 200)
    satin_pixels = (pred_bgr[:, :, 0] > 200) & (pred_bgr[:, :, 1] < 50) & (pred_bgr[:, :, 2] > 200)

    pred_mask[fill_pixels] = 1
    pred_mask[satin_pixels] = 2

    return pred_mask


def generate_visual_error_map(pred_mask: np.ndarray, gt_mask: np.ndarray, output_path: str, tolerance_px: int):
    """
    So sánh ma trận dự đoán và chuẩn SVG có sử dụng Vùng Dung Sai.
    Bỏ qua hoàn toàn các sai số nhỏ ở viền, chỉ đánh dấu các lỗi Lõi thực sự.
    """
    h, w = gt_mask.shape
    error_visual = np.zeros((h, w, 3), dtype=np.uint8)

    # Vùng đoán ĐÚNG cốt lõi -> Tô xám mờ để giữ dáng logo
    match_mask = (pred_mask == gt_mask) & (gt_mask != 0)
    error_visual[match_mask] = [70, 70, 70]

    # Khởi tạo kernel dung sai
    kernel = np.ones((tolerance_px * 2 + 1, tolerance_px * 2 + 1), np.uint8)

    # 1. LỖI THIẾU SATIN (Missed Core Error)
    pred_satin_dilated = cv2.dilate((pred_mask == 2).astype(np.uint8), kernel)
    missed_satin_core = (gt_mask == 2) & (pred_satin_dilated == 0)
    error_visual[missed_satin_core] = [0, 0, 255] # Tô Đỏ rực

    # 2. LỖI DƯ SATIN (Over Core Error)
    gt_satin_dilated = cv2.dilate((gt_mask == 2).astype(np.uint8), kernel)
    over_satin_core = (pred_mask == 2) & (gt_satin_dilated == 0)
    error_visual[over_satin_core] = [0, 255, 255] # Tô Vàng chói

    cv2.imwrite(output_path, error_visual)

    return np.count_nonzero(missed_satin_core), np.count_nonzero(over_satin_core)


def evaluate_directly_from_svg(svg_dir: str, pred_dir: str, error_map_dir: str, tolerance_px: int):
    if not os.path.exists(error_map_dir):
        os.makedirs(error_map_dir)

    print(f"{'Tên File SVG':<30} | {'Sót Satin (Lỗi Lõi Đỏ)':<25} | {'Dư Satin (Lỗi Lõi Vàng)':<25}")
    print("-" * 85)

    total_checked = 0

    for filename in os.listdir(svg_dir):
        if not filename.endswith(".svg"):
            continue

        svg_path = os.path.join(svg_dir, filename)
        
        # SỬA ĐỔI TẠI ĐÂY: Chuyển "14.svg" thành "14_pred.png"
        pred_filename = filename.replace(".svg", "_pred.png")
        pred_path = os.path.join(pred_dir, pred_filename)

        if not os.path.exists(pred_path):
            print(f"[CẢNH BÁO] Không tìm thấy ảnh dự đoán kết quả cho: {filename} (Mong đợi: {pred_filename})")
            continue

        try:
            gt_mask = render_svg_to_gt_mask(svg_path, target_w=4200, target_h=4800)
            pred_mask = load_pred_mask_smart(pred_path)
            
            if pred_mask is None:
                continue

            error_output_path = os.path.join(error_map_dir, f"error_{pred_filename}")
            
            # Tính toán lỗi đã lọc dung sai
            missed_core, over_core = generate_visual_error_map(pred_mask, gt_mask, error_output_path, tolerance_px)

            total_checked += 1
            if missed_core > 0 or over_core > 0:
                print(f"{filename:<30} | {missed_core:<25} | {over_core:<25}")
            else:
                print(f"{filename:<30} | Khớp hoàn hảo 100% (trong vùng dung sai)!")

        except Exception as e:
            print(f"Lỗi xử lý file {filename}: {e}")

    print(f"\n[INFO] Đã quét xong {total_checked} file. Lỗi viền do răng cưa ({tolerance_px}px) đã được tự động loại bỏ.")


if __name__ == "__main__":
    # Cấu hình thư mục
    SVG_INKSCAPE_DIR = "data/opencv_test/svg/"
    PREDICTED_MASK_DIR = "data/opencv_test/predictions/"
    ERROR_MAP_OUTPUT_DIR = "data/opencv_test/error_maps/"

    # Mặc định lấy dung sai 5 pixel (Có thể chỉnh qua dòng lệnh: python script.py 10)
    tolerance = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOLERANCE_PX

    evaluate_directly_from_svg(SVG_INKSCAPE_DIR, PREDICTED_MASK_DIR, ERROR_MAP_OUTPUT_DIR, tolerance)