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
    fill_pixels = (pred_bgr[:,:,0] < 50) & (pred_bgr[:,:,1] > 200) & (pred_bgr[:,:,2] > 200)
    # Màu Hồng (Magenta BGR) -> Gán Satin (2)
    satin_pixels = (pred_bgr[:,:,0] > 200) & (pred_bgr[:,:,1] < 50) & (pred_bgr[:,:,2] > 200)
    
    pred_mask[fill_pixels] = 1
    pred_mask[satin_pixels] = 2
    
    return pred_mask

def generate_visual_error_map(pred_mask: np.ndarray, gt_mask: np.ndarray, output_path: str):
    """So sánh ma trận dự đoán và ma trận chuẩn từ SVG, xuất ảnh trực quan hóa lỗi"""
    h, w = gt_mask.shape
    error_visual = np.zeros((h, w, 3), dtype=np.uint8)

    # Vùng đoán ĐÚNG (Match) -> Tô màu xám mờ để giữ dáng logo
    match_mask = (pred_mask == gt_mask) & (gt_mask != 0)
    error_visual[match_mask] = [70, 70, 70]

    # LỖI THIẾU SATIN (Trong SVG là Satin=2, nhưng code đoán là Fill=1 hoặc bỏ sót=0) -> TÔ ĐỎ RỰC
    missed_satin = (gt_mask == 2) & (pred_mask != 2)
    error_visual[missed_satin] = [0, 0, 255]

    # LỖI DƯ SATIN (Trong SVG là Fill=1 hoặc Nền=0, nhưng code đoán nhầm là Satin=2) -> TÔ VÀNG CHÓI
    over_satin = (gt_mask != 2) & (pred_mask == 2)
    error_visual[over_satin] = [0, 255, 255]

    # Ghi file ảnh lỗi
    cv2.imwrite(output_path, error_visual)
    
    return np.count_nonzero(missed_satin), np.count_nonzero(over_satin)

def evaluate_directly_from_svg(svg_dir: str, pred_dir: str, error_map_dir: str):
    if not os.path.exists(error_map_dir):
        os.makedirs(error_map_dir)

    print(f"{'Tên File SVG':<30} | {'Sót Satin (Tô Đỏ)':<20} | {'Dư Satin (Tô Vàng)':<20}")
    print("-" * 78)
    
    total_checked = 0

    for filename in os.listdir(svg_dir):
        if not filename.endswith(".svg"):
            continue
            
        svg_path = os.path.join(svg_dir, filename)
        
        # Tìm file ảnh dự đoán tương ứng (.png)
        pred_filename = filename.replace(".svg", ".png")
        pred_path = os.path.join(pred_dir, pred_filename)
        
        if not os.path.exists(pred_path):
            print(f"[CẢNH BÁO] Không tìm thấy ảnh dự đoán kết quả cho: {filename} (định dạng mong đợi: {pred_filename})")
            continue

        try:
            # 1. Dựng ảnh chuẩn từ file SVG trực tiếp trên RAM
            gt_mask = render_svg_to_gt_mask(svg_path, target_w=4200, target_h=4800)
            
            # 2. Đọc ảnh dự đoán bằng hàm Smart Load (tự động quy đổi màu)
            pred_mask = load_pred_mask_smart(pred_path)
            if pred_mask is None:
                continue
            
            # 3. So sánh và tạo bản đồ lỗi
            error_output_path = os.path.join(error_map_dir, f"error_{pred_filename}")
            missed_px, over_px = generate_visual_error_map(pred_mask, gt_mask, error_output_path)
            
            total_checked += 1
            if missed_px > 0 or over_px > 0:
                print(f"{filename:<30} | {missed_px:<20} | {over_px:<20}")
            else:
                print(f"{filename:<30} | Khớp hoàn hảo 100%!")
                
        except Exception as e:
            print(f"Lỗi xử lý file {filename}: {e}")
            
    print(f"\n[INFO] Hoàn thành! Đã kiểm tra tổng cộng {total_checked} file.")

if __name__ == "__main__":
    # Đã cấu hình theo thư mục dự án của bạn
    SVG_INKSCAPE_DIR = "data/opencv_test/svg/"
    PREDICTED_MASK_DIR = "data/opencv_test/predictions/" 
    ERROR_MAP_OUTPUT_DIR = "data/opencv_test/error_maps/" 
    
    evaluate_directly_from_svg(SVG_INKSCAPE_DIR, PREDICTED_MASK_DIR, ERROR_MAP_OUTPUT_DIR)