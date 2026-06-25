import os
import glob
import cv2
import numpy as np
import vtracer
from tqdm import tqdm

def clean_rgba_image(img):
    """
    Làm sạch kênh Alpha (trong suốt) bằng cách khử nhiễu,
    giữ nguyên format RGBA để vtracer có thể tạo SVG với nền trong suốt.
    """
    if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        alpha = img[:, :, 3]
        
        # Làm mượt alpha channel bằng Gaussian blur để giảm răng cưa
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        
        # Khử nhiễu alpha channel bằng morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_clean = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        alpha_clean = cv2.morphologyEx(alpha_clean, cv2.MORPH_CLOSE, kernel)
        
        # Cập nhật alpha channel đã làm sạch
        result = img.copy()
        result[:, :, 3] = alpha_clean
        
        return result
    
    # Nếu ảnh không có alpha, thêm alpha channel (đen = trong suốt)
    if len(img.shape) == 3 and img.shape[2] == 3:  # BGR
        alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        result = cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha])
        return result
    
    return img

def process_pipeline(input_dir, clean_dir, svg_dir):
    """
    Pipeline: Đọc PNG -> Làm sạch -> Lưu PNG tạm -> Vector hóa (Color Mode)
    """
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    # Quét tất cả các định dạng ảnh phổ biến
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG'): 
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong thư mục: {input_dir}")
        return

    print(f"Bắt đầu xử lý {len(image_paths)} ảnh...")

    for img_path in tqdm(image_paths, desc="Vector hóa"):
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)

        clean_path = os.path.join(clean_dir, f"{name}.png")
        svg_path = os.path.join(svg_dir, f"{name}.svg")

        # 1. Đọc ảnh & Làm sạch
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        
        clean_img = clean_rgba_image(img)
        cv2.imwrite(clean_path, clean_img)

        # 2. Vector hóa (Chế độ color để giữ màu logo và transparency)
        vtracer.convert_image_to_svg_py(
            clean_path,
            svg_path,
            colormode="color",        # Giữ nguyên màu
            hierarchical="stacked",   # Tách layer
            mode="spline",            # Đường cong Bezier mượt
            filter_speckle=1,         # Giảm ngưỡng khử nhiễu để giữ chi tiết
            color_precision=12,       # Tăng độ chính xác màu để giảm quantization
            layer_difference=12,      # Tăng phân tách vùng màu
            corner_threshold=50,      # Giảm ngưỡng góc để đường cong mượt hơn
            length_threshold=1.5,     # Giảm ngưỡng độ dài đường cong
            max_iterations=25,        # Tăng iterations để hội tụ tốt hơn
            splice_threshold=40,      # Giảm ngưỡng nối đường
            path_precision=12         # Tăng precision đường cong
        )

    print(f"\nHoàn thành!")
    print(f"SVG (Vector): {svg_dir}")
    print(f"PNG (Cleaned): {clean_dir}")

if __name__ == "__main__":
    # Lấy thư mục hiện tại là data_prep/
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Đi ngược lên 1 cấp (..) rồi vào thư mục data/
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 
    
    # Trỏ đến đúng các thư mục bạn mong muốn
    INPUT_DIR = os.path.join(PROJECT_ROOT, "data_prep", "data", "dirty_png")
    CLEAN_DIR = os.path.join(PROJECT_ROOT, "data_prep", "data", "clean_png")
    SVG_DIR = os.path.join(PROJECT_ROOT, "data_prep", "data", "svg")

    print(f"DEBUG: PATH INPUT = {INPUT_DIR}")
    
    process_pipeline(INPUT_DIR, CLEAN_DIR, SVG_DIR)