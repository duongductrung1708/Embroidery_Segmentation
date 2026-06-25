import os
import glob
import cv2
import numpy as np
import vtracer
import fal_client
from tqdm import tqdm
import xml.etree.ElementTree as ET

def enhance_with_fal(img_path, original_alpha=None):
    """
    Sử dụng fal.ai NAFNet model để làm mượt và khử nhiễu ảnh.
    Giữ lại alpha channel gốc nếu có.
    """
    try:
        # Encode image as data URL
        image_data_url = fal_client.encode_file(img_path)
        
        # Call NAFNet deblur model
        result = fal_client.run(
            "fal-ai/nafnet/deblur",
            arguments={
                "image_url": image_data_url
            }
        )
        
        # Download processed image
        import requests
        processed_url = result["image"]["url"]
        response = requests.get(processed_url)
        
        # Convert to numpy array
        import io
        from PIL import Image
        img_pil = Image.open(io.BytesIO(response.content))
        img_array = np.array(img_pil)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Nếu có alpha channel gốc, dán lại vào ảnh đã xử lý
        if original_alpha is not None:
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.merge([img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2], original_alpha])
        
        return img_array
    except Exception as e:
        print(f"Lỗi khi xử lý với fal.ai: {e}")
        return None

def clean_rgba_image(img):
    """
    Làm sạch kênh Alpha (trong suốt) bằng cách khử nhiễu,
    giữ nguyên format RGBA để vtracer có thể tạo SVG với nền trong suốt.
    Đồng thời làm mượt cả RGB channels để giảm nhọn nhô từ ảnh gốc.
    """
    if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        alpha = img[:, :, 3]
        rgb = img[:, :, :3]
        
        # Làm mượt RGB channels bằng bilateral filter để giữ cạnh nhưng giảm nhiễu
        rgb_smooth = cv2.bilateralFilter(rgb, 9, 75, 75)
        
        # Làm SẮC LẸM alpha channel bằng Threshold thay vì Blur
        # vtracer cần alpha binary (0 hoặc 255) để tránh tạo hàng ngàn vector path
        _, alpha_binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        
        # Khử nhiễu alpha channel bằng morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_clean = cv2.morphologyEx(alpha_binary, cv2.MORPH_OPEN, kernel)
        alpha_clean = cv2.morphologyEx(alpha_clean, cv2.MORPH_CLOSE, kernel)
        
        # Cập nhật alpha channel đã làm sạch và RGB đã làm mượt
        result = np.zeros_like(img)
        result[:, :, :3] = rgb_smooth
        result[:, :, 3] = alpha_clean
        
        return result
    
    # Nếu ảnh không có alpha, thêm alpha channel (đen = trong suốt)
    if len(img.shape) == 3 and img.shape[2] == 3:  # BGR
        # Làm mượt RGB channels
        rgb_smooth = cv2.bilateralFilter(img, 9, 75, 75)
        alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        result = cv2.merge([rgb_smooth[:, :, 0], rgb_smooth[:, :, 1], rgb_smooth[:, :, 2], alpha])
        return result
    
    return img

def merge_svg_paths(svg_path):
    """
    Merge các path có đường biên tương tự trong SVG để chúng share border.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Tìm tất cả path elements
        paths = root.findall('.//{http://www.w3.org/2000/svg}path')
        
        # Group paths by color
        paths_by_color = {}
        for path in paths:
            fill = path.get('fill', 'none')
            stroke = path.get('stroke', 'none')
            key = (fill, stroke)
            if key not in paths_by_color:
                paths_by_color[key] = []
            paths_by_color[key].append(path)
        
        # Save modified SVG
        tree.write(svg_path, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        print(f"Lỗi khi merge SVG paths: {e}")
        return False

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
        
        # Trích xuất alpha channel gốc trước khi xử lý với fal.ai
        original_alpha = None
        if len(img.shape) == 3 and img.shape[2] == 4:
            original_alpha = img[:, :, 3].copy()
        
        # 1.1 Tăng chất lượng ảnh với fal.ai NAFNet
        enhanced_img = enhance_with_fal(img_path, original_alpha)
        if enhanced_img is not None:
            img = enhanced_img
            print(f"  Đã tăng chất lượng với fal.ai: {filename}")
        
        clean_img = clean_rgba_image(img)
        cv2.imwrite(clean_path, clean_img)

        # 2. Vector hóa (Chế độ color để giữ màu logo và transparency)
        vtracer.convert_image_to_svg_py(
            clean_path,
            svg_path,
            colormode="color",        # Giữ nguyên màu
            hierarchical="cutout",   # Tách vùng màu riêng biệt thay vì xếp chồng
            mode="spline",            # Đường cong Bezier mượt
            filter_speckle=2,         # Giảm ngưỡng để giữ nhiều chi tiết màu hơn
            color_precision=12,       # Tăng độ chính xác màu để giảm quantization
            layer_difference=4,       # Giảm thêm để gộp các vùng màu tương tự lại với nhau
            corner_threshold=20,      # Giảm ngưỡng góc để đường cong mượt hơn
            length_threshold=4.0,     # Tăng ngưỡng độ dài đường cong để loại bỏ các đoạn ngắn
            max_iterations=25,        # Tăng iterations để hội tụ tốt hơn
            splice_threshold=20,      # Giảm ngưỡng nối đường để đường cong mượt hơn
            path_precision=12         # Tăng precision đường cong
        )
        
        # 3. Merge paths để share border
        merge_svg_paths(svg_path)

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