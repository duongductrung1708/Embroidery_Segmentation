import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Thêm đường dẫn gốc để import src
# Lùi 3 cấp: predict_single_logo.py -> inference -> scripts -> Embroidery_Segmentation
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# -------------------------

from src.model import U2NET 

# ==========================================
# 1. CẤU HÌNH DỰ ĐOÁN LOGO 3 CLASS
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
IMAGE_SIZE = 512
MODEL_PATH = "checkpoints/logo/checkpoints_logo_u2net_logo_best.pth"  # File trọng số Logo của bạn

# ---> Cấu hình đường dẫn cho 1 ẢNH DUY NHẤT ở đây <---
SINGLE_IMAGE_PATH = "./data/test/logo/test1_logo.png" # SỬA TÊN FILE CỦA BẠN TẠI ĐÂY
OUTPUT_DIR = "./data/test/logo/predictions/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Khởi tạo U-2-Net (Bắt buộc in_ch=1, out_ch=3 đối với Logo)
model = U2NET(in_ch=1, out_ch=3).to(DEVICE)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    # Hỗ trợ cả file checkpoint đầy đủ hoặc chỉ weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("-> Da nap thanh cong bo nao AI LOGO (U-2-NET)!")
else:
    print(f"-> Khong tim thay file model '{MODEL_PATH}'. Ban da train xong chua?")
    exit()

def process_image(img_path):
    """
    Hàm xử lý lõi Logo: 
    Resize -> Padding -> AI Predict -> Crop ngược -> Đổ màu (Xanh/Đỏ).
    """
    # ==========================================
    # BƯỚC 1: ĐỌC ẢNH VÀ CHUẨN BỊ CANVAS
    # ==========================================
    img_rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_rgba is None:
        raise ValueError(f"Không thể đọc ảnh: {img_path}")

    orig_h, orig_w = img_rgba.shape[:2]

    # Nếu ảnh có alpha thì lấy alpha giống lúc train
    if img_rgba.shape[2] == 4:
        input_image = img_rgba[:, :, 3]
    else:
        # fallback nếu ảnh không có alpha
        input_image = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2GRAY)

    # chỉ dùng để hiển thị
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
    
    # Tính toán tỷ lệ để thu gọn cạnh dài nhất về 512
    scale = min(IMAGE_SIZE / orig_h, IMAGE_SIZE / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    
    resized_gray = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Bọc thêm viền đen (Padding) để ảnh đạt đúng khung 512x512
    pad_top = (IMAGE_SIZE - new_h) // 2
    pad_bottom = IMAGE_SIZE - new_h - pad_top
    pad_left = (IMAGE_SIZE - new_w) // 2
    pad_right = IMAGE_SIZE - new_w - pad_left
    
    padded_gray = cv2.copyMakeBorder(resized_gray, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    # ==========================================
    # BƯỚC 2: AI PHÂN TÍCH (U-2-NET)
    # ==========================================
    # Đưa ảnh uint8 (0-255) về Tensor float32 (0-1)
    input_tensor = torch.tensor(padded_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        final_output = outputs[0] 
        
        # Argmax để chọn lớp có xác suất cao nhất (0: Nền, 1: Fill, 2: Satin)
        pred_mask = torch.argmax(final_output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # ==========================================
    # BƯỚC 3: CẮT TỈA VÀ HIỂN THỊ (HẬU XỬ LÝ)
    # ==========================================
    # Cắt bỏ cái viền đen đã đắp vào ở Bước 1
    crop_mask = pred_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    
    # Phóng to ảnh dự đoán về lại đúng số pixel nguyên bản của Logo
    final_mask = cv2.resize(crop_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Đổ màu chuẩn: Lớp 1 (Fill) -> Cyan | Lớp 2 (Satin) -> Magenta
    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    color_mask[final_mask == 1] = [255, 255, 0]   # BGR của OpenCV: Cyan
    color_mask[final_mask == 2] = [255, 0, 255]   # BGR của OpenCV: Magenta   
    
    # Tạo Overlay trộn 60% ảnh gốc, 40% Mask màu
    overlay_img = cv2.addWeighted(img_bgr, 0.6, color_mask, 0.4, 0)

    return overlay_img, color_mask

def single_image_inference(img_path, save_output=True):
    """
    Kích hoạt quá trình dự đoán và Render lên Matplotlib.
    """
    if not os.path.exists(img_path):
        print(f"\n[LỖI] Khong tim thay anh tai duong dan: {img_path}")
        return

    filename = os.path.basename(img_path)
    print(f"\n[*] Bat dau phan tich Logo: {filename}")
    
    overlay_img, color_mask = process_image(img_path)

    if save_output:
        out_overlay = os.path.join(OUTPUT_DIR, f"overlay_{filename}")
        out_mask = os.path.join(OUTPUT_DIR, f"mask_{filename}")
        
        cv2.imwrite(out_overlay, overlay_img)
        cv2.imwrite(out_mask, color_mask)
        print(f"-> Da luu anh Overlay tai: {out_overlay}")
        print(f"-> Da luu anh Mask tai: {out_mask}")

    # Giao diện Matplotlib trực quan (Chuyển BGR -> RGB)
    print("[*] Dang hien thi ket qua...")
    img_bgr = cv2.imread(img_path)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Anh Goc")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
    plt.title("AI Mask (Cyan=Fill, Magenta=Satin)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.title("Overlay Ket Qua")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    single_image_inference(SINGLE_IMAGE_PATH, save_output=True)