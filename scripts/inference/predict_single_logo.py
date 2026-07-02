import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Thêm đường dẫn gốc để import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.model import U2NET 

# ==========================================
# 1. CẤU HÌNH DỰ ĐOÁN LOGO 3 CLASS
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
IMAGE_SIZE = 768 
MODEL_PATH = "checkpoints/logo/checkpoints_logo_u2net_logo_best.pth" 

SINGLE_IMAGE_PATH = "./data/test/logo/test2_logo.png"
OUTPUT_DIR = "./data/test/logo/predictions/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CẬP NHẬT: Khởi tạo U-2-Net (Bắt buộc in_ch=4 cho ảnh Transparent)
model = U2NET(in_ch=4, out_ch=3).to(DEVICE)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f"-> Da nap thanh cong bo nao AI LOGO (U-2-NET 4 Kênh) - Size {IMAGE_SIZE}x{IMAGE_SIZE}!")
else:
    print(f"-> Khong tim thay file model '{MODEL_PATH}'. Ban da train xong chua?")
    exit()

def process_image(img_path):
    # ==========================================
    # BƯỚC 1: ĐỌC ẢNH VÀ CHUẨN BỊ CANVAS (RGBA)
    # ==========================================
    img_rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_rgba is None:
        raise ValueError(f"Không thể đọc ảnh: {img_path}")

    orig_h, orig_w = img_rgba.shape[:2]

    # Đảm bảo 4 kênh
    if len(img_rgba.shape) == 2:
        img_rgba = cv2.cvtColor(img_rgba, cv2.COLOR_GRAY2BGRA)
    elif img_rgba.shape[2] == 3:
        img_rgba = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2BGRA)

    img_bgr = img_rgba[:, :, :3].copy() 

    # Tính toán tỷ lệ
    scale = min(IMAGE_SIZE / orig_h, IMAGE_SIZE / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    
    # Resize giữ nguyên 4 kênh
    resized_input = cv2.resize(img_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Tính Padding
    pad_top = (IMAGE_SIZE - new_h) // 2
    pad_bottom = IMAGE_SIZE - new_h - pad_top
    pad_left = (IMAGE_SIZE - new_w) // 2
    pad_right = IMAGE_SIZE - new_w - pad_left
    
    # Bọc thêm viền trong suốt [0, 0, 0, 0] bằng OpenCV
    padded_input = cv2.copyMakeBorder(resized_input, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

    # ==========================================
    # BƯỚC 2: AI PHÂN TÍCH
    # ==========================================
    # PyTorch yêu cầu (Channel, Height, Width)
    padded_input = padded_input.transpose(2, 0, 1) 
    
    input_tensor = torch.tensor(padded_input, dtype=torch.float32).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        final_output = outputs[0] 
        pred_mask = torch.argmax(final_output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # ==========================================
    # BƯỚC 3: CẮT TỈA VÀ HIỂN THỊ
    # ==========================================
    crop_mask = pred_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    
    final_mask = cv2.resize(crop_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    color_mask[final_mask == 1] = [255, 255, 0]   # Cyan
    color_mask[final_mask == 2] = [255, 0, 255]   # Magenta   
    
    overlay_img = cv2.addWeighted(img_bgr, 0.6, color_mask, 0.4, 0)

    return overlay_img, color_mask, img_rgba

def single_image_inference(img_path, save_output=True):
    if not os.path.exists(img_path):
        print(f"\n[LỖI] Khong tim thay anh tai duong dan: {img_path}")
        return

    filename = os.path.basename(img_path)
    print(f"\n[*] Bat dau phan tich Logo: {filename}")
    
    overlay_img, color_mask, img_rgba = process_image(img_path)

    if save_output:
        out_overlay = os.path.join(OUTPUT_DIR, f"overlay_{filename}")
        out_mask = os.path.join(OUTPUT_DIR, f"mask_{filename}")
        
        cv2.imwrite(out_overlay, overlay_img)
        cv2.imwrite(out_mask, color_mask)
        print(f"-> Da luu ket qua tai: {OUTPUT_DIR}")

    print("[*] Dang hien thi ket qua...")
    
    img_bgr = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3:4] / 255.0
    bg_check = np.full_like(img_bgr, 128)
    img_display = (img_bgr * alpha + bg_check * (1 - alpha)).astype(np.uint8)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title("Anh Goc (Lót nen xam)")
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