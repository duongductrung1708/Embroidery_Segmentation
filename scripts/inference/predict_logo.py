#!/usr/bin/env python3
"""
Embroidery Segmentation - Batch Inference Script
Quét toàn bộ ảnh trong thư mục, đưa qua U2-Net và lưu kết quả hàng loạt.
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm # Thêm thư viện tạo thanh tiến trình
import matplotlib.pyplot as plt

# 1. ĐỊNH VỊ ĐƯỜNG DẪN DỰ ÁN
PROJECT_ROOT = Path(__file__).resolve().parent
if "scripts" in PROJECT_ROOT.parts or "training" in PROJECT_ROOT.parts:
    PROJECT_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import U2NET

# ==========================================
# CẤU HÌNH CÁC THAM SỐ THỬ NGHIỆM
# ==========================================
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints/logo/checkpoints_logo_u2net_logo_best.pth")
TEST_DIR = os.path.join(PROJECT_ROOT, "data/test/logo")         # Thư mục chứa ảnh cần test
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/test/logo/predictions") # Nơi cất kết quả dự đoán
IMAGE_SIZE = 512

def load_testing_model(weights_path, device):
    """Khởi tạo cấu trúc mạng U2-Net và nạp file trọng số đã train."""
    print(f"[*] Đang tải bộ não U2-Net (1 kênh vào, 3 kênh ra)...")
    model = U2NET(in_ch=1, out_ch=3)
    
    if not os.path.exists(weights_path):
        print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy file trọng số tại {weights_path}")
        sys.exit(1)
        
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print("-> Nạp file trọng số thành công! Mô hình đã sẵn sàng.")
    return model

def color_code_mask(mask_argmax):
    """
    Biến ma trận nhãn thành ảnh màu trực quan
    0: Đen (Nền), 1: Xanh lá (Fill/Tatami), 2: Đỏ (Satin)
    """
    h, w = mask_argmax.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    color_mask[mask_argmax == 1] = [0, 255, 0]   # Fill -> Green
    color_mask[mask_argmax == 2] = [0, 0, 255]   # Satin -> Red
    
    return color_mask

def main():
    # 2. THIẾT LẬP THIẾT BỊ VÀ THƯ MỤC
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"[*] Đang chạy test trên thiết bị: {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. TẢI MÔ HÌNH (Chỉ tải 1 lần duy nhất ở ngoài vòng lặp)
    model = load_testing_model(MODEL_WEIGHTS_PATH, device)
    
    # 4. TÌM KIẾM ẢNH TRONG THƯ MỤC
    if not os.path.exists(TEST_DIR):
        print(f"LỖI: Không tìm thấy thư mục ảnh test tại {TEST_DIR}")
        return
        
    valid_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_paths = [p for p in Path(TEST_DIR).glob("*") if p.suffix in valid_extensions]
    
    if not image_paths:
        print(f"Thư mục {TEST_DIR} hiện đang trống hoặc không có file ảnh hợp lệ.")
        return
        
    print(f"\n[*] Tìm thấy {len(image_paths)} ảnh. Bắt đầu quét hàng loạt...")
    
    # Transform cố định cho mọi ảnh
    transform = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        ToTensorV2()
    ])
    
    # 5. VÒNG LẶP XỬ LÝ TỪNG ẢNH
    for img_path in tqdm(image_paths, desc="Dự đoán ảnh"):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
            
        orig_h, orig_w = img_bgr.shape[:2]
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Tiền xử lý
        transformed = transform(image=img_gray)
        input_tensor = transformed['image'].unsqueeze(0).float().to(device)
        
        # Suy luận
        with torch.no_grad():
            outputs = model(input_tensor)
            primary_output = outputs[0] 
            pred_mask = torch.argmax(primary_output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Hậu xử lý màu sắc
        predicted_color_mask = color_code_mask(pred_mask)
        
        # Nghịch đảo biến đổi (Inverse Transform) để đưa mask về kích thước gốc
        scale = min(IMAGE_SIZE / orig_h, IMAGE_SIZE / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        pad_top = (IMAGE_SIZE - new_h) // 2
        pad_left = (IMAGE_SIZE - new_w) // 2
        
        crop_mask = predicted_color_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
        final_mask = cv2.resize(crop_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # Overlay
        overlay_img = cv2.addWeighted(img_bgr, 0.6, final_mask, 0.4, 0)
        
        # Lưu file (Thêm hậu tố để phân biệt)
        filename = img_path.stem
        mask_out_path = os.path.join(OUTPUT_DIR, f"{filename}_mask.png")
        overlay_out_path = os.path.join(OUTPUT_DIR, f"{filename}_overlay.png")
        
        cv2.imwrite(mask_out_path, final_mask)
        cv2.imwrite(overlay_out_path, overlay_img)

        # [MỚI] VISUALIZATION BẰNG MATPLOTLIB
        plt.figure(figsize=(12, 4))
        
        # Ảnh 1: Ảnh gốc
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Ảnh gốc")
        plt.axis('off')
        
        # Ảnh 2: Mask dự đoán
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB))
        plt.title("Mask dự đoán")
        plt.axis('off')
        
        # Ảnh 3: Overlay kết quả
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
        plt.title("Kết quả Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show(block=False) # Không chặn vòng lặp
        plt.pause(2.0)        # Dừng 2 giây cho bạn xem, rồi tự đóng
        plt.close()           # Đóng cửa sổ để giải phóng RAM
        
    print(f"\nHOÀN THÀNH BATCH INFERENCE!")
    print(f"-> Đã xử lý xong {len(image_paths)} ảnh.")
    print(f"-> Bạn có thể kiểm tra kết quả Overlay tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
