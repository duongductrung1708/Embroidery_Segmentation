#!/usr/bin/env python3
"""
Embroidery Segmentation - Batch Inference Script
Quét toàn bộ ảnh trong thư mục, đưa qua U2-Net và lưu kết quả hàng loạt.
Đã đồng bộ kỹ thuật Global Context (Toàn cảnh) với quá trình Training.
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import matplotlib.pyplot as plt

# 1. ĐỊNH VỊ ĐƯỜNG DẪN DỰ ÁN
# Lùi 3 cấp: predict_logo.py -> inference -> scripts -> Embroidery_Segmentation
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import U2NET

# ==========================================
# CẤU HÌNH CÁC THAM SỐ
# ==========================================
MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints/logo/checkpoints_logo_u2net_logo_best.pth") # Sửa lại tên file pth nếu bạn lưu tên khác
TEST_DIR = os.path.join(PROJECT_ROOT, "data/test/logo")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/test/logo/predictions")
IMAGE_SIZE = 512

def load_testing_model(weights_path, device):
    print(f"[*] Đang tải bộ não U2-Net (1 kênh vào, 3 kênh ra)...")
    model = U2NET(in_ch=1, out_ch=3)
    
    if not os.path.exists(weights_path):
        print(f"LỖI: Không tìm thấy file trọng số tại {weights_path}")
        sys.exit(1)
        
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print("-> Nạp file trọng số thành công!")
    return model

def color_code_mask(mask_argmax):
    """0: Nền (Đen), 1: Fill (Cyan), 2: Satin (Magenta)"""
    h, w = mask_argmax.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask_argmax == 1] = [0, 255, 255]   # Fill -> Cyan
    color_mask[mask_argmax == 2] = [255, 0, 255]   # Satin -> Magenta
    return color_mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_testing_model(MODEL_WEIGHTS_PATH, device)
    
    image_paths = [p for p in Path(TEST_DIR).glob("*") if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    
    if not image_paths:
        print(f"Thư mục {TEST_DIR} trống!")
        return
        
    # TIỀN XỬ LÝ: Đồng bộ 100% với File Train (Chỉ bóp tỷ lệ & lót viền, không crop)
    transform = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, fill=0),
        ToTensorV2()
    ])
    
    for img_path in tqdm(image_paths, desc="Dự đoán"):
        # Đọc ảnh giữ nguyên Alpha channel
        img_rgba = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_rgba is None:
            continue

        orig_h, orig_w = img_rgba.shape[:2]

        # Giữ ảnh BGR để làm overlay
        if img_rgba.ndim == 3 and img_rgba.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)

            # ===============================
            # QUAN TRỌNG:
            # Lấy Alpha channel làm input giống hệt lúc train
            # ===============================
            img_gray = img_rgba[:, :, 3]

        elif img_rgba.ndim == 3:
            img_bgr = img_rgba
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        else:
            # Ảnh grayscale
            img_gray = img_rgba
            img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        
        # 1. Transform ảnh
        transformed = transform(image=img_gray)
        
        # 2. CHUẨN HÓA DỮ LIỆU (/ 255.0) VÀ ĐƯA LÊN GPU
        input_tensor = transformed['image'].unsqueeze(0).float().to(device) / 255.0
        
        # 3. Suy luận
        with torch.no_grad():
            outputs = model(input_tensor)
            pred_mask = torch.argmax(outputs[0], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 4. Hậu xử lý & Crop ngược phần viền đen để về đúng kích thước gốc
        predicted_color_mask = color_code_mask(pred_mask)
        scale = min(IMAGE_SIZE / orig_h, IMAGE_SIZE / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        pad_top = (IMAGE_SIZE - new_h) // 2
        pad_left = (IMAGE_SIZE - new_w) // 2
        
        crop_mask = predicted_color_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
        # Dùng INTER_NEAREST để màu không bị lem
        final_mask = cv2.resize(crop_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        overlay_img = cv2.addWeighted(img_bgr, 0.6, final_mask, 0.4, 0)
        
        # 5. Lưu kết quả
        filename = img_path.stem
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename}_mask.png"), final_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename}_overlay.png"), overlay_img)

        # 6. Lưu file Báo cáo trực quan (Visualization PNG)
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Ảnh gốc")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB))
        plt.title("AI Mask")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
        plt.title("Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}_viz.png"), dpi=100, bbox_inches='tight')
        plt.close(fig) # Đóng figure giải phóng RAM
        
    print(f"\nHOÀN THÀNH BATCH INFERENCE!")
    print(f"-> Đã xử lý xong {len(image_paths)} ảnh.")
    print(f"-> Bạn có thể kiểm tra kết quả tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
