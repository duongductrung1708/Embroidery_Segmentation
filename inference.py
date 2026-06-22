import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Import model của bạn (nhớ cấu trúc thư mục src)
from src.model import UNet 

# ==========================================
# 1. CẤU HÌNH V4 PRO
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
PATCH_SIZE = 512
MODEL_PATH = "unet_binary_best.pth" 

RESIZE_FACTOR = 0.5            

# Khởi tạo và nạp tệp trọng số (bộ não AI)
model = UNet(in_channels=1, out_channels=2).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("Đã nạp thành công bộ não AI V4 PRO!")
else:
    print("Không tìm thấy file model. Bạn đã train xong chưa?")
    exit()

def predict_full_image(img_path, save_output=True):
    print(f"\nĐang phân tích ảnh: {img_path}")
    
    # ==========================================
    # 2. ĐỌC ẢNH VÀ ĐỒNG BỘ SCALE (RESIZE)
    # ==========================================
    rgba_img = Image.open(img_path).convert("RGBA")
    orig_w, orig_h = rgba_img.size
    alpha_channel_orig = np.array(rgba_img.getchannel("A"), dtype=np.float32)

    new_w = int(orig_w * RESIZE_FACTOR)
    new_h = int(orig_h * RESIZE_FACTOR)
    alpha_channel_resized = cv2.resize(alpha_channel_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ==========================================
    # 3. KỸ THUẬT PADDING 
    # ==========================================
    pad_h = (PATCH_SIZE - new_h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - new_w % PATCH_SIZE) % PATCH_SIZE
    
    padded_alpha = np.pad(alpha_channel_resized, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    # [NÂNG CẤP V4 PRO]: Lưu xác suất Float thay vì Binary
    padded_prob_canvas = np.zeros_like(padded_alpha, dtype=np.float32)

    # ==========================================
    # 4. QUÉT SLIDING WINDOW VÀ TRÍCH XUẤT XÁC SUẤT
    # ==========================================
    print(f"AI đang phân tích luồng xác suất trên các đường cong...")
    with torch.no_grad():
        for y in range(0, padded_alpha.shape[0] - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, padded_alpha.shape[1] - PATCH_SIZE + 1, PATCH_SIZE):
                patch = padded_alpha[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                if np.max(patch) == 0: continue
                
                input_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0) / 255.0
                input_tensor = input_tensor.to(DEVICE)
                
                output = model(input_tensor)
                
                probs = torch.softmax(output, dim=1) 
                # Lấy xác suất dạng float (0.0 -> 1.0)
                fill_probs = probs[0, 1, :, :].cpu().numpy()
                
                padded_prob_canvas[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = fill_probs

    # Cắt bỏ phần viền đệm thừa
    prob_resized = padded_prob_canvas[:new_h, :new_w]

    # [NÂNG CẤP V4 PRO]: Phóng to xác suất bằng INTER_LINEAR để giữ đường cong mượt mà
    raw_final_prob = cv2.resize(prob_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # ==========================================
    # 5. BỘ LỌC ĐƯỜNG CONG TỰ NHIÊN (ADVANCED POST-PROCESSING)
    # ==========================================
    print("Đang áp dụng bộ lọc hình học làm mịn đường cong...")

    # Bước 1: Làm mịn luồng xác suất bằng Gaussian Blur (Triệt tiêu lởm chởm)
    smoothed_prob = cv2.GaussianBlur(raw_final_prob, (5, 5), 0)

    # Bước 2: Siết ngưỡng nghiêm ngặt để diệt viền thừa
    NEW_CONFIDENCE_THRESHOLD = 0.65
    binary_mask = (smoothed_prob > NEW_CONFIDENCE_THRESHOLD).astype(np.uint8) * 255

    # Bước 3: Tổ hợp Morphology đóng lỗ thủng bên trong đường cong
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Bước 4: Mài nhẹ viền (Giấy nhám) nếu vẫn còn hơi tràn
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.erode(cleaned_mask, kernel_erode, iterations=1)

    final_cleaned_mask = (cleaned_mask > 127).astype(np.uint8)

    if save_output:
        out_name = "AI_Mask_Cleaned.png"
        cv2.imwrite(out_name, cleaned_mask)
        print(f"Đã lưu Mask hoàn hảo đường cong ra file: {out_name}")

    # ==========================================
    # 6. HIỂN THỊ KẾT QUẢ
    # ==========================================
    plt.figure(figsize=(14, 7)) 
    
    display_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    display_img[alpha_channel_orig > 0] = [255, 255, 255]
    
    plt.subplot(1, 2, 1)
    plt.title("Ảnh gốc (Nét vẽ)")
    plt.imshow(display_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Mask từ AI (Threshold: {NEW_CONFIDENCE_THRESHOLD})")
    overlay_clean = display_img.copy()
    overlay_clean[final_cleaned_mask == 1] = [0, 255, 0] # Màu Xanh Lá
    plt.imshow(overlay_clean)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# CHẠY THỬ VỚI ẢNH CỦA BẠN
predict_full_image("./data/test/test10.png")
