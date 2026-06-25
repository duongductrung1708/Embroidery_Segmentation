import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# IMPORT U2NET thay vì UNet
from src.model import U2NET 

# ==========================================
# 1. CẤU HÌNH V6 (U-2-NET) - SINGLE IMAGE
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
PATCH_SIZE = 512
MODEL_PATH = "u2net_best.pth"  # File trọng số U-2-Net

# ---> Cấu hình đường dẫn cho 1 ẢNH DUY NHẤT ở đây <---
SINGLE_IMAGE_PATH = "./data/test/test11.png" # SỬA TÊN FILE CỦA BẠN TẠI ĐÂY
OUTPUT_DIR = "./data/test/predictions/"

RESIZE_FACTOR = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Khởi tạo U-2-Net
model = U2NET(in_ch=1, out_ch=2).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("Da nap thanh cong bo nao AI V6 (U-2-NET)!")
else:
    print(f"Khong tim thay file model '{MODEL_PATH}'. Ban da train xong chua?")
    exit()

def process_image(img_path):
    """
    Hàm xử lý lõi: Chỉ chạy AI và khoanh vùng Fill (màu Xanh Lá), có nền TRONG SUỐT.
    """
    # ==========================================
    # BƯỚC 1: ĐỌC ẢNH VÀ TẠO MASK NÉT MỰC GỐC
    # ==========================================
    rgba_img = Image.open(img_path).convert("RGBA")
    orig_w, orig_h = rgba_img.size
    alpha_channel_orig = np.array(rgba_img.getchannel("A"), dtype=np.float32)

    ink_mask = (alpha_channel_orig > 127).astype(np.uint8) * 255

    new_w = int(orig_w * RESIZE_FACTOR)
    new_h = int(orig_h * RESIZE_FACTOR)
    alpha_channel_resized = cv2.resize(alpha_channel_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_h = (PATCH_SIZE - new_h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - new_w % PATCH_SIZE) % PATCH_SIZE
    padded_alpha = np.pad(alpha_channel_resized, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    padded_prob_canvas = np.zeros_like(padded_alpha, dtype=np.float32)

    # ==========================================
    # BƯỚC 2: AI DỰ ĐOÁN VÙNG FILL (BẰNG U-2-NET)
    # ==========================================
    with torch.no_grad():
        for y in range(0, padded_alpha.shape[0] - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, padded_alpha.shape[1] - PATCH_SIZE + 1, PATCH_SIZE):
                patch = padded_alpha[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                if np.max(patch) == 0: continue
                
                input_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0) / 255.0
                input_tensor = input_tensor.to(DEVICE)
                
                outputs = model(input_tensor)
                
                # U2-Net lấy output đầu tiên (d0)
                final_output = outputs[0] 
                
                probs = torch.softmax(final_output, dim=1) 
                fill_probs = probs[0, 1, :, :].cpu().numpy()
                padded_prob_canvas[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = fill_probs

    prob_resized = padded_prob_canvas[:new_h, :new_w]
    raw_final_prob = cv2.resize(prob_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # BỘ LỌC ĐƯỜNG CONG TỰ NHIÊN
    smoothed_prob = cv2.GaussianBlur(raw_final_prob, (5, 5), 0)
    
    # BẠN CÓ THỂ ĐIỀU CHỈNH NGƯỠNG NÀY ĐỂ AI TÔ BẠO DẠN HƠN HOẶC CẨN THẬN HƠN
    NEW_CONFIDENCE_THRESHOLD = 0.50 # Đã hạ xuống 0.5 theo tư vấn để lấp đầy tốt hơn
    
    binary_mask = (smoothed_prob > NEW_CONFIDENCE_THRESHOLD).astype(np.uint8) * 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.erode(cleaned_mask, kernel_erode, iterations=1)

    fill_mask = (cleaned_mask > 127).astype(np.uint8) * 255

    # ==========================================
    # BƯỚC 3: TẠO ẢNH OVERLAY (CÓ NỀN TRONG SUỐT RGBA)
    # ==========================================
    display_img = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    display_img[ink_mask > 0] = [0, 0, 0, 255] 
    display_img[fill_mask > 0] = [0, 255, 0, 255]

    return display_img, fill_mask

def single_image_inference(img_path, save_output=True):
    """
    Xử lý và hiển thị một bức ảnh duy nhất
    """
    if not os.path.exists(img_path):
        print(f"\n[LỖI] Khong tim thay anh tai duong dan: {img_path}")
        print("Vui long kiem tra lai bien SINGLE_IMAGE_PATH!")
        return

    filename = os.path.basename(img_path)
    print(f"\nBat dau phan tich anh: {filename}")
    
    # Chạy AI
    overlay_img, fill_mask = process_image(img_path)

    # Lưu kết quả
    if save_output:
        out_overlay = os.path.join(OUTPUT_DIR, f"overlay_{filename}")
        out_mask = os.path.join(OUTPUT_DIR, f"mask_{filename}")
        
        cv2.imwrite(out_overlay, cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(out_mask, fill_mask)
        print(f"-> Da luu anh Overlay tai: {out_overlay}")
        print(f"-> Da luu anh Mask trang/den tai: {out_mask}")

    # Hiển thị lên màn hình
    print("Dang hien thi ket qua...")
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_img)
    plt.title(f"AI Prediction: {filename}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Gọi hàm xử lý ảnh đơn
    single_image_inference(SINGLE_IMAGE_PATH, save_output=True)