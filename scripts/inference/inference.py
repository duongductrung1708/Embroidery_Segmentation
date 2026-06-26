import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IMPORT U2NET thay vì UNet
from src.model import U2NET 

# ==========================================
# 1. CẤU HÌNH V6 (U-2-NET)
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
PATCH_SIZE = 512
MODEL_PATH = "checkpoints/lineart/u2net_best.pth"  # Đổi tên file trọng số cho đúng với U2-Net
TEST_DIR = "./data/lineart/test/"               
OUTPUT_DIR = "./data/lineart/test/predictions/" 

RESIZE_FACTOR = 0.5            

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Khởi tạo U-2-Net thay vì U-Net
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
                
                # --- SỬA Ở ĐÂY: U2-Net trả về 7 outputs ---
                outputs = model(input_tensor)
                
                # Chúng ta chỉ lấy output đầu tiên (d0 - kết quả sắc nét nhất đã gộp từ 6 chặng)
                final_output = outputs[0] 
                
                probs = torch.softmax(final_output, dim=1) 
                fill_probs = probs[0, 1, :, :].cpu().numpy()
                padded_prob_canvas[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = fill_probs

    prob_resized = padded_prob_canvas[:new_h, :new_w]
    raw_final_prob = cv2.resize(prob_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # BỘ LỌC ĐƯỜNG CONG TỰ NHIÊN
    smoothed_prob = cv2.GaussianBlur(raw_final_prob, (5, 5), 0)
    NEW_CONFIDENCE_THRESHOLD = 0.65
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

def batch_inference_and_display(test_dir, save_output=True):
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
    
    if not image_paths:
        print(f"Khong tim thay anh nao trong thu muc: {test_dir}")
        return

    num_images = len(image_paths)
    print(f"\nBat dau phan tich hang loat {num_images} buc anh bang U-2-Net...")

    MAX_DISPLAY = 10 
    display_count = min(num_images, MAX_DISPLAY)
    
    cols = 5 
    rows = 2 
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8)) 
    axes = np.array(axes).reshape(rows, cols)

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        print(f"[{idx+1}/{num_images}] Dang xu ly: {filename}")
        
        overlay_img, fill_mask = process_image(img_path)

        if save_output:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"overlay_{filename}"), cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_{filename}"), fill_mask)

        if idx < MAX_DISPLAY:
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            ax.imshow(overlay_img)
            ax.set_title(filename[:15] + "..." if len(filename) > 15 else filename) 
            ax.axis('off')

    for i in range(display_count, rows * cols):
        r = i // cols
        c = i % cols
        fig.delaxes(axes[r, c])

    print(f"\nHOAN THANH! Da luu anh Fill Overlay vao thu muc '{OUTPUT_DIR}'.")
    
    if display_count > 0:
        print(f"Dang hien thi truoc {display_count} anh mau tren man hinh...")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    batch_inference_and_display(TEST_DIR, save_output=True)