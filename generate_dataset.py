import os
import glob
import math
import pyembroidery
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# Cấu hình
PATCH_SIZE = 512
STRIDE = 256
CONSTANT_SCALE = 3.0

def setup_directories():
    dirs = [
        "data/train/images", "data/train/masks",
        "data/val/images", "data/val/masks",
        "data/test/images", "data/test/masks"
    ]
    print("Đang dọn dẹp và thiết lập cấu trúc thư mục mới...")
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def process_file_to_patches(filepath, out_img_dir, out_mask_dir):
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    
    pattern = pyembroidery.read(filepath)
    coords = [s for s in pattern.stitches if s[2] == 0] 
    if len(coords) < 2: return 0
    
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = int((max_x - min_x) * CONSTANT_SCALE)
    height = int((max_y - min_y) * CONSTANT_SCALE)
    pad = PATCH_SIZE
    img_h, img_w = height + pad * 2, width + pad * 2
    
    if img_h > 20000 or img_w > 20000: return 0

    large_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    temp_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # 1. Vẽ nét mảnh để tính toán, nét mập để AI học
    for i in range(len(coords) - 1):
        x1 = int((coords[i][0] - min_x) * CONSTANT_SCALE) + pad
        y1 = int((coords[i][1] - min_y) * CONSTANT_SCALE) + pad
        x2 = int((coords[i+1][0] - min_x) * CONSTANT_SCALE) + pad
        y2 = int((coords[i+1][1] - min_y) * CONSTANT_SCALE) + pad
        
        cv2.line(large_img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
        cv2.line(temp_mask, (x1, y1), (x2, y2), 255, thickness=1)

    # 2. Phép thuật OpenCV: Cắt Râu Outline
    
    # BƯỚC A: ĐÓNG (CLOSE) - Dính Fill thành mảng đặc
    kernel_close = np.ones((11, 11), np.uint8)
    closed_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # BƯỚC B: MỞ (OPEN) - Cắt đứt râu Outline dính vào Fill
    kernel_open = np.ones((5, 5), np.uint8)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel_open)
    
    # 3. Lọc bỏ các đường viền/chỉ nối (diện tích nhỏ)
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000: # Chỉ giữ lại các mảng thêu lớn (Fill)
            cv2.drawContours(large_mask, [cnt], -1, 255, -1)
            
    # 4. Cắt Lát (Sliding Window)
    patch_count = 0
    for y in range(0, img_h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, img_w - PATCH_SIZE + 1, STRIDE):
            img_patch = large_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_patch = large_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            if np.max(img_patch) > 0: # Lưu nếu có nét thêu
                cv2.imwrite(f"{out_img_dir}/{name}_y{y}_x{x}.jpg", img_patch)
                cv2.imwrite(f"{out_mask_dir}/{name}_y{y}_x{x}.png", mask_patch)
                patch_count += 1
                
    return patch_count

# --- CHẠY CHƯƠNG TRÌNH ---
setup_directories()
raw_dir = "data/raw"
dst_files = sorted(glob.glob(f"{raw_dir}/**/*.dst", recursive=True) + glob.glob(f"{raw_dir}/**/*.DST", recursive=True))

print(f"Tìm thấy tổng cộng {len(dst_files)} file .dst")

total_train, total_val, total_test = 0, 0, 0

for idx, f in enumerate(tqdm(dst_files, desc="Đang xử lý")):
    if idx < 21:      # 21 file đầu vào Train
        total_train += process_file_to_patches(f, "data/train/images", "data/train/masks")
    elif idx < 24:    # 3 file tiếp theo vào Val
        total_val += process_file_to_patches(f, "data/val/images", "data/val/masks")
    else:             # 3 file cuối vào Test
        total_test += process_file_to_patches(f, "data/test/images", "data/test/masks")

print("\nHOÀN THÀNH TẠO DATASET!")
print(f"Train: {total_train} ảnh | Val: {total_val} ảnh | Test: {total_test} ảnh")
