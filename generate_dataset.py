import os
import glob
import math
import pyembroidery
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# Cấu hình Cắt Lát (Patching)
PATCH_SIZE = 512
STRIDE = 256
CONSTANT_SCALE = 3.0
THRESHOLD = 30

# Hàm tự động tạo và làm sạch thư mục
def setup_directories():
    dirs = [
        "data/train/images", "data/train/masks",
        "data/val/images", "data/val/masks",
        "data/test/images", "data/test/masks"
    ]
    print("Đang dọn dẹp và thiết lập cấu trúc thư mục mới...")
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d) # Xóa sạch ảnh cũ nếu có
        os.makedirs(d, exist_ok=True)

def process_file_to_patches(filepath, out_img_dir, out_mask_dir):
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    
    pattern = pyembroidery.read(filepath)
    coords = [s for s in pattern.stitches if s[2] == 0] 
    if len(coords) < 2: return 0
    
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = int((max_x - min_x) * CONSTANT_SCALE)
    height = int((max_y - min_y) * CONSTANT_SCALE)
    
    pad = PATCH_SIZE
    img_h, img_w = height + pad * 2, width + pad * 2
    
    if img_h > 20000 or img_w > 20000: return 0

    large_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    large_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    for i in range(len(coords) - 1):
        dist_orig = math.hypot(coords[i+1][0] - coords[i][0], coords[i+1][1] - coords[i][1])
        
        x1 = int((coords[i][0] - min_x) * CONSTANT_SCALE) + pad
        y1 = int((coords[i][1] - min_y) * CONSTANT_SCALE) + pad
        x2 = int((coords[i+1][0] - min_x) * CONSTANT_SCALE) + pad
        y2 = int((coords[i+1][1] - min_y) * CONSTANT_SCALE) + pad
        
        cv2.line(large_img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
        
        if dist_orig > THRESHOLD:
            cv2.line(large_mask, (x1, y1), (x2, y2), 1, thickness=2)
        else:
            cv2.line(large_mask, (x1, y1), (x2, y2), 2, thickness=2)
            
    patch_count = 0
    for y in range(0, img_h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, img_w - PATCH_SIZE + 1, STRIDE):
            img_patch = large_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_patch = large_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            if np.max(mask_patch) > 0:
                cv2.imwrite(f"{out_img_dir}/{name}_y{y}_x{x}.jpg", img_patch)
                cv2.imwrite(f"{out_mask_dir}/{name}_y{y}_x{x}.png", mask_patch)
                patch_count += 1
    return patch_count

# --- CHẠY CHƯƠNG TRÌNH ---
setup_directories()

# Đọc và SẮP XẾP file theo tên (Cực kỳ quan trọng để đảm bảo 1->21 vào Train)
raw_dir = "data/raw"
dst_files = sorted(glob.glob(f"{raw_dir}/**/*.dst", recursive=True) + glob.glob(f"{raw_dir}/**/*.DST", recursive=True))

print(f"Tìm thấy tổng cộng {len(dst_files)} file .dst")

total_train, total_val, total_test = 0, 0, 0

for idx, f in enumerate(tqdm(dst_files, desc="Đang xử lý")):
    # Logic chia data (Dựa trên Index của danh sách đã sắp xếp)
    if idx < 21:      # 21 file đầu
        img_dir, mask_dir = "data/train/images", "data/train/masks"
        total_train += process_file_to_patches(f, img_dir, mask_dir)
    elif idx < 24:    # 3 file tiếp theo
        img_dir, mask_dir = "data/val/images", "data/val/masks"
        total_val += process_file_to_patches(f, img_dir, mask_dir)
    else:             # 3 file cuối cùng
        img_dir, mask_dir = "data/test/images", "data/test/masks"
        total_test += process_file_to_patches(f, img_dir, mask_dir)

print("\nHOÀN THÀNH TẠO DATASET!")
print(f"Train: {total_train} ảnh | Val: {total_val} ảnh | Test: {total_test} ảnh")
