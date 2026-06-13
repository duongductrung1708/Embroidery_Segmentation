import os
import glob
import math
import pyembroidery
import cv2
import numpy as np
from tqdm import tqdm

# Cấu hình đường dẫn
RAW_DIR = "data/raw"
IMG_DIR = "data/images"
MASK_DIR = "data/masks"

# Cấu hình Cắt Lát (Patching)
PATCH_SIZE = 512
STRIDE = 256          # Bước nhảy 256 giúp các bức ảnh có độ lợp (overlap) lên nhau, AI học ngữ cảnh tốt hơn
CONSTANT_SCALE = 3.0  # VŨ KHÍ 1: Tỷ lệ cố định. Tùy chỉnh (2.0 - 5.0) sao cho sợi chỉ rõ nét nhất
THRESHOLD = 30        # Ngưỡng mũi kim thực tế (Đo trên file dst gốc)

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

def process_file_to_patches(filepath):
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    
    # 1. Đọc file
    pattern = pyembroidery.read(filepath)
    coords = [s for s in pattern.stitches if s[2] == 0] 
    if len(coords) < 2: return 0
    
    # 2. Tìm kích thước thật
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Tính kích thước bức tranh khổng lồ với tỷ lệ CỐ ĐỊNH
    width = int((max_x - min_x) * CONSTANT_SCALE)
    height = int((max_y - min_y) * CONSTANT_SCALE)
    
    # Thêm lề (padding) để không bị cắt lẹm ở mép
    pad = PATCH_SIZE
    img_h, img_w = height + pad * 2, width + pad * 2
    
    # Bỏ qua nếu file quá khủng khiếp gây tràn RAM máy tính
    if img_h > 20000 or img_w > 20000:
        return 0

    # 3. Tạo Canvas Khổng Lồ Siêu Nét
    large_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    large_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # Vẽ toàn bộ chi tiết lên bức tranh khổng lồ
    for i in range(len(coords) - 1):
        dist_orig = math.hypot(coords[i+1][0] - coords[i][0], coords[i+1][1] - coords[i][1])
        
        x1 = int((coords[i][0] - min_x) * CONSTANT_SCALE) + pad
        y1 = int((coords[i][1] - min_y) * CONSTANT_SCALE) + pad
        x2 = int((coords[i+1][0] - min_x) * CONSTANT_SCALE) + pad
        y2 = int((coords[i+1][1] - min_y) * CONSTANT_SCALE) + pad
        
        # Dùng thickness=2 để sợi chỉ mập mạp, AI dễ nhìn hơn
        cv2.line(large_img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
        
        if dist_orig > THRESHOLD:
            cv2.line(large_mask, (x1, y1), (x2, y2), 1, thickness=2) # Satin
        else:
            cv2.line(large_mask, (x1, y1), (x2, y2), 2, thickness=2) # Tatami
            
    # 4. VŨ KHÍ 2: Cắt Lát (Sliding Window)
    patch_count = 0
    # Quét cái khuôn 512x512 qua lại trên bức tranh lớn
    for y in range(0, img_h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, img_w - PATCH_SIZE + 1, STRIDE):
            img_patch = large_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_patch = large_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Chỉ lưu những mảnh cắt CÓ CHỨA NÉT THÊU (Bỏ qua ảnh đen xì)
            if np.max(mask_patch) > 0:
                cv2.imwrite(f"{IMG_DIR}/{name}_y{y}_x{x}.jpg", img_patch)
                cv2.imwrite(f"{MASK_DIR}/{name}_y{y}_x{x}.png", mask_patch)
                patch_count += 1
                
    return patch_count

# --- QUÉT VÀ THỰC THI ---
print("Đang dọn dẹp ảnh cũ...")
for f in glob.glob(f"{IMG_DIR}/*.jpg") + glob.glob(f"{MASK_DIR}/*.png"):
    os.remove(f)

print("Đang tiến hành tạo Dataset cắt lát siêu nét...")
dst_files = glob.glob(f"{RAW_DIR}/**/*.dst", recursive=True) + glob.glob(f"{RAW_DIR}/**/*.DST", recursive=True)

total_patches = 0
for f in tqdm(dst_files):
    total_patches += process_file_to_patches(f)
        
print(f"Hoàn thành! Đã sinh ra tổng cộng {total_patches} bức ảnh siêu nét.")
