import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# ==========================================
# CẤU HÌNH
# ==========================================
PATCH_SIZE = 512
STRIDE = 256  
BASE_DIR = "data"
RESIZE_FACTOR = 0.5  # Thu nhỏ ảnh gốc 50% trước khi băm

def setup_directories():
    """Tạo cấu trúc thư mục chứa dữ liệu đã băm (CHỈ CÒN TRAIN VÀ VAL)"""
    dirs = [
        f"{BASE_DIR}/train/images", f"{BASE_DIR}/train/masks",
        f"{BASE_DIR}/val/images", f"{BASE_DIR}/val/masks"
    ]
    for d in dirs:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def slice_image_and_mask(img_path, mask_path, out_img_dir, out_mask_dir):
    filename = os.path.basename(img_path)
    name, _ = os.path.splitext(filename)
    
    img_rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if len(img_rgba.shape) == 2 or img_rgba.shape[2] == 3:
        img_rgba = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2BGRA)
        
    if RESIZE_FACTOR != 1.0:
        new_w = int(img_rgba.shape[1] * RESIZE_FACTOR)
        new_h = int(img_rgba.shape[0] * RESIZE_FACTOR)
        img_rgba = cv2.resize(img_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_gray = cv2.resize(mask_gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    
    img_h, img_w = img_rgba.shape[:2]
    patch_count = 0
    
    for y in range(0, img_h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, img_w - PATCH_SIZE + 1, STRIDE):
            img_patch = img_rgba[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_patch = mask_binary[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            alpha_channel = img_patch[:, :, 3]
            
            if np.max(alpha_channel) > 0 or np.max(mask_patch) > 0: 
                cv2.imwrite(f"{out_img_dir}/{name}_y{y}_x{x}.png", img_patch)
                cv2.imwrite(f"{out_mask_dir}/{name}_y{y}_x{x}.png", mask_patch)
                patch_count += 1
                
    return patch_count

if __name__ == "__main__":
    setup_directories()
    
    img_files = sorted(glob.glob(f"{BASE_DIR}/raw/images/*.png"))
    total_files = len(img_files)
    
    if total_files == 0:
        print(f"LỖI: Không tìm thấy ảnh nào trong thư mục '{BASE_DIR}/raw/images/'")
        exit()
        
    # --- CHIA LẠI DATASET: TRAIN (80%) - VAL (20%) ---
    train_split = int(total_files * 0.8)
    
    if train_split == total_files:
        train_split = max(1, total_files - 1)

    total_train, total_val = 0, 0

    for idx, img_path in enumerate(tqdm(img_files, desc="Đang băm ảnh")):
        mask_path = img_path.replace('images', 'masks')
        
        if not os.path.exists(mask_path):
            print(f"\nBỏ qua {img_path} vì không tìm thấy file Mask tương ứng.")
            continue
            
        if idx < train_split:
            total_train += slice_image_and_mask(img_path, mask_path, f"{BASE_DIR}/train/images", f"{BASE_DIR}/train/masks")
        else:
            total_val += slice_image_and_mask(img_path, mask_path, f"{BASE_DIR}/val/images", f"{BASE_DIR}/val/masks")

    print("\nHOÀN THÀNH!")
    print(f"Patch sinh ra -> Train: {total_train} | Val: {total_val}")
