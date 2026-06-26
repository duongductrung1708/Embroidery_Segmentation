import os
import shutil
import random
from glob import glob

# ==========================================
# CẤU HÌNH
# ==========================================
RAW_DIR = "data/raw"
TRAIN_DIR = "data/lineart/train"
VAL_DIR = "data/lineart/val"

CATEGORIES = ["easy", "medium", "hard"]
SPLIT_RATIO = 0.7  # 70% Train, 30% Val
SEED = 42

def reset_directories():
    """Xóa thư mục train/val cũ và tạo lại mới tinh"""
    for folder in [TRAIN_DIR, VAL_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(os.path.join(folder, "images"))
        os.makedirs(os.path.join(folder, "masks"))

def stratified_split():
    random.seed(SEED)
    reset_directories()
    
    total_train = 0
    total_val = 0

    print("BAT DAU CHIA DU LIEU PHAN TANG (STRATIFIED SPLIT)...\n")

    for category in CATEGORIES:
        cat_img_dir = os.path.join(RAW_DIR, category, "images")
        cat_mask_dir = os.path.join(RAW_DIR, category, "masks")
        
        if not os.path.exists(cat_img_dir):
            print(f"[-] Bo qua '{category}' vi khong tim thay thu muc.")
            continue
            
        # Quét toàn bộ ảnh (hỗ trợ cả png và jpg)
        images = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            images.extend(glob(os.path.join(cat_img_dir, ext)))
            
        images.sort()
        random.shuffle(images) # Xáo trộn ngẫu nhiên
        
        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        # Hàm copy file sang đích đến
        def copy_files(file_list, dest_folder):
            count = 0
            for img_path in file_list:
                filename = os.path.basename(img_path)
                # Giả định file mask cùng tên, đổi đuôi thành .png (tùy format của bạn)
                mask_filename = filename.rsplit('.', 1)[0] + '.png' 
                mask_path = os.path.join(cat_mask_dir, mask_filename)
                
                if os.path.exists(mask_path):
                    shutil.copy(img_path, os.path.join(dest_folder, "images", filename))
                    shutil.copy(mask_path, os.path.join(dest_folder, "masks", mask_filename))
                    count += 1
                else:
                    print(f"    [CANH BAO] Khong tim thay Mask cho {filename}")
            return count

        # Bắt đầu copy
        c_train = copy_files(train_imgs, TRAIN_DIR)
        c_val = copy_files(val_imgs, VAL_DIR)
        
        total_train += c_train
        total_val += c_val
        
        print(f"[+] Nhom {category.upper()}: Tong {len(images)} anh -> Train: {c_train} | Val: {c_val}")

    print("\n==========================================")
    print(f"HOAN THANH! TONG CONG:")
    print(f"  -> Tap TRAIN: {total_train} anh (Hoc tap phan tang)")
    print(f"  -> Tap VAL:   {total_val} anh (De thi cong bang)")
    print("==========================================")

if __name__ == "__main__":
    stratified_split()