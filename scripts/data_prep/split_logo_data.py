import os
import shutil
import random
from glob import glob

# ==========================================
# CẤU HÌNH
# ==========================================
RAW_DIR = "data/logo/easy"
TRAIN_DIR = "data/logo/train"
VAL_DIR = "data/logo/val"

SPLIT_RATIO = 0.8  # 80% Train, 20% Val
SEED = 42

def reset_directories():
    """Xóa thư mục train/val cũ và tạo lại mới tinh"""
    for folder in [TRAIN_DIR, VAL_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(os.path.join(folder, "images"))
        os.makedirs(os.path.join(folder, "masks"))

def split_logo_data():
    random.seed(SEED)
    reset_directories()
    
    img_dir = os.path.join(RAW_DIR, "images")
    mask_dir = os.path.join(RAW_DIR, "masks")
    
    if not os.path.exists(img_dir):
        print(f"[-] Không tìm thấy thư mục {img_dir}")
        return
    
    # Quét toàn bộ ảnh
    images = sorted(glob(os.path.join(img_dir, "*.png")))
    random.shuffle(images)
    
    split_idx = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    def copy_files(file_list, dest_folder):
        count = 0
        for img_path in file_list:
            filename = os.path.basename(img_path)
            mask_filename = filename  # Giữ nguyên tên file
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if os.path.exists(mask_path):
                shutil.copy(img_path, os.path.join(dest_folder, "images", filename))
                shutil.copy(mask_path, os.path.join(dest_folder, "masks", mask_filename))
                count += 1
            else:
                print(f"    [CANH BAO] Không tìm thấy Mask cho {filename}")
        return count

    c_train = copy_files(train_imgs, TRAIN_DIR)
    c_val = copy_files(val_imgs, VAL_DIR)
    
    print("\n==========================================")
    print(f"HOÀN THÀNH! TỔNG CỘNG:")
    print(f"  -> Tập TRAIN: {c_train} ảnh")
    print(f"  -> Tập VAL:   {c_val} ảnh")
    print("==========================================")

if __name__ == "__main__":
    split_logo_data()
