import os
import glob
import shutil
import random

# ==========================================
# CẤU HÌNH HỆ THỐNG V5 (STRATIFIED)
# ==========================================
NEW_DATA_DIR = "data/raw_new"  # Trạm trung chuyển ảnh mới (raw data)
TARGET_DIR = "data/lineart"            # Thư mục đích (chứa train/val hiện tại)
RAW_DIR = "data/raw"           # Kho chứa toàn bộ ảnh gốc phân tầng

CATEGORIES = ["easy", "medium", "hard"]
random.seed(99) 

def append_stratified_data():
    total_train_added = 0
    total_val_added = 0
    
    print(f"ĐANG QUÉT TRẠM TRUNG CHUYỂN: {NEW_DATA_DIR}...\n")
    
    for category in CATEGORIES:
        cat_new_img_dir = f"{NEW_DATA_DIR}/{category}/images"
        cat_raw_img_dir = f"{RAW_DIR}/{category}/images"
        cat_raw_mask_dir = f"{RAW_DIR}/{category}/masks"
        
        # Nếu thư mục độ khó không tồn tại hoặc không có ảnh, bỏ qua
        if not os.path.exists(cat_new_img_dir):
            continue
            
        new_img_files = sorted(glob.glob(f"{cat_new_img_dir}/*.png")) + sorted(glob.glob(f"{cat_new_img_dir}/*.jpg"))
        total_new = len(new_img_files)
        
        if total_new == 0:
            continue
            
        print(f"[*] Phát hiện {total_new} ảnh mới ở độ khó [{category.upper()}]. Đang xử lý...")
        
        random.shuffle(new_img_files)
        
        # Cắt 80/20 cho nhóm độ khó này
        train_split = int(total_new * 0.8)
        if train_split == total_new: train_split = max(1, total_new - 1)
        
        # Đảm bảo kho gốc có thư mục để chứa ảnh cất đi
        os.makedirs(cat_raw_img_dir, exist_ok=True)
        os.makedirs(cat_raw_mask_dir, exist_ok=True)

        for idx, img_path in enumerate(new_img_files):
            # Tìm file mask tương ứng
            filename = os.path.basename(img_path)
            mask_filename = filename.rsplit('.', 1)[0] + '.png'
            mask_path = os.path.join(f"{NEW_DATA_DIR}/{category}/masks", mask_filename)
            
            if not os.path.exists(mask_path):
                print(f"   -> Cảnh báo: Bỏ qua {filename} vì thiếu file Mask!")
                continue
                
            # 1. Copy vào Train hoặc Val
            if idx < train_split:
                shutil.copy(img_path, f"{TARGET_DIR}/train/images/{filename}")
                shutil.copy(mask_path, f"{TARGET_DIR}/train/masks/{mask_filename}")
                total_train_added += 1
            else:
                shutil.copy(img_path, f"{TARGET_DIR}/val/images/{filename}")
                shutil.copy(mask_path, f"{TARGET_DIR}/val/masks/{mask_filename}")
                total_val_added += 1
                
            # 2. Dọn dẹp: Move thẳng ảnh mới về kho RAW chung để lưu trữ
            shutil.move(img_path, os.path.join(cat_raw_img_dir, filename))
            shutil.move(mask_path, os.path.join(cat_raw_mask_dir, mask_filename))

    # ==========================================
    # BÁO CÁO KẾT QUẢ
    # ==========================================
    if total_train_added == 0 and total_val_added == 0:
        print("\nKhông có ảnh mới nào được nạp. Hãy kiểm tra lại thư mục raw_new.")
    else:
        print("\n==============================================")
        print("HOÀN THÀNH GHI NỐI DỮ LIỆU MỚI (BẢO TOÀN PHÂN TẦNG)!")
        print(f" -> Đã BƠM THÊM vào tập Train: +{total_train_added} ảnh")
        print(f" -> Đã BƠM THÊM vào tập Val  : +{total_val_added} ảnh")
        print(f" -> Toàn bộ ảnh đã được phân loại và cất vào '{RAW_DIR}'.")
        print("Bạn có thể chạy 'python train.py' để Fine-tune ngay bây giờ.")
        print("==============================================")

if __name__ == "__main__":
    if not os.path.exists(f"{TARGET_DIR}/train/images"):
        print("LỖI: Không tìm thấy thư mục Train/Val cũ. Bạn hãy chạy split_raw_data.py trước.")
    else:
        append_stratified_data()