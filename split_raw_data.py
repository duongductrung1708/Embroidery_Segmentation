import os
import glob
import shutil
import random

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
BASE_DIR = "data"
# Cố định random seed để mỗi lần chạy lại, ảnh được chia y hệt nhau
random.seed(42) 

def setup_dirs():
    """Xóa trắng thư mục cũ và tạo cấu trúc thư mục mới"""
    print("Đang dọn dẹp thư mục cũ...")
    dirs = [f"{BASE_DIR}/train/images", f"{BASE_DIR}/train/masks",
            f"{BASE_DIR}/val/images", f"{BASE_DIR}/val/masks"]
    for d in dirs:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    setup_dirs()
    
    # Tìm toàn bộ ảnh trong thư mục raw
    img_files = sorted(glob.glob(f"{BASE_DIR}/raw/images/*.png"))
    total_files = len(img_files)
    
    if total_files == 0:
        print(f"LỖI: Không tìm thấy ảnh nào trong thư mục '{BASE_DIR}/raw/images/'")
        exit()
        
    print(f"Phát hiện tổng cộng {total_files} ảnh gốc.")
    
    # Xáo trộn danh sách ảnh để đảm bảo tính ngẫu nhiên
    random.shuffle(img_files)
    
    # Tính mốc cắt 80% (Train) - 20% (Val)
    train_split = int(total_files * 0.8)
    if train_split == total_files: train_split = max(1, total_files - 1)

    train_count = 0
    val_count = 0

    print("\nĐang bắt đầu sao chép (KHÔNG BĂM) sang các thư mục...")
    for idx, img_path in enumerate(img_files):
        mask_path = img_path.replace('images', 'masks')
        
        if not os.path.exists(mask_path):
            print(f" Cảnh báo: Bỏ qua {os.path.basename(img_path)} vì thiếu file Mask!")
            continue
            
        filename = os.path.basename(img_path)
        
        # 80% đầu tiên vào Train
        if idx < train_split:
            shutil.copy(img_path, f"{BASE_DIR}/train/images/{filename}")
            shutil.copy(mask_path, f"{BASE_DIR}/train/masks/{filename}")
            train_count += 1
        # 20% còn lại vào Val
        else:
            shutil.copy(img_path, f"{BASE_DIR}/val/images/{filename}")
            shutil.copy(mask_path, f"{BASE_DIR}/val/masks/{filename}")
            val_count += 1

    print("\n==============================================")
    print("HOÀN THÀNH CHIA TẬP DỮ LIỆU GỐC!")
    print(f" -> Tập Train: {train_count} ảnh")
    print(f" -> Tập Val  : {val_count} ảnh")
    print("Bạn đã có thể chạy 'python train.py' ngay bây giờ.")
    print("==============================================")