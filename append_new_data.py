import os
import glob
import shutil
import random

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
NEW_DATA_DIR = "data/raw_new"  # Thư mục chứa ảnh mới tinh
TARGET_DIR = "data"            # Thư mục đích (chứa train/val hiện tại)
RAW_DIR = "data/raw"           # Kho chứa toàn bộ ảnh gốc của dự án

# Vẫn dùng seed để nếu có lỗi chạy lại thì nó chia giống hệt lúc nãy
random.seed(99) 

def append_data():
    print(f"Đang quét thư mục ảnh mới: {NEW_DATA_DIR}/images/...")
    
    # 1. Tìm toàn bộ ảnh trong thư mục raw_new
    new_img_files = sorted(glob.glob(f"{NEW_DATA_DIR}/images/*.png"))
    total_new = len(new_img_files)
    
    if total_new == 0:
        print(f"LỖI: Không tìm thấy ảnh mới nào trong '{NEW_DATA_DIR}/images/'. Bạn đã chép nhầm chỗ?")
        return
        
    print(f"Phát hiện {total_new} ảnh mới. Đang tiến hành phân loại và ghi nối (Append)...")
    
    # 2. Xáo trộn ngẫu nhiên danh sách ảnh MỚI
    random.shuffle(new_img_files)
    
    # 3. Tính mốc cắt 80% (Train) - 20% (Val) cho lượng ảnh mới này
    train_split = int(total_new * 0.8)
    if train_split == total_new: train_split = max(1, total_new - 1)

    train_added = 0
    val_added = 0

    # 4. Bắt đầu Copy nối thêm vào nhà cũ
    for idx, img_path in enumerate(new_img_files):
        mask_path = img_path.replace('images', 'masks')
        
        if not os.path.exists(mask_path):
            print(f"Cảnh báo: Bỏ qua {os.path.basename(img_path)} vì thiếu file Mask!")
            continue
            
        filename = os.path.basename(img_path)
        
        # 80% số ảnh mới sẽ được đẩy vào Train
        if idx < train_split:
            shutil.copy(img_path, f"{TARGET_DIR}/train/images/{filename}")
            shutil.copy(mask_path, f"{TARGET_DIR}/train/masks/{filename}")
            train_added += 1
            
        # 20% số ảnh mới sẽ được đẩy vào Val
        else:
            shutil.copy(img_path, f"{TARGET_DIR}/val/images/{filename}")
            shutil.copy(mask_path, f"{TARGET_DIR}/val/masks/{filename}")
            val_added += 1

    # ==========================================
    # 5. TỰ ĐỘNG DỌN DẸP: CẤT ẢNH MỚI VÀO KHO RAW CHUNG
    # ==========================================
    os.makedirs(f"{RAW_DIR}/images", exist_ok=True)
    os.makedirs(f"{RAW_DIR}/masks", exist_ok=True)

    print(f"\nĐang dọn dẹp trạm trung chuyển '{NEW_DATA_DIR}' và cất ảnh vào kho '{RAW_DIR}'...")
    for img_path in new_img_files:
        mask_path = img_path.replace('images', 'masks')
        filename = os.path.basename(img_path)
        
        # Chuyển thẳng từ raw_new sang raw
        shutil.move(img_path, f"{RAW_DIR}/images/{filename}")
        if os.path.exists(mask_path):
            shutil.move(mask_path, f"{RAW_DIR}/masks/{filename}")

    # ==========================================
    # BÁO CÁO KẾT QUẢ
    # ==========================================
    print("\n==============================================")
    print("HOÀN THÀNH GHI NỐI DỮ LIỆU MỚI!")
    print(f" -> Đã BƠM THÊM vào tập Train: +{train_added} ảnh")
    print(f" -> Đã BƠM THÊM vào tập Val  : +{val_added} ảnh")
    print(f" -> Toàn bộ ảnh mới đã được cất gọn gàng vào '{RAW_DIR}' cùng với các ảnh cũ.")
    print("Tập Validation cũ của bạn vẫn được an toàn 100%.")
    print("\nBạn có thể chạy 'python train.py' để Train tiếp (Fine-tune) ngay bây giờ.")
    print("==============================================")

if __name__ == "__main__":
    # Kiểm tra xem các thư mục đích đã tồn tại chưa
    if not os.path.exists(f"{TARGET_DIR}/train/images"):
        print("LỖI: Không tìm thấy thư mục Train/Val cũ. Bạn đã chạy split_raw_data.py lần nào chưa?")
    else:
        append_data()
