import os
import glob
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH THƯ MỤC
# ==========================================
SRC_DIR = "data/svg/logo"
BASE_DST_DIR = "data/logo"

DIRS = {
    "Easy": os.path.join(BASE_DST_DIR, "easy"),       # Score >= 80
    "Medium": os.path.join(BASE_DST_DIR, "medium"),   # 60 <= Score < 80
    "Hard": os.path.join(BASE_DST_DIR, "hard")        # Score < 60
}

# [TÍNH NĂNG MỚI]: Xóa sạch thư mục cũ trước khi phân loại lại
def reset_directories():
    print(f"[*] Đang làm sạch thư mục cũ: {BASE_DST_DIR}...")
    for d in DIRS.values():
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

# Gọi hàm reset
reset_directories()

# ==========================================
# 2. HÀM TRÍCH XUẤT ĐẶC TRƯNG TỪ SVG
# ==========================================
def extract_svg_metrics(svg_path):
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception:
        return None

    metrics = {
        "total_paths": 0,
        "fill_count": 0,
        "satin_count": 0,
        "total_path_length": 0,
        "small_paths": 0 
    }

    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            metrics["total_paths"] += 1
            
            label = child.get("{http://www.inkscape.org/namespaces/inkscape}label", "").strip().lower()
            if label == "satin":
                metrics["satin_count"] += 1
            else:
                metrics["fill_count"] += 1

            d_attr = child.get("d", "")
            d_len = len(d_attr)
            metrics["total_path_length"] += d_len
            
            # Định nghĩa: Nếu chuỗi 'd' < 50 ký tự -> Đây là nét thêu cực ngắn/vụn
            if d_len < 50:
                metrics["small_paths"] += 1

    if metrics["total_paths"] > 0:
        metrics["avg_path_length"] = metrics["total_path_length"] / metrics["total_paths"]
    else:
        metrics["avg_path_length"] = 0

    return metrics

# ==========================================
# 3. THUẬT TOÁN CHẤM ĐIỂM
# ==========================================
def calculate_difficulty_score(metrics):
    if metrics["total_paths"] == 0:
        return 0
        
    score = 100.0

    # Phạt theo số lượng path
    score -= (metrics["total_paths"] * 0.1)
    # Phạt độ loằng ngoằng
    score -= (metrics["avg_path_length"] * 0.002)
    # Phạt path vụn
    score -= (metrics["small_paths"] * 0.2)

    # Phạt Mất cân bằng Class
    total_labeled = metrics["fill_count"] + metrics["satin_count"]
    if total_labeled > 0:
        min_class = min(metrics["fill_count"], metrics["satin_count"])
        imbalance_ratio = min_class / total_labeled
        if imbalance_ratio < 0.10:
            score -= 10.0
        if min_class == 0:
            score -= 20.0

    return max(0.0, min(100.0, score))


# ==========================================
# 4. THỰC THI PIPELINE
# ==========================================
def main():
    svg_files = glob.glob(os.path.join(SRC_DIR, "*.svg"))
    if not svg_files:
        print(f"Không tìm thấy file SVG nào trong {SRC_DIR}")
        return

    print(f"[*] Bắt đầu chấm điểm và phân loại {len(svg_files)} SVG...")
    
    stats = {"Easy": 0, "Medium": 0, "Hard": 0}
    
    for svg_path in tqdm(svg_files, desc="Đang xử lý"):
        metrics = extract_svg_metrics(svg_path)
        if metrics is None or metrics["total_paths"] == 0:
            continue
            
        score = calculate_difficulty_score(metrics)
        
        # Phân loại theo ngưỡng
        if score >= 80:
            category = "Easy"
        elif score >= 60:
            category = "Medium"
        else:
            category = "Hard"
            
        stats[category] += 1
        
        # Copy file sang thư mục tương ứng
        filename = os.path.basename(svg_path)
        dst_path = os.path.join(DIRS[category], filename)
        shutil.copy2(svg_path, dst_path)

    print("\n" + "="*50)
    print(" BÁO CÁO CURRICULUM DATASET")
    print("="*50)
    print(f" [Easy]   (Điểm 80-100) : {stats['Easy']} ảnh")
    print(f" [Medium] (Điểm 60-79)  : {stats['Medium']} ảnh")
    print(f" [Hard]   (Điểm 0-59)   : {stats['Hard']} ảnh")
    print("="*50)
    print(f"-> Dữ liệu đã được cập nhật sạch sẽ tại: {BASE_DST_DIR}")

if __name__ == "__main__":
    main()
