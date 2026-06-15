import os
import glob
import math
import pyembroidery
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from src.model import UNet

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
PATCH_SIZE = 512
CONSTANT_SCALE = 3.0

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Chạy hệ thống trên thiết bị: {device}")

# ==========================================
# 2. KHỞI TẠO BỘ NÃO AI
# ==========================================
model = UNet(in_channels=3, out_channels=2).to(device)
# Nạp trọng số tốt nhất bạn vừa train xong
model.load_state_dict(torch.load("unet_binary_best.pth", map_location=device, weights_only=True))
model.eval() 
transform = transforms.Compose([transforms.ToTensor()])

# ==========================================
# 3. CHỌN 1 FILE THIẾT KẾ ĐỂ TEST
# ==========================================
# Lấy ngẫu nhiên 1 file .dst trong thư mục raw (hoặc bạn có thể tự trỏ đường dẫn)
raw_files = glob.glob("data/raw/On4MSQyRQ-1-1-1_(4).DST", recursive=True) + glob.glob("data/raw/On4MSQyRQ-1-1-1_(4).DST", recursive=True)
test_file = raw_files[0] 
print(f"Đang phân tích file gốc: {test_file}")

# ==========================================
# 4. RENDER ẢNH KHỔNG LỒ TỪ TỌA ĐỘ .DST
# ==========================================
pattern = pyembroidery.read(test_file)
coords = [s for s in pattern.stitches if s[2] == 0] 

xs, ys = [c[0] for c in coords], [c[1] for c in coords]
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)

width = int((max_x - min_x) * CONSTANT_SCALE)
height = int((max_y - min_y) * CONSTANT_SCALE)
pad = PATCH_SIZE
img_h, img_w = height + pad * 2, width + pad * 2

print(f"Kích thước bức tranh khổng lồ: {img_w} x {img_h} pixel")
large_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

for i in range(len(coords) - 1):
    x1 = int((coords[i][0] - min_x) * CONSTANT_SCALE) + pad
    y1 = int((coords[i][1] - min_y) * CONSTANT_SCALE) + pad
    x2 = int((coords[i+1][0] - min_x) * CONSTANT_SCALE) + pad
    y2 = int((coords[i+1][1] - min_y) * CONSTANT_SCALE) + pad
    cv2.line(large_img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)

# ==========================================
# 5. KỸ THUẬT PADDING & STITCHING
# ==========================================
# Làm tròn kích thước ảnh để chia hết cho 512
pad_h = (PATCH_SIZE - img_h % PATCH_SIZE) % PATCH_SIZE
pad_w = (PATCH_SIZE - img_w % PATCH_SIZE) % PATCH_SIZE
large_img_padded = cv2.copyMakeBorder(large_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

padded_h, padded_w = large_img_padded.shape[:2]

# Tạo một "Bức tranh nháp" khổng lồ đen thui để dán kết quả dự đoán vào
large_pred_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)

print("AI đang quét (Sliding Window) toàn bộ bề mặt thiết kế...")
with torch.no_grad():
    for y in tqdm(range(0, padded_h, PATCH_SIZE), desc="Quét chiều dọc"):
        for x in range(0, padded_w, PATCH_SIZE):
            # 5.1. Cắt 1 mảnh 512x512
            patch = large_img_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # 5.2. Nếu mảnh đó đen thui (không có nét thêu), bỏ qua luôn cho lẹ
            if np.max(patch) == 0:
                continue
                
            # 5.3. Bơm vào AI
            input_tensor = transform(patch).unsqueeze(0).to(device)
            output = model(input_tensor)
            pred_patch = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # 5.4. Lấy cái mask 512x512 AI vừa nhả ra, dán đè lại đúng tọa độ đó
            large_pred_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = pred_patch

# Gọt bỏ đi cái phần viền (padding) đã thêm vào lúc nãy
final_mask = large_pred_mask[:img_h, :img_w]

# ==========================================
# 6. TÔ MÀU BÁO CÁO VÀ HIỂN THỊ
# ==========================================
print("Đang đổ màu báo cáo...")
color_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
color_mask[final_mask == 0] = [255, 255, 255]       # Nền -> Trắng
color_mask[final_mask == 1] = [0, 0, 0]     # Vùng Fill -> Đen

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Ảnh gốc (Bề mặt thiết kế)")
plt.imshow(cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("AI Phân vùng (Đen: Vùng Fill)")
plt.imshow(color_mask)
plt.axis('off')

plt.tight_layout()
# Lưu luôn kết quả thành file ảnh sắc nét để Kỹ sư đem đi báo cáo
plt.savefig("result_full_inference.png", dpi=300) 
print("Hoàn thành! Đã lưu kết quả tại file: result_full_inference.png")
plt.show()