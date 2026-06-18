import cv2
import numpy as np
import os

# ==========================================
# CẤU HÌNH GỐC
# ==========================================
FILE_NAME = "test.png" # Ảnh gốc vớ vẩn trên mạng (PNG rỗng, JPG nền trắng...)
OUTPUT_NAME = "test_ready.png" # SẢN PHẨM HOÀN HẢO ĐỂ TRAIN V2 (PHẢI .PNG)
TARGET_W = 4200
TARGET_H = 2340

print(f"Đang xử lý file: {FILE_NAME}")
img_raw = cv2.imread(FILE_NAME, cv2.IMREAD_UNCHANGED) # Giữ nguyên Kênh Alpha

if img_raw is None:
    print("Không tìm thấy ảnh gốc!")
    exit()

# ==========================================
# 1. KỸ THUẬT ALPHA-COMPOSITE (Standardize to Black-on-White)
# Bước này xử lý triệt để vụ mắt mũi bị tô đen!
# Chúng ta hợp nhất ảnh vào nền TRẮNG tuyệt đối.
# ==========================================
if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
    print("Nhận diện: Ảnh có kênh Alpha. Đang hợp nhất vào nền trắng...")
    rgb_raw = img_raw[:,:,0:3]
    alpha_raw = img_raw[:,:,3]
    
    # Tạo nền trắng tuyệt đối
    white_bg = np.full_like(rgb_raw, 255, dtype=np.uint8)
    
    # Hợp nhất màu theo kênh Alpha (Hàm nội suy)
    alpha_factor = alpha_raw[:,:,np.newaxis].astype(np.float32) / 255.0
    # standard_rgb -> Nét đen trên nền Trắng sạch
    standard_rgb = (rgb_raw.astype(np.float32) * alpha_factor + white_bg.astype(np.float32) * (1 - alpha_factor)).astype(np.uint8)
else:
    print("Nhận diện: Ảnh không có kênh Alpha (JPG/BMP...). Giả định nền trắng.")
    standard_rgb = img_raw

# ==========================================
# 2. PHÓNG TO VÀ PADDING (Đệm khung chuẩn 4200x2340)
# ==========================================
print(f"Phóng to ảnh {FILE_NAME} lên {TARGET_W}px...")
# Tính tỷ lệ phóng to (Giữ đúng tỷ lệ mặt con vật)
scale = min(TARGET_W / standard_rgb.shape[1], TARGET_H / standard_rgb.shape[0])
new_w = int(standard_rgb.shape[1] * scale)
new_h = int(standard_rgb.shape[0] * scale)

# Phóng to ảnh (Sức mạnh nội suy hình khối)
standard_rgb_resized = cv2.resize(standard_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

# Chuyển sang Grayscale để chuẩn bị threshold
standard_gray_resized = cv2.cvtColor(standard_rgb_resized, cv2.COLOR_RGB2GRAY)

# 3. Threshold: Ép nét thành đen tuyền, xóa nền trắng thành trong suốt
_, binary = cv2.threshold(standard_gray_resized, 200, 255, cv2.THRESH_BINARY)
final_alpha = cv2.bitwise_not(binary) # Nét đen -> Alpha=255 (Đục), Trắng -> Alpha=0 (Trong suốt)

# ==========================================
# 4. RE-ASSEMBLE: RGBA [0, 0, 0, A] (All Black RGB + Mask A)
# Dữ liệu chuẩn cho U-Net của bạn
# ==========================================
# Tạo ảnh mới: RGB toàn màu Đen, Kênh A ghép từ bước 3
rgba_tiger_ready = np.zeros((new_h, new_w, 4), dtype=np.uint8)
rgba_tiger_ready[:, :, 3] = final_alpha # Nét vẽ giữ nguyên, nền trắng sạch sẽ

# KỸ THUẬT CANVAS: Dán ảnh lên tấm bạt khổng lồ khép kín
canvas = np.zeros((TARGET_H, TARGET_W, 4), dtype=np.uint8)
y_offset = (TARGET_H - new_h) // 2
x_offset = (TARGET_W - new_w) // 2
canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = rgba_tiger_ready

# ==========================================
# 5. LƯU SẢN PHẨM (BẮT BUỘC .PNG ĐỂ CÓ TRANSPARENCY)
# ==========================================
cv2.imwrite(OUTPUT_NAME, canvas)
print(f"HOÀN THÀNH TOÀN DIỆN! Mắt mũi trong suốt rỗng. File '{OUTPUT_NAME}' đã sẵn sàng.")
