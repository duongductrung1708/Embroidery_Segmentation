import cv2
import numpy as np

# Đọc ảnh gốc (tải 1 ảnh nền trắng từ Google về)
img_gray = cv2.imread("./9.png", cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print("Không tìm thấy ảnh gốc!")
else:
    TARGET_W = 4200
    TARGET_H = 2340

    # 1. Tính tỷ lệ phóng to sao cho vừa khít khung mà không bị méo
    scale = min(TARGET_W / img_gray.shape[1], TARGET_H / img_gray.shape[0])
    new_w = int(img_gray.shape[1] * scale)
    new_h = int(img_gray.shape[0] * scale)

    # Phóng to ảnh
    img_resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 2. Ép nét thành đen tuyền, xóa nền trắng
    _, binary = cv2.threshold(img_resized, 200, 255, cv2.THRESH_BINARY)
    alpha_channel = cv2.bitwise_not(binary)
    
    # Tạo ảnh con hổ trong suốt (tạm thời)
    rgba_tiger = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    rgba_tiger[:, :, 3] = alpha_channel 
    
    # 3. KỸ THUẬT CANVAS: Tạo tấm bạt trong suốt chuẩn 4200x2340
    canvas = np.zeros((TARGET_H, TARGET_W, 4), dtype=np.uint8)
    
    # 4. Dán con hổ vào chính giữa tấm bạt
    y_offset = (TARGET_H - new_h) // 2
    x_offset = (TARGET_W - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = rgba_tiger
    
    # Lưu kết quả
    cv2.imwrite("9_transparent_4200x2340.png", canvas)
    print("Đã tạo ảnh ĐÚNG CHUẨN 4200x2340 và căn giữa thành công!")