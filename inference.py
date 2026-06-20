import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Import model của bạn (nhớ cấu trúc thư mục src)
from src.model import UNet 

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
PATCH_SIZE = 512
MODEL_PATH = "unet_binary_best.pth" 

# --- 2 CON ỐC ĐÃ ĐƯỢC VẶN LẠI CHO MODEL V3 ---
RESIZE_FACTOR = 0.5            
CONFIDENCE_THRESHOLD = 0.50    # Sửa thành 0.5 vì model V3 (Dice Loss) đã quá chuẩn, không cần siết gắt nữa

# Khởi tạo và nạp tệp trọng số (bộ não AI)
model = UNet(in_channels=1, out_channels=2).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("Đã nạp thành công bộ não AI V3!")
else:
    print("Không tìm thấy file model. Bạn đã train xong chưa?")
    exit()

def predict_full_image(img_path, save_output=True):
    print(f"\nĐang phân tích ảnh: {img_path}")
    
    # ==========================================
    # 2. ĐỌC ẢNH VÀ ĐỒNG BỘ SCALE (RESIZE)
    # ==========================================
    rgba_img = Image.open(img_path).convert("RGBA")
    orig_w, orig_h = rgba_img.size
    alpha_channel_orig = np.array(rgba_img.getchannel("A"), dtype=np.float32)

    # Thu nhỏ ảnh lại giống hệt lúc train
    new_w = int(orig_w * RESIZE_FACTOR)
    new_h = int(orig_h * RESIZE_FACTOR)
    alpha_channel_resized = cv2.resize(alpha_channel_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ==========================================
    # 3. KỸ THUẬT PADDING 
    # ==========================================
    pad_h = (PATCH_SIZE - new_h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - new_w % PATCH_SIZE) % PATCH_SIZE
    
    padded_alpha = np.pad(alpha_channel_resized, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    padded_pred = np.zeros_like(padded_alpha, dtype=np.uint8)

    # ==========================================
    # 4. QUÉT SLIDING WINDOW VÀ SIẾT NGƯỠNG
    # ==========================================
    print(f"AI đang nhận diện (Ngưỡng tự tin > {int(CONFIDENCE_THRESHOLD * 100)}%)...")
    with torch.no_grad():
        for y in range(0, padded_alpha.shape[0] - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, padded_alpha.shape[1] - PATCH_SIZE + 1, PATCH_SIZE):
                patch = padded_alpha[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                # Chỗ nào trong suốt toàn tập thì bỏ qua cho nhanh
                if np.max(patch) == 0: continue
                
                # Ép kiểu cho PyTorch
                input_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0) / 255.0
                input_tensor = input_tensor.to(DEVICE)
                
                # AI dự đoán
                output = model(input_tensor)
                
                probs = torch.softmax(output, dim=1) 
                fill_probs = probs[0, 1, :, :] 
                
                # Tô trắng dựa trên ngưỡng 0.5
                pred_patch = (fill_probs > CONFIDENCE_THRESHOLD).cpu().numpy().astype(np.uint8)
                
                # Dán kết quả vào canvas
                padded_pred[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = pred_patch

    # Cắt bỏ phần viền thừa đã đệm lúc nãy
    pred_resized_mask = padded_pred[:new_h, :new_w]

    # Phóng to Mask trả lại kích thước gốc (Dùng INTER_NEAREST để viền cứng)
    raw_final_mask = cv2.resize(pred_resized_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # ==========================================
    # 5. THỢ MỘC OPENCV (Chỉ vá lỗ, KHÔNG gọt viền nữa)
    # ==========================================
    print("OpenCV đang vá các lỗ thủng li ti (nếu có)...")
    mask_255 = (raw_final_mask * 255).astype(np.uint8)
    
    kernel = np.ones((3, 3), np.uint8) 
    
    # Đóng lỗ (Closing): Lấp các khe nứt li ti bên trong
    cleaned_mask = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # --- ĐÃ CẤT GIẤY NHÁM ĐI ĐỂ TRÁNH MASK BỊ TEO TÓP ---
    # cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=1)
    
    final_cleaned_mask = (cleaned_mask > 127).astype(np.uint8)

    if save_output:
        out_name = "AI_Mask_Cleaned.png"
        cv2.imwrite(out_name, cleaned_mask)
        print(f"Đã lưu Mask hoàn hảo ra file: {out_name}")

    # ==========================================
    # 6. HIỂN THỊ KẾT QUẢ
    # ==========================================
    plt.figure(figsize=(14, 7)) 
    
    display_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    display_img[alpha_channel_orig > 0] = [255, 255, 255]
    
    plt.subplot(1, 2, 1)
    plt.title("Ảnh gốc (Nét vẽ)")
    plt.imshow(display_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mask từ AI V4 PRO)")
    overlay_clean = display_img.copy()
    overlay_clean[final_cleaned_mask == 1] = [0, 255, 0] # Màu Xanh Lá
    plt.imshow(overlay_clean)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# CHẠY THỬ VỚI ẢNH CỦA BẠN
predict_full_image("./data/test/test1.png")
