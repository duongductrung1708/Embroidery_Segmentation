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
MODEL_PATH = "unet_binary_best.pth" # File tạ sẽ sinh ra sau khi train xong 20 epochs

# Khởi tạo và nạp tệp trọng số (bộ não AI)
model = UNet(in_channels=1, out_channels=2).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("Đã nạp thành công bộ não AI!")
else:
    print("Không tìm thấy file model. Bạn đã train xong chưa?")
    exit()

def predict_full_image(img_path):
    print(f"Đang phân tích ảnh: {img_path}")
    
    # 2. ĐỌC ẢNH VÀ LẤY KÊNH ALPHA
    rgba_img = Image.open(img_path).convert("RGBA")
    img_w, img_h = rgba_img.size
    alpha_channel = np.array(rgba_img.getchannel("A"), dtype=np.float32)

    # 3. KỸ THUẬT PADDING (Đệm viền để ảnh chia hết cho 512)
    pad_h = (PATCH_SIZE - img_h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - img_w % PATCH_SIZE) % PATCH_SIZE
    
    padded_alpha = np.pad(alpha_channel, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    padded_pred = np.zeros_like(padded_alpha, dtype=np.uint8)

    # 4. QUÉT SLIDING WINDOW (Băm -> Nhờ AI đoán -> Ghép lại)
    print("AI đang nhận diện...")
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
                pred_patch = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                
                # Dán kết quả vào canvas khổng lồ
                padded_pred[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = pred_patch

    # Cắt bỏ phần viền thừa đã đệm lúc nãy
    final_mask = padded_pred[:img_h, :img_w]

    # 5. HIỂN THỊ KẾT QUẢ ĐỂ KỸ SƯ CHẤM ĐIỂM
    plt.figure(figsize=(14, 7))
    
    # Vẽ ảnh gốc (Nét trắng nền đen)
    display_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    display_img[alpha_channel > 0] = [255, 255, 255]
    
    plt.subplot(1, 2, 1)
    plt.title("Ảnh gốc (Nét vẽ)")
    plt.imshow(display_img)
    plt.axis('off')

    # Vẽ ảnh AI đã tô vùng Fill thành MÀU ĐỎ
    plt.subplot(1, 2, 2)
    plt.title("AI Nhận diện (Vùng Đỏ là Fill)")
    overlay = display_img.copy()
    overlay[final_mask == 1] = [255, 0, 0] # Tô đỏ vùng AI dự đoán là Fill
    plt.imshow(overlay)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ĐƯA 1 BỨC ẢNH VÀO ĐÂY ĐỂ TEST
# Bạn có thể lấy 1 trong 7 ảnh gốc, hoặc 1 ảnh hoàn toàn mới chưa từng tô mask!
predict_full_image("data/raw/images/1.png") # Thay tên file ảnh vào đây