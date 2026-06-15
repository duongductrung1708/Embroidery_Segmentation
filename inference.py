import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.model import UNet

# 1. Cấu hình thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Chạy dự đoán trên: {device}")

# 2. Khởi tạo và nạp "Bộ não" cho AI
model = UNet(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load("unet_embroidery.pth", map_location=device, weights_only=True))
model.eval() # Bật chế độ làm bài thi (Tắt dropout/batchnorm update)

# 3. Tiền xử lý ảnh đầu vào
transform = transforms.Compose([transforms.ToTensor()])

# Lấy thử 1 bức ảnh bất kỳ trong tập Data để xem AI tô màu
# (Bạn có thể đổi tên file thành ảnh mới mà bạn có)
test_image_name = os.listdir("data/test/images")[0]
img_path = os.path.join("data/test/images", test_image_name)

image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device) # Thêm chiều Batch: [1, 3, 512, 512]

# 4. Yêu cầu AI dự đoán
print("AI đang phân tích nét thêu...")
with torch.no_grad(): # Tắt đạo hàm để tăng tốc
    output = model(input_tensor) 
    
    # Lấy nhãn có xác suất cao nhất tại mỗi pixel (Kích thước: [512, 512])
    predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# 5. Trực quan hóa kết quả (Biến ma trận 0, 1, 2 thành ảnh màu)
# Nền (0) -> Đen, Satin (1) -> Đỏ, Tatami (2) -> Xanh lá
color_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
color_mask[predicted_mask == 0] = [0, 0, 0]       # Nền
color_mask[predicted_mask == 1] = [255, 0, 0]     # Satin
color_mask[predicted_mask == 2] = [0, 255, 0]     # Tatami

# Vẽ 2 ảnh song song để so sánh
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Ảnh gốc (Bề mặt thêu)")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("AI Dự đoán (Satin: Đỏ, Tatami: Xanh)")
plt.imshow(color_mask)
plt.axis('off')

plt.tight_layout()
plt.show()