import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.model import UNet

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Chạy dự đoán trên: {device}")

# NHỚ ĐỔI out_channels=2 ĐỂ KHỚP VỚI LÚC TRAIN
model = UNet(in_channels=3, out_channels=2).to(device)
model.load_state_dict(torch.load("unet_binary_best.pth", map_location=device, weights_only=True))
model.eval() 

transform = transforms.Compose([transforms.ToTensor()])

# Lấy 1 bức ảnh từ tập test
test_image_name = os.listdir("data/test/images")[0] 
img_path = os.path.join("data/test/images", test_image_name)

image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device) 

print("AI đang phân tích tìm mảng thêu...")
with torch.no_grad(): 
    output = model(input_tensor) 
    predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# TÔ MÀU KẾT QUẢ (Binary)
color_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
color_mask[predicted_mask == 0] = [0, 0, 0]       # Nền: Đen
color_mask[predicted_mask == 1] = [0, 255, 0]     # Vùng Fill: Xanh lá cây

# Hiển thị
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Ảnh gốc (Bề mặt thêu)")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("AI Phân vùng (Fill: Xanh)")
plt.imshow(color_mask)
plt.axis('off')

plt.tight_layout()
plt.show()