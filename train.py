import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # Thêm thư viện vẽ thanh tiến trình
from src.dataset import EmbroideryDataset
from src.model import UNet

# 1. Cấu hình thiết bị (GPU/CPU/MPS cho máy Mac)
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Đang sử dụng thiết bị tính toán: {device}")

# 2. Khởi tạo Băng chuyền dữ liệu
# transform.ToTensor() sẽ tự động đổi ảnh từ [0-255] thành dải [0.0 - 1.0] (Rất tốt cho AI)
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = EmbroideryDataset(image_dir="data/images", mask_dir="data/masks", transform=transform)
# Dùng batch_size nhỏ (ví dụ: 2) vì U-Net ngốn khá nhiều VRAM
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 3. Khởi tạo Mô hình, Hàm Lỗi và Bác sĩ tối ưu
model = UNet(in_channels=3, out_channels=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4) # lr: Learning Rate (Bước chân học tập)

# 4. BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN
EPOCHS = 20 # Số lần học sinh ôn lại toàn bộ sách giáo khoa
print("Bắt đầu quá trình huấn luyện...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # Bọc train_loader bằng tqdm để sinh ra thanh tiến trình
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for images, masks in loop:
        # Ném dữ liệu lên GPU (hoặc CPU)
        images = images.to(device)
        masks = masks.to(device)

        # BƯỚC 1: Xóa sạch bộ nhớ của lần học trước
        optimizer.zero_grad()

        # BƯỚC 2: Học sinh làm bài (AI dự đoán)
        outputs = model(images)

        # BƯỚC 3: Thầy giáo chấm điểm (So sánh outputs và masks)
        loss = criterion(outputs, masks)

        # BƯỚC 4: Tìm nguyên nhân lỗi sai (Lan truyền ngược)
        loss.backward()

        # BƯỚC 5: Học sinh tự sửa sai trong não (Cập nhật trọng số)
        optimizer.step()

        running_loss += loss.item()

        # Cập nhật liên tục chỉ số Loss lên thanh tiến trình
        loop.set_postfix(loss=loss.item())

    # In ra báo cáo sau mỗi Epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"Tổng kết Epoch {epoch+1} - Loss trung bình: {epoch_loss:.4f}\n")

print("Hoàn thành huấn luyện!")

# 5. Lưu lại "bộ não" của AI sau khi đã học xong
torch.save(model.state_dict(), "unet_embroidery.pth")
print("Đã lưu trọng số mô hình tại file: unet_embroidery.pth")
