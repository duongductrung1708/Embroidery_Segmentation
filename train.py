import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb # 1. Gọi W&B

from src.dataset import EmbroideryDataset
from src.model import UNet

def main():
    # ==========================================
    # KHỞI TẠO W&B (Mở sổ tay đám mây)
    # ==========================================
    wandb.init(
        project="embroidery-segmentation", # Tên dự án trên Web
        name="unet-baseline-run1",         # Tên lần chạy này
        config={                           # Ghi nhớ các thông số cấu hình
            "learning_rate": 1e-4,
            "architecture": "U-Net",
            "dataset": "Embroidery_DST",
            "epochs": 20,
            "batch_size": 4,
            "image_size": 512,
            "threshold": 30
        }
    )
    config = wandb.config # Rút gọn tên biến

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Đang sử dụng thiết bị tính toán: {device}")

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = EmbroideryDataset(image_dir="data/train/images", mask_dir="data/train/masks", transform=transform)
    val_dataset = EmbroideryDataset(image_dir="data/val/images", mask_dir="data/val/masks", transform=transform)

    # Dùng config.batch_size thay vì fix cứng số 4
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, persistent_workers=True) 

    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"Dữ liệu học (Train): {len(train_dataset)} ảnh")
    print(f"Dữ liệu kiểm tra (Val): {len(val_dataset)} ảnh")

    best_val_loss = float('inf') 

    print("\nBắt đầu quá trình huấn luyện...")
    for epoch in range(config.epochs):
        
        # =========================
        # PHA 1: HỌC TẬP (TRAINING)
        # =========================
        model.train()
        running_train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.epochs}] Train")

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
            # (Tùy chọn) Có thể log loss của từng Batch lên W&B ở đây nếu muốn biểu đồ chi tiết:
            # wandb.log({"batch_train_loss": loss.item()})

        avg_train_loss = running_train_loss / len(train_loader)

        # =========================
        # PHA 2: THI THỬ (VALIDATION)
        # =========================
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_masks)
                running_val_loss += val_loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)

        print(f"Kết quả Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # ==========================================
        # ĐẨY DATA LÊN W&B SAU MỖI EPOCH
        # ==========================================
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "unet_embroidery_best.pth")
            print(f"   => Đã lưu kỷ lục mới (best_val_loss: {best_val_loss:.4f})")
            
            # Đính kèm model tốt nhất vào W&B (Để làm backup trên Cloud)
            wandb.save("unet_embroidery_best.pth")

    print("\nHoàn thành huấn luyện!")
    wandb.finish() # Đóng sổ tay

if __name__ == "__main__":
    main()