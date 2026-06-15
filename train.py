import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb

from src.dataset import EmbroideryDataset
from src.model import UNet

# Hàm tính toán Metrics toán học
def calculate_metrics(tp, fp, fn, tn):
    epsilon = 1e-7 # Chống chia cho 0
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return accuracy, precision, recall, f1

def main():
    wandb.init(
        project="embroidery-segmentation", 
        name="binary-fill-metrics",         
        config={                           
            "learning_rate": 1e-4,
            "architecture": "U-Net",
            "dataset": "Embroidery_DST_Binary",
            "epochs": 20,
            "batch_size": 4,
            "image_size": 512
        }
    )
    config = wandb.config 

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Đang sử dụng thiết bị tính toán: {device}")

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = EmbroideryDataset(image_dir="data/train/images", mask_dir="data/train/masks", transform=transform)
    val_dataset = EmbroideryDataset(image_dir="data/val/images", mask_dir="data/val/masks", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, persistent_workers=True) 

    model = UNet(in_channels=3, out_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_f1 = 0.0 # Đổi kỷ lục từ Loss sang F1 (F1 càng cao càng tốt)

    print("\nBắt đầu quá trình huấn luyện...")
    for epoch in range(config.epochs):
        
        # =========================
        # PHA 1: HỌC TẬP (TRAIN)
        # =========================
        model.train()
        running_train_loss = 0.0
        train_tp = train_fp = train_fn = train_tn = 0 # Bộ đếm Pixel cho Train

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
            
            # Đếm Pixel (Dự đoán vs Thực tế) trên GPU
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                train_tp += ((preds == 1) & (masks == 1)).sum().item() # Đoán là Fill, thực tế là Fill
                train_fp += ((preds == 1) & (masks == 0)).sum().item() # Đoán là Fill, thực tế là Nền (Đoán lố)
                train_fn += ((preds == 0) & (masks == 1)).sum().item() # Đoán là Nền, thực tế là Fill (Bỏ sót)
                train_tn += ((preds == 0) & (masks == 0)).sum().item() # Đoán là Nền, thực tế là Nền

        avg_train_loss = running_train_loss / len(train_loader)
        train_acc, train_prec, train_recall, train_f1 = calculate_metrics(train_tp, train_fp, train_fn, train_tn)

        # =========================
        # PHA 2: THI THỬ (VALIDATION)
        # =========================
        model.eval()
        running_val_loss = 0.0
        val_tp = val_fp = val_fn = val_tn = 0 # Bộ đếm Pixel cho Val
        
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_masks)
                running_val_loss += val_loss.item()

                preds = torch.argmax(val_outputs, dim=1)
                val_tp += ((preds == 1) & (val_masks == 1)).sum().item()
                val_fp += ((preds == 1) & (val_masks == 0)).sum().item()
                val_fn += ((preds == 0) & (val_masks == 1)).sum().item()
                val_tn += ((preds == 0) & (val_masks == 0)).sum().item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_acc, val_prec, val_recall, val_f1 = calculate_metrics(val_tp, val_fp, val_fn, val_tn)

        # =========================
        # BÁO CÁO VÀ GHI LOG WANDB
        # =========================
        print(f"\n[Epoch {epoch+1}] Báo cáo:")
        print(f"   Train | Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(f"   Val   | Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}\n")

        wandb.log({
            "epoch": epoch + 1,
            "Loss/Train": avg_train_loss,
            "Loss/Val": avg_val_loss,
            "Accuracy/Train": train_acc,
            "Accuracy/Val": val_acc,
            "Recall/Train": train_recall,
            "Recall/Val": val_recall,
            "F1_Score/Train": train_f1,
            "F1_Score/Val": val_f1
        })

        # CHÚ Ý: Đánh giá mô hình tốt nhất dựa trên F1-Score của tập Val thay vì Loss
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "unet_binary_best.pth")
            print(f"Đã lưu kỷ lục mới (Best Val F1: {best_val_f1:.4f})")
            wandb.save("unet_binary_best.pth")

    print("\nHoàn thành huấn luyện!")
    wandb.finish() 

if __name__ == "__main__":
    main()