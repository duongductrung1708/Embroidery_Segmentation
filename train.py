import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import cv2

# BỘ VŨ KHÍ AUGMENTATION HẠNG NẶNG
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset import EmbroideryDataset
from src.model import UNet

# ==========================================
# 1. BỘ VŨ KHÍ CHỐNG TRÀN VIỀN: DICE LOSS
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Lấy xác suất dự đoán của class 1 (Vùng Fill)
        probs = torch.softmax(inputs, dim=1)[:, 1] 
        targets_float = targets.float()
        
        # Tính độ giao nhau (Intersection) và phần hợp (Union)
        intersection = (probs * targets_float).sum(dim=(1,2))
        union = probs.sum(dim=(1,2)) + targets_float.sum(dim=(1,2))
        
        # Dice Score: 1 là hoàn hảo, 0 là trật lất
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Vì là hàm Loss (cần giảm về 0), ta lấy 1 trừ đi Dice Score
        return 1.0 - dice.mean()

# ==========================================
# HÀM TÍNH TOÁN METRICS
# ==========================================
def calculate_metrics(tp, fp, fn, tn):
    epsilon = 1e-7 
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return accuracy, precision, recall, f1

# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    # Khởi tạo Weights & Biases để theo dõi
    wandb.init(
        project="embroidery-segmentation", 
        name="v4-pro-augmentation-onthefly",         
        config={                           
            "learning_rate": 1e-4,
            "architecture": "U-Net",
            "dataset": "Embroidery_V2",
            "epochs": 50,
            "batch_size": 4,
            "image_size": 512,
            "fill_weight": 2.0,
            "crops_per_image": 20 # 1 ảnh gốc sẽ được cắt 20 lần ngẫu nhiên trong 1 Epoch
        }
    )
    config = wandb.config 

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Đang sử dụng thiết bị tính toán: {device}")

    # ==========================================
    # 2. KHAI BÁO DATA AUGMENTATION (BẢN PRO)
    # ==========================================
    train_transform = A.Compose([
        # Cắt thông minh: Ưu tiên nhắm vào chỗ có nét vẽ
        A.CropNonEmptyMaskIfExists(width=config.image_size, height=config.image_size), 
        
        # Lật ảnh cơ bản
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5),   
        
        # Xoay tự do & Dịch chuyển (Đắp viền đen)
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.15, rotate_limit=180, 
            interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, 
            value=0, mask_value=0, p=0.7
        ),

        # Biến dạng đàn hồi (Elastic) - Uốn cong nét vẽ
        A.ElasticTransform(
            alpha=1, sigma=50, alpha_affine=50, 
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.3
        ),

        # Giả lập đứt nét / Mất chi tiết (Khoét lỗ đen)
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32, 
            min_holes=2, min_height=8, min_width=8,
            fill_value=0, mask_fill_value=0, p=0.3
        ),

        ToTensorV2()             
    ])

    val_transform = A.Compose([
        # Validation cũng cắt vào chỗ có nét vẽ để đánh giá F1 cho chuẩn
        A.CropNonEmptyMaskIfExists(width=config.image_size, height=config.image_size),
        ToTensorV2()
    ])

    # ==========================================
    # 3. NẠP DỮ LIỆU (ON-THE-FLY)
    # ==========================================
    train_dataset = EmbroideryDataset(
        image_dir="data/train/images", 
        mask_dir="data/train/masks", 
        transform=train_transform,
        resize_factor=0.5,
        crops_per_image=config.crops_per_image
    )
    
    val_dataset = EmbroideryDataset(
        image_dir="data/val/images", 
        mask_dir="data/val/masks", 
        transform=val_transform,
        resize_factor=0.5,
        crops_per_image=max(1, config.crops_per_image // 2)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, persistent_workers=True) 

    model = UNet(in_channels=1, out_channels=2).to(device)
    
    # Kết hợp CE Loss và Dice Loss
    class_weights = torch.tensor([1.0, config.fill_weight]).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss_fn = DiceLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_f1 = 0.0

    print("\nBẮT ĐẦU HUẤN LUYỆN V4 PRO (ON-THE-FLY & ADVANCED AUGMENTATION)...")
    for epoch in range(config.epochs):
        
        # ------------------------------------------
        # PHA 1: HỌC TẬP (TRAIN)
        # ------------------------------------------
        model.train()
        running_train_loss = 0.0
        train_tp = train_fp = train_fn = train_tn = 0 

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.epochs}] Train")

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Tính tổng 2 Loss
            loss_ce = ce_loss_fn(outputs, masks)
            loss_dice = dice_loss_fn(outputs, masks)
            loss = loss_ce + loss_dice 
            
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=loss.item(), lr=current_lr)
            
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                train_tp += ((preds == 1) & (masks == 1)).sum().item() 
                train_fp += ((preds == 1) & (masks == 0)).sum().item() 
                train_fn += ((preds == 0) & (masks == 1)).sum().item() 
                train_tn += ((preds == 0) & (masks == 0)).sum().item() 

        avg_train_loss = running_train_loss / len(train_loader)
        train_acc, train_prec, train_recall, train_f1 = calculate_metrics(train_tp, train_fp, train_fn, train_tn)

        # ------------------------------------------
        # PHA 2: THI THỬ (VALIDATION)
        # ------------------------------------------
        model.eval()
        running_val_loss = 0.0
        val_tp = val_fp = val_fn = val_tn = 0 
        
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                
                val_outputs = model(val_images)
                
                v_loss_ce = ce_loss_fn(val_outputs, val_masks)
                v_loss_dice = dice_loss_fn(val_outputs, val_masks)
                val_loss = v_loss_ce + v_loss_dice
                running_val_loss += val_loss.item()

                preds = torch.argmax(val_outputs, dim=1)
                val_tp += ((preds == 1) & (val_masks == 1)).sum().item()
                val_fp += ((preds == 1) & (val_masks == 0)).sum().item()
                val_fn += ((preds == 0) & (val_masks == 1)).sum().item()
                val_tn += ((preds == 0) & (val_masks == 0)).sum().item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_acc, val_prec, val_recall, val_f1 = calculate_metrics(val_tp, val_fp, val_fn, val_tn)

        # Giảm Learning Rate nếu Loss đi ngang
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"\nCảnh báo: Val Loss đi ngang, Learning Rate giảm xuống: {new_lr}")

        # ------------------------------------------
        # BÁO CÁO VÀ GHI LOG WANDB
        # ------------------------------------------
        print(f"\n[Epoch {epoch+1}] Báo cáo:")
        print(f"   Train | Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | Precision: {train_prec:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(f"   Val   | Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}\n")

        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "Loss/Train": avg_train_loss,
            "Loss/Val": avg_val_loss,
            "Accuracy/Train": train_acc,
            "Accuracy/Val": val_acc,
            "Precision/Train": train_prec,
            "Precision/Val": val_prec,
            "Recall/Train": train_recall,
            "Recall/Val": val_recall,
            "F1_Score/Train": train_f1,
            "F1_Score/Val": val_f1
        })

        # Lưu Checkpoint tốt nhất
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "unet_binary_best.pth")
            print(f"Đã lưu kỷ lục mới (Best Val F1: {best_val_f1:.4f})")
            wandb.save("unet_binary_best.pth")

    print("\n==============================================")
    print("HOÀN THÀNH HUẤN LUYỆN!")
    print("==============================================")
    wandb.finish() 

if __name__ == "__main__":
    main()
