import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset import EmbroideryDataset
from src.model import UNet
from src.utils import DiceLoss, calculate_metrics, seed_everything

def main():
    seed_everything(42)

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
            "crops_per_image": 20 
        }
    )
    config = wandb.config 

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Đang sử dụng thiết bị tính toán: {device}")

    # Khai báo Transform
    train_transform = A.Compose([
        A.OneOf([
            A.CropNonEmptyMaskIfExists(width=config.image_size, height=config.image_size, p=0.8),
            A.RandomCrop(width=config.image_size, height=config.image_size, p=0.2),
        ], p=1.0),
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5),   
        A.Affine(translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)}, scale=(0.85, 1.15), rotate=(-180, 180), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7),
        A.ElasticTransform(alpha=1, sigma=50, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.3),
        A.CoarseDropout(num_holes_range=(2, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, fill_mask=0, p=0.3),
        ToTensorV2()             
    ])

    val_transform = A.Compose([
        A.CropNonEmptyMaskIfExists(width=config.image_size, height=config.image_size),
        ToTensorV2()
    ])

    train_dataset = EmbroideryDataset(image_dir="data/train/images", mask_dir="data/train/masks", transform=train_transform, resize_factor=0.5, crops_per_image=config.crops_per_image)
    val_dataset = EmbroideryDataset(image_dir="data/val/images", mask_dir="data/val/masks", transform=val_transform, resize_factor=0.5, crops_per_image=max(1, config.crops_per_image // 2))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, persistent_workers=True) 

    model = UNet(in_channels=1, out_channels=2).to(device)
    
    class_weights = torch.tensor([1.0, config.fill_weight]).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss_fn = DiceLoss()
    
    # Khởi tạo Optimizer ban đầu
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # ĐỊNH NGHĨA FILE CHỐNG SẬP TUYỆT ĐỐI
    LAST_CHECKPOINT_PATH = "unet_binary_last.pth"
    BEST_MODEL_PATH = "unet_binary_best.pth"
    
    start_epoch = 0
    best_val_f1 = 0.0

    # KIỂM TRA PHỤC HỒI SAU SẬP
    if os.path.exists(LAST_CHECKPOINT_PATH):
        print(f"\n[PHỤC HỒI] Phát hiện sự cố sập nguồn trước đó. Đang khôi phục tiến trình từ '{LAST_CHECKPOINT_PATH}'...")
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device, weights_only=False)
        
        # Nạp lại tạ model, trạng thái optimizer và mốc Epoch đang chạy dở
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        
        print(f"-> Khôi phục thành công! Sẽ chạy tiếp từ Epoch {start_epoch + 1} với đầy đủ trí nhớ Gradient.")
    else:
        print("\nKhông phát hiện sự cố cũ. Bắt đầu huấn luyện từ Epoch 1.")

    print("\nBẮT ĐẦU HUẤN LUYỆN V4 PRO...")
    for epoch in range(start_epoch, config.epochs):
        
        # Pha Train
        model.train()
        running_train_loss = 0.0
        train_tp = train_fp = train_fn = train_tn = 0 

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.epochs}] Train")

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
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

        # Pha Validation
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

        scheduler.step(avg_val_loss)

        print(f"\n[Epoch {epoch+1}] Báo cáo:")
        print(f"   Train | Loss: {avg_train_loss:.4f} | F1: {train_f1:.4f}")
        print(f"   Val   | Loss: {avg_val_loss:.4f} | F1: {val_f1:.4f}\n")

        wandb.log({
            "epoch": epoch + 1, "learning_rate": current_lr,
            "Loss/Train": avg_train_loss, "Loss/Val": avg_val_loss,
            "F1_Score/Train": train_f1, "F1_Score/Val": val_f1
        })

        # HÀNH ĐỘNG 1: CỨ CUỐI ĐÊM LÀ GHI NHẬT KÝ (Lưu trạng thái Last để chống sập)
        checkpoint_last = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': best_val_f1
        }
        torch.save(checkpoint_last, LAST_CHECKPOINT_PATH)

        # HÀNH ĐỘNG 2: LƯU KỶ LỤC HOÀNG KIM (Best Model giữ nguyên chỉ lưu weights để đem đi Inference)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Đã lưu kỷ lục mới (Best Val F1: {best_val_f1:.4f})")
            wandb.save(BEST_MODEL_PATH)

    # Train xong xuôi an toàn thì xóa file Nhật ký chống sập đi
    if os.path.exists(LAST_CHECKPOINT_PATH):
        os.remove(LAST_CHECKPOINT_PATH)

    print("\nHOÀN THÀNH HUẤN LUYỆN!")
    wandb.finish() 

if __name__ == "__main__":
    main()
