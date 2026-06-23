import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset import EmbroideryDataset
from src.model import UNet
from src.utils import DiceLoss, calculate_metrics, seed_everything

def main():
    seed_everything(42)

    # ==========================================
    # 1. CẤU HÌNH & KHAI BÁO DỮ LIỆU ĐỂ ĐẾM
    # ==========================================
    TEMP_IMAGE_SIZE = 512
    TEMP_CROPS = 20
    BATCH_SIZE = 4

    train_transform = A.Compose([
        A.OneOf([
            A.CropNonEmptyMaskIfExists(width=TEMP_IMAGE_SIZE, height=TEMP_IMAGE_SIZE, p=0.8),
            A.RandomCrop(width=TEMP_IMAGE_SIZE, height=TEMP_IMAGE_SIZE, p=0.2),
        ], p=1.0),
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5),   
        A.Affine(translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)}, scale=(0.85, 1.15), rotate=(-180, 180), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7),
        A.ElasticTransform(alpha=1, sigma=50, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.3),
        A.CoarseDropout(num_holes_range=(2, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, fill_mask=0, p=0.3),
        ToTensorV2()             
    ])

    val_transform = A.Compose([
        A.CropNonEmptyMaskIfExists(width=TEMP_IMAGE_SIZE, height=TEMP_IMAGE_SIZE),
        ToTensorV2()
    ])

    # Khởi tạo Dataset trước để đo lường độ lớn của Data
    train_dataset = EmbroideryDataset(image_dir="data/train/images", mask_dir="data/train/masks", transform=train_transform, resize_factor=0.5, crops_per_image=TEMP_CROPS)
    val_dataset = EmbroideryDataset(image_dir="data/val/images", mask_dir="data/val/masks", transform=val_transform, resize_factor=0.5, crops_per_image=max(1, TEMP_CROPS // 2))

    num_train_images = len(train_dataset) // TEMP_CROPS 
    num_val_images = len(val_dataset) // max(1, TEMP_CROPS // 2)

    # ==========================================
    # 2. KHỞI TẠO WANDB VỚI ĐẦY ĐỦ THÔNG SỐ ĐỘNG
    # ==========================================
    wandb.init(
        project="embroidery-segmentation", 
        name="v4-pro-augmentation-onthefly",         
        config={                           
            "learning_rate": 1e-4,
            "architecture": "U-Net",
            "dataset": "Embroidery_V2",
            "epochs": 50,
            "batch_size": BATCH_SIZE,
            "image_size": TEMP_IMAGE_SIZE,
            "fill_weight": 2.5,
            "crops_per_image": TEMP_CROPS,
            
            # Thông số thực tế đếm được từ thư mục
            "num_raw_train_images": num_train_images,
            "num_raw_val_images": num_val_images,
            "total_train_crops_per_epoch": len(train_dataset),
            "total_val_crops_per_epoch": len(val_dataset)
        }
    )
    config = wandb.config 

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Dang su dung thiet bi tinh toan: {device}")

    # ==========================================
    # 3. CHUẨN BỊ MÔ HÌNH VÀ MÔI TRƯỜNG
    # ==========================================
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, persistent_workers=True) 

    model = UNet(in_channels=1, out_channels=2).to(device)
    class_weights = torch.tensor([1.0, config.fill_weight]).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss_fn = DiceLoss()
    
    LAST_CHECKPOINT_PATH = "unet_binary_last.pth"
    BEST_MODEL_PATH = "unet_binary_best.pth"
    
    start_epoch = 0
    best_val_f1 = 0.0
    active_lr = config.learning_rate

    optimizer = optim.Adam(model.parameters(), lr=active_lr)

    # Cơ chế Chống Sập & Fine-Tune
    if os.path.exists(LAST_CHECKPOINT_PATH):
        try:
            print(f"\n[PHUC HOI] Phat hien su co sap nguon. Dang khoi phuc tu '{LAST_CHECKPOINT_PATH}'...")
            checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device, weights_only=False)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint.get('best_val_f1', 0.0)
            print(f"-> Thanh cong! Se chay tiep tu Epoch {start_epoch + 1}.")
        except KeyError:
            print(f"\nLOI: File '{LAST_CHECKPOINT_PATH}' bi sai dinh dang nhat ky!")
            print("-> Hay xoa file 'unet_binary_last.pth' di va chay lai.")
            exit()
            
    elif os.path.exists(BEST_MODEL_PATH):
        print(f"\n[FINE-TUNE] Phat hien ta cu '{BEST_MODEL_PATH}'. Nap tri nho de hoc them anh moi...")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
        
        active_lr = 1e-5 
        for param_group in optimizer.param_groups:
            param_group['lr'] = active_lr
        print(f"-> Da ha Learning Rate xuong {active_lr} de got giua an toan.")
        
    else:
        print("\n[TRAIN MOI] Khong tim thay du lieu cu. Bat dau train tu to giay trang...")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # ==========================================
    # CƠ CHẾ DỪNG SỚM (EARLY STOPPING)
    # ==========================================
    EARLY_STOPPING_PATIENCE = 7
    epochs_no_improve = 0

    # ==========================================
    # 4. VÒNG LẶP HUẤN LUYỆN CHÍNH
    # ==========================================
    print("\nBAT DAU HUAN LUYEN V4 PRO...")
    for epoch in range(start_epoch, config.epochs):
        
        # --- PHA TRAIN ---
        model.train()
        running_train_loss = 0.0
        train_tp = train_fp = train_fn = train_tn = 0 

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.epochs}] Train")

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = ce_loss_fn(outputs, masks) + dice_loss_fn(outputs, masks)
            
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

        # --- PHA VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        val_tp = val_fp = val_fn = val_tn = 0 
        
        wandb_log_images = []

        with torch.no_grad():
            for batch_idx, (val_images, val_masks) in enumerate(val_loader):
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)
                
                val_loss = ce_loss_fn(val_outputs, val_masks) + dice_loss_fn(val_outputs, val_masks)
                running_val_loss += val_loss.item()

                preds = torch.argmax(val_outputs, dim=1)
                
                val_tp += ((preds == 1) & (val_masks == 1)).sum().item()
                val_fp += ((preds == 1) & (val_masks == 0)).sum().item()
                val_fn += ((preds == 0) & (val_masks == 1)).sum().item()
                val_tn += ((preds == 0) & (val_masks == 0)).sum().item()
                
                # --- CHỤP ẢNH VALIDATION LOG LÊN W&B ---
                if batch_idx == 0:
                    num_images = min(4, val_images.size(0))
                    for i in range(num_images):
                        img_np = val_images[i].cpu().numpy().squeeze() 
                        
                        # Fix lỗi torch.uint8 thành np.uint8
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img_np.astype(np.uint8)
                            
                        true_mask_np = val_masks[i].cpu().numpy().astype(np.uint8)
                        pred_mask_np = preds[i].cpu().numpy().astype(np.uint8)
                        
                        wandb_img = wandb.Image(
                            img_np, 
                            caption=f"Mau Validation so {i+1}",
                            masks={
                                "ground_truth": {
                                    "mask_data": true_mask_np,
                                    "class_labels": {0: "Nen", 1: "Fill chuan"}
                                },
                                "predictions": {
                                    "mask_data": pred_mask_np,
                                    "class_labels": {0: "Nen", 1: "AI Du doan"}
                                }
                            }
                        )
                        wandb_log_images.append(wandb_img)

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc, val_prec, val_recall, val_f1 = calculate_metrics(val_tp, val_fp, val_fn, val_tn)

        scheduler.step(avg_val_loss)

        # --- BÁO CÁO & GHI LOG W&B ---
        print(f"\n[Epoch {epoch+1}] Bao cao:")
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
            "F1_Score/Val": val_f1,
            "Validation_Images": wandb_log_images
        })

        # --- GHI NHẬT KÝ ---
        checkpoint_last = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': best_val_f1
        }
        torch.save(checkpoint_last, LAST_CHECKPOINT_PATH)

        # --- KIỂM TRA HỘI TỤ (EARLY STOPPING) ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Da luu ky luc moi (Best Val F1: {best_val_f1:.4f})")
            wandb.save(BEST_MODEL_PATH)
            
            # Reset bộ đếm kiên nhẫn
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation F1 khong tang. Canh bao: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        # Kéo phanh khẩn cấp nếu cạn kiệt kiên nhẫn
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nMO HINH DA HOI TU TAI EPOCH {epoch + 1}! Da kich hoat Dung Som de tiet kiem thoi gian.")
            break

    if os.path.exists(LAST_CHECKPOINT_PATH):
        os.remove(LAST_CHECKPOINT_PATH)

    print("\nHOAN THANH HUAN LUYEN!")
    wandb.finish() 

if __name__ == "__main__":
    main()
