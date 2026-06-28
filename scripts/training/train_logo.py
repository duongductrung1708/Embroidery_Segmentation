import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import sys
import cv2
import numpy as np
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Xác định đường dẫn gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT)) 

from src.dataset_logo import EmbroideryDatasetLogo
from src.model import U2NET 

# Kéo toàn bộ "đồ nghề" từ hộp utils_logo
from src.utils_logo import DiceLoss, FocalLoss, get_boundary_mask, calculate_metrics, seed_everything

def main():
    seed_everything(42)

    # ==========================================
    # 1. CẤU HÌNH HỆ THỐNG
    # ==========================================
    TEMP_IMAGE_SIZE = 512
    TEMP_CROPS = 1  # QUAN TRỌNG: Chỉ cần 1 vì giờ ta nạp nguyên ảnh toàn cảnh
    BATCH_SIZE = 4

    # --- TẬP TRAIN: THU NHỎ & CHÈN VIỀN TOÀN CẢNH ---
    train_transform = A.Compose([
        # Bắt buộc: Ép logo về 512x512 mà không làm méo tỷ lệ
        A.LongestMaxSize(max_size=TEMP_IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=TEMP_IMAGE_SIZE, 
            min_width=TEMP_IMAGE_SIZE, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, 
            mask_value=0
        ),
        
        # Các phép biến đổi hình học (Rất cần thiết để AI học hình dáng đa dạng)
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5),   
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, 
            scale=(0.85, 1.15), 
            rotate=(-180, 180), 
            interpolation=cv2.INTER_NEAREST, # Bắt buộc NEAREST để giữ nguyên nhãn 0,1,2
            border_mode=cv2.BORDER_CONSTANT, 
            fill=0, 
            fill_mask=0, 
            p=0.7
        ),
        A.ElasticTransform(
            alpha=1, 
            sigma=50, 
            interpolation=cv2.INTER_NEAREST, 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=0, 
            fill_mask=0, 
            p=0.3
        ),
        
        ToTensorV2()             
    ])

    # --- TẬP VAL: ĐỒNG BỘ VỚI TẬP TRAIN ---
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=TEMP_IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=TEMP_IMAGE_SIZE, 
            min_width=TEMP_IMAGE_SIZE, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, 
            mask_value=0
        ),
        ToTensorV2()
    ])

    # --- TẬP TRACKING: Vẫn giữ nguyên ---
    tracking_transform = A.Compose([
        A.LongestMaxSize(max_size=TEMP_IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=TEMP_IMAGE_SIZE, 
            min_width=TEMP_IMAGE_SIZE, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0, 
            mask_value=0
        ),
        ToTensorV2()
    ])

    # ==========================================
    # 2. KHỞI TẠO DỮ LIỆU & DATALOADER
    # ==========================================
    train_dataset = EmbroideryDatasetLogo(image_dir="data/logo/train/images", mask_dir="data/logo/train/masks", transform=train_transform, resize_factor=0.5, crops_per_image=TEMP_CROPS)
    val_dataset = EmbroideryDatasetLogo(image_dir="data/logo/val/images", mask_dir="data/logo/val/masks", transform=val_transform, resize_factor=0.5, crops_per_image=TEMP_CROPS)
    tracking_dataset = EmbroideryDatasetLogo(image_dir="data/logo/val/images", mask_dir="data/logo/val/masks", transform=tracking_transform, resize_factor=0.5, crops_per_image=1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True) 
    tracking_loader = DataLoader(tracking_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Number of training images (Full Scale): {len(train_dataset)}")
    print(f"Number of validation images (Full Scale): {len(val_dataset)}")

    # ==========================================
    # 3. KHỞI TẠO WANDB & THIẾT BỊ
    # ==========================================
    wandb.init(
        project="embroidery-segmentation", 
        name="logo-3class-global-context",         
        config={                           
            "learning_rate": 1e-4,
            "architecture": "U2-Net",
            "dataset": "Logo_3Class",
            "epochs": 50,
            "batch_size": BATCH_SIZE,
            "image_size": TEMP_IMAGE_SIZE,
            "fill_weight": 2.5, 
            "satin_weight": 2.5,
            "crops_per_image": TEMP_CROPS
        }
    )
    config = wandb.config 

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Dang su dung thiet bi tinh toan: {device}")

    print("Dang trich xuat tap mau Toan Canh de log len WandB...")
    fixed_val_batch = next(iter(tracking_loader))
    fixed_val_images = fixed_val_batch[0].to(device)
    fixed_val_masks = fixed_val_batch[1].to(device)

    # ==========================================
    # 4. CHUẨN BỊ BỘ NÃO U-2-NET & LOSS FUNCTION
    # ==========================================
    model = U2NET(in_ch=1, out_ch=3).to(device)
    class_weights = torch.tensor([1.0, config.fill_weight, config.satin_weight]).to(device)
    
    focal_loss_fn = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.1)
    dice_loss_fn = DiceLoss()
    bce_boundary_fn = nn.BCEWithLogitsLoss() 
    
    LAST_CHECKPOINT_PATH = "checkpoints/logo/u2net_logo_last.pth"
    BEST_MODEL_PATH = "checkpoints/logo/u2net_logo_best.pth"

    os.makedirs(os.path.dirname(LAST_CHECKPOINT_PATH), exist_ok=True)
    
    start_epoch = 0
    best_val_f1 = 0.0
    active_lr = config.learning_rate

    optimizer = optim.Adam(model.parameters(), lr=active_lr)

    if os.path.exists(LAST_CHECKPOINT_PATH):
        try:
            print(f"\n[PHUC HOI] Khoi phuc tu '{LAST_CHECKPOINT_PATH}'...")
            checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint.get('best_val_f1', 0.0)
            print(f"-> Thanh cong! Chay tiep tu Epoch {start_epoch + 1}.")
        except KeyError:
            print(f"\nLOI: File '{LAST_CHECKPOINT_PATH}' sai dinh dang!")
            exit()
    else:
        print("\n[TRAIN MOI] Bat dau huan luyen U2-Net cho Logo 3-class...")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    EARLY_STOPPING_PATIENCE = 12
    epochs_no_improve = 0

    # ==========================================
    # 5. VÒNG LẶP HUẤN LUYỆN
    # ==========================================
    print("\nBAT DAU HUAN LUYEN LOGO 3-CLASS (GLOBAL CONTEXT)...")
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
            boundary_targets = get_boundary_mask(masks, device)
            
            loss = 0.0
            # Deep Supervision
            for d in outputs:
                seg_loss = focal_loss_fn(d, masks) + dice_loss_fn(d, masks)
                boundary_loss = bce_boundary_fn(d[:, 1, :, :], boundary_targets)
                loss += (seg_loss + 0.5 * boundary_loss)
            
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=loss.item(), lr=current_lr)
            
            with torch.no_grad():
                preds = torch.argmax(outputs[0], dim=1) 
                for cls in range(3):
                    train_tp += ((preds == cls) & (masks == cls)).sum().item()
                    train_fp += ((preds == cls) & (masks != cls)).sum().item()
                    train_fn += ((preds != cls) & (masks == cls)).sum().item()
                    train_tn += ((preds != cls) & (masks != cls)).sum().item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_acc, train_prec, train_recall, train_f1 = calculate_metrics(train_tp, train_fp, train_fn, train_tn)

        # --- PHA VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        val_tp = val_fp = val_fn = val_tn = 0 
        
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)
                
                val_boundary_targets = get_boundary_mask(val_masks, device)
                
                val_loss = 0.0
                for d in val_outputs:
                    v_seg_loss = focal_loss_fn(d, val_masks) + dice_loss_fn(d, val_masks)
                    v_bound_loss = bce_boundary_fn(d[:, 1, :, :], val_boundary_targets)
                    val_loss += (v_seg_loss + 0.5 * v_bound_loss)
                    
                running_val_loss += val_loss.item()

                preds = torch.argmax(val_outputs[0], dim=1)
                for cls in range(3):
                    val_tp += ((preds == cls) & (val_masks == cls)).sum().item()
                    val_fp += ((preds == cls) & (val_masks != cls)).sum().item()
                    val_fn += ((preds != cls) & (val_masks == cls)).sum().item()
                    val_tn += ((preds != cls) & (val_masks != cls)).sum().item()

            # --- DỰ ĐOÁN ẢNH TOÀN CẢNH CHO WANDB ---
            fixed_outputs = model(fixed_val_images)
            fixed_preds = torch.argmax(fixed_outputs[0], dim=1)

            wandb_log_images = []
            num_images = min(4, fixed_val_images.size(0))
            for i in range(num_images):
                img_np = fixed_val_images[i].cpu().numpy().squeeze() 
                
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
                    
                true_mask_np = fixed_val_masks[i].cpu().numpy().astype(np.uint8)
                pred_mask_np = fixed_preds[i].cpu().numpy().astype(np.uint8)
                
                wandb_img = wandb.Image(
                    img_np, 
                    caption=f"Toan Canh #{i+1} (Epoch {epoch+1})",
                    masks={
                        "ground_truth": {
                            "mask_data": true_mask_np,
                            "class_labels": {0: "Nen", 1: "Fill", 2: "Satin"}
                        },
                        "predictions": {
                            "mask_data": pred_mask_np,
                            "class_labels": {0: "Nen", 1: "Fill", 2: "Satin"}
                        }
                    }
                )
                wandb_log_images.append(wandb_img)

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc, val_prec, val_recall, val_f1 = calculate_metrics(val_tp, val_fp, val_fn, val_tn)

        scheduler.step(avg_val_loss)

        print(f"\n[Epoch {epoch+1}] Bao cao U-2-NET:")
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

        checkpoint_last = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': best_val_f1
        }
        torch.save(checkpoint_last, LAST_CHECKPOINT_PATH)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Da luu ky luc moi (Best Val F1: {best_val_f1:.4f})")
            wandb.save(BEST_MODEL_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation F1 khong tang. Canh bao: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nMO HINH DA HOI TU TAI EPOCH {epoch + 1}! Da kich hoat Dung Som de tiet kiem thoi gian.")
            break

    if os.path.exists(LAST_CHECKPOINT_PATH):
        os.remove(LAST_CHECKPOINT_PATH)

    print("\nHOAN THANH HUAN LUYEN LOGO 3-CLASS!")
    wandb.finish() 

if __name__ == "__main__":
    main()
