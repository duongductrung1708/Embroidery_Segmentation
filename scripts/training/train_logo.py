import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
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

from src.dataset_svg import EmbroideryDatasetSVG
from src.model import U2NET 

# Kéo toàn bộ "đồ nghề" từ hộp utils_logo
from src.utils_logo import GeneralizedDiceLoss, FocalLoss, get_boundary_mask, calculate_metrics, calculate_metrics_torchmetrics, seed_everything

def main():
    seed_everything(42)

    # ==========================================
    # 1. CẤU HÌNH HỆ THỐNG
    # ==========================================
    TEMP_IMAGE_SIZE = 768  # Tăng từ 512 lên 768 để giữ chi tiết nhỏ
    TEMP_CROPS = 1  # QUAN TRỌNG: Chỉ cần 1 vì giờ ta nạp nguyên ảnh toàn cảnh
    BATCH_SIZE = 2  # Giảm từ 4 xuống 2 để tránh OOM với resolution 768
    NUM_CLASSES = 3  # Background, Fill, Satin

    # --- TẬP TRAIN: THU NHỎ & CHÈN VIỀN TOÀN CẢNH ---
    train_transform = A.Compose([
        # Bắt buộc: Ép logo về 512x512 mà không làm méo tỷ lệ
        A.LongestMaxSize(max_size=TEMP_IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=TEMP_IMAGE_SIZE, 
            min_width=TEMP_IMAGE_SIZE, 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=0, 
            fill_mask=0
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
        
        ToTensorV2()             
    ])

    # --- TẬP VAL: ĐỒNG BỘ VỚI TẬP TRAIN ---
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=TEMP_IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=TEMP_IMAGE_SIZE, 
            min_width=TEMP_IMAGE_SIZE, 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=0, 
            fill_mask=0
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
            fill=0, 
            fill_mask=0
        ),
        ToTensorV2()
    ])

    # ==========================================
    # 2. KHỞI TẠO DỮ LIỆU & DATALOADER
    # ==========================================
    train_dataset = EmbroideryDatasetSVG(svg_dir="data/logo/train_svg", transform=train_transform, crops_per_image=TEMP_CROPS, augment_color=True, target_size=TEMP_IMAGE_SIZE, supersample_factor=2)
    val_dataset = EmbroideryDatasetSVG(svg_dir="data/logo/val_svg", transform=val_transform, crops_per_image=TEMP_CROPS, augment_color=False, target_size=TEMP_IMAGE_SIZE, supersample_factor=2)
    tracking_dataset = EmbroideryDatasetSVG(svg_dir="data/logo/val_svg", transform=tracking_transform, crops_per_image=1, augment_color=False, target_size=TEMP_IMAGE_SIZE, supersample_factor=2)

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
        name="logo-v8-3class-improved",         
        config={                           
            "learning_rate": 1e-4,
            "architecture": "U2-Net",
            "dataset": "Logo_3Class_V2",
            "epochs": 50,
            "batch_size": BATCH_SIZE,
            "image_size": TEMP_IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
            "fill_weight": 3, 
            "satin_weight": 8,
            "crops_per_image": TEMP_CROPS,
            "label_smoothing": 0.02,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "supersample_factor": 2,
        }
    )
    config = wandb.config 

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Dang su dung thiet bi tinh toan: {device}")

    print("Dang trich xuat tap mau Toan Canh de log len WandB...")
    fixed_val_batch = next(iter(tracking_loader))
    fixed_val_images = fixed_val_batch[0].to(device)
    fixed_val_masks = fixed_val_batch[1].to(device)
    fixed_val_rgb = fixed_val_batch[2]  # Original colored SVG for visualization

    # ==========================================
    # 4. CHUẨN BỊ BỘ NÃO U-2-NET & LOSS FUNCTION
    # ==========================================
    model = U2NET(in_ch=1, out_ch=NUM_CLASSES).to(device)
    class_weights = torch.tensor([1.0, config.fill_weight, config.satin_weight]).to(device)
    
    focal_loss_fn = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.02)
    dice_loss_fn = GeneralizedDiceLoss(num_classes=NUM_CLASSES)
    bce_boundary_fn = nn.BCEWithLogitsLoss()
    
    # Deep supervision weights: cao hơn cho output chính (d0), giảm dần cho các nhánh phụ
    deep_supervision_weights = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1] 
    
    LAST_CHECKPOINT_PATH = "checkpoints/logo/u2net_logo_last.pth"
    BEST_MODEL_PATH = "checkpoints/logo/u2net_logo_best.pth"

    os.makedirs(os.path.dirname(LAST_CHECKPOINT_PATH), exist_ok=True)
    
    start_epoch = 0
    best_val_f1 = 0.0
    active_lr = config.learning_rate

    optimizer = optim.AdamW(model.parameters(), lr=active_lr, weight_decay=1e-4)
    scaler = GradScaler(device.type)  # Mixed Precision training
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

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

    EARLY_STOPPING_PATIENCE = 100
    epochs_no_improve = 0

    # ==========================================
    # 5. VÒNG LẶP HUẤN LUYỆN
    # ==========================================
    print("\nBAT DAU HUAN LUYEN LOGO 3-CLASS (GLOBAL CONTEXT)...")
    for epoch in range(start_epoch, config.epochs):
        
        # --- PHA TRAIN ---
        model.train()
        running_train_loss = 0.0
        all_train_preds = []
        all_train_masks = []
        train_rgb_samples = []  # Store RGB samples for visualization
        train_mask_samples = []  # Store mask samples for visualization
        train_pred_samples = []  # Store prediction samples for visualization

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.epochs}] Train")

        for batch_idx, (images, masks, rgb_images) in enumerate(loop):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            # Store first batch RGB images and masks for visualization
            if batch_idx == 0:
                train_rgb_samples = rgb_images[:min(4, rgb_images.size(0))]
                train_mask_samples = masks[:min(4, masks.size(0))]
            
            with torch.autocast(device_type=device.type):
                outputs = model(images)  
                boundary_targets = get_boundary_mask(masks, device, num_classes=NUM_CLASSES)
                
                loss = 0.0
                # Deep Supervision: Áp trọng số khác nhau cho từng output
                for idx, d in enumerate(outputs):
                    weight = deep_supervision_weights[idx]
                    seg_loss = focal_loss_fn(d, masks) + dice_loss_fn(d, masks)
                    loss += weight * seg_loss
                
                # Boundary loss chỉ áp dụng cho output chính (d0)
                boundary_loss = 0.0
                for class_idx in range(NUM_CLASSES):
                    boundary_loss += bce_boundary_fn(outputs[0][:, class_idx, :, :], boundary_targets[:, class_idx, :, :])
                boundary_loss /= NUM_CLASSES  # Average over classes
                loss += 0.5 * boundary_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=loss.item(), lr=current_lr)
            
            with torch.no_grad():
                preds = torch.argmax(outputs[0], dim=1)
                all_train_preds.append(preds.cpu())
                all_train_masks.append(masks.cpu())
                
                # Store predictions from first batch for visualization
                if batch_idx == 0:
                    train_pred_samples = preds[:min(4, preds.size(0))]

        avg_train_loss = running_train_loss / len(train_loader)
        # Sử dụng metrics từ torchmetrics trên toàn bộ train set
        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_masks = torch.cat(all_train_masks, dim=0)
        train_metrics = calculate_metrics_torchmetrics(all_train_preds, all_train_masks, num_classes=NUM_CLASSES)
        train_macro_f1 = train_metrics['macro_f1']
        train_iou_bg = train_metrics['iou_background']
        train_iou_fill = train_metrics['iou_fill']
        train_iou_satin = train_metrics['iou_satin']
        train_mean_iou = train_metrics['mean_iou']
        
        # Log training RGB samples with masks to wandb
        if len(train_rgb_samples) > 0 and len(train_pred_samples) > 0:
            train_wandb_images = []
            for i in range(min(4, len(train_rgb_samples))):
                rgb_np = train_rgb_samples[i].numpy()
                true_mask_np = train_mask_samples[i].cpu().numpy().astype(np.uint8)
                pred_mask_np = train_pred_samples[i].cpu().numpy().astype(np.uint8)
                
                train_wandb_images.append(wandb.Image(
                    rgb_np,
                    caption=f"Train Input #{i+1} (Epoch {epoch+1})",
                    masks={
                        "ground_truth": {
                            "mask_data": true_mask_np,
                            "class_labels": {0: "Background", 1: "Fill", 2: "Satin"},
                            "classes": [
                                {"name": "Background", "color": "#000000"},
                                {"name": "Fill", "color": "#00FFFF"},  # Cyan
                                {"name": "Satin", "color": "#FF00FF"}  # Magenta
                            ]
                        },
                        "predictions": {
                            "mask_data": pred_mask_np,
                            "class_labels": {0: "Background", 1: "Fill", 2: "Satin"},
                            "classes": [
                                {"name": "Background", "color": "#000000"},
                                {"name": "Fill", "color": "#00FFFF"},  # Cyan
                                {"name": "Satin", "color": "#FF00FF"}  # Magenta
                            ]
                        }
                    }
                ))

        # --- PHA VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        all_val_preds = []
        all_val_masks = []
        
        with torch.no_grad():
            for val_images, val_masks, val_rgb_images in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)
                
                val_boundary_targets = get_boundary_mask(val_masks, device, num_classes=NUM_CLASSES)
                
                val_loss = 0.0
                for idx, d in enumerate(val_outputs):
                    weight = deep_supervision_weights[idx]
                    v_seg_loss = focal_loss_fn(d, val_masks) + dice_loss_fn(d, val_masks)
                    val_loss += weight * v_seg_loss
                
                # Boundary loss chỉ áp dụng cho output chính (d0)
                v_boundary_loss = 0.0
                for class_idx in range(NUM_CLASSES):
                    v_boundary_loss += bce_boundary_fn(val_outputs[0][:, class_idx, :, :], val_boundary_targets[:, class_idx, :, :])
                v_boundary_loss /= NUM_CLASSES
                val_loss += 0.5 * v_boundary_loss
                    
                running_val_loss += val_loss.item()

                preds = torch.argmax(val_outputs[0], dim=1)
                all_val_preds.append(preds.cpu())
                all_val_masks.append(val_masks.cpu())

            # --- DỰ ĐOÁN ẢNH TOÀN CẢNH CHO WANDB ---
            fixed_outputs = model(fixed_val_images)
            fixed_preds = torch.argmax(fixed_outputs[0], dim=1)

            wandb_log_images = []
            num_images = min(4, fixed_val_images.size(0))
            for i in range(num_images):
                # Original colored SVG input
                rgb_np = fixed_val_rgb[i].numpy()  # RGB image
                
                # Alpha channel input (what model sees)
                img_np = fixed_val_images[i].cpu().numpy().squeeze() 
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
                    
                true_mask_np = fixed_val_masks[i].cpu().numpy().astype(np.uint8)
                pred_mask_np = fixed_preds[i].cpu().numpy().astype(np.uint8)
                
                # Log original colored SVG with masks
                wandb_img = wandb.Image(
                    rgb_np,  # Use original colored SVG as base image
                    caption=f"Original SVG #{i+1} (Epoch {epoch+1})",
                    masks={
                        "ground_truth": {
                            "mask_data": true_mask_np,
                            "class_labels": {0: "Background", 1: "Fill", 2: "Satin"},
                            "classes": [
                                {"name": "Background", "color": "#000000"},
                                {"name": "Fill", "color": "#00FFFF"},  # Cyan
                                {"name": "Satin", "color": "#FF00FF"}  # Magenta
                            ]
                        },
                        "predictions": {
                            "mask_data": pred_mask_np,
                            "class_labels": {0: "Background", 1: "Fill", 2: "Satin"},
                            "classes": [
                                {"name": "Background", "color": "#000000"},
                                {"name": "Fill", "color": "#00FFFF"},  # Cyan
                                {"name": "Satin", "color": "#FF00FF"}  # Magenta
                            ]
                        }
                    }
                )
                wandb_log_images.append(wandb_img)
                
                # Also log alpha channel input for comparison
                wandb_log_images.append(wandb.Image(
                    img_np,
                    caption=f"Alpha Input #{i+1} (Epoch {epoch+1})"
                ))

        avg_val_loss = running_val_loss / len(val_loader)
        # Sử dụng metrics từ torchmetrics trên toàn bộ val set
        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_masks = torch.cat(all_val_masks, dim=0)
        val_metrics = calculate_metrics_torchmetrics(all_val_preds, all_val_masks, num_classes=NUM_CLASSES)
        val_macro_f1 = val_metrics['macro_f1']
        val_iou_bg = val_metrics['iou_background']
        val_iou_fill = val_metrics['iou_fill']
        val_iou_satin = val_metrics['iou_satin']
        val_mean_iou = val_metrics['mean_iou']

        scheduler.step()

        print(f"\n[Epoch {epoch+1}] Bao cao U-2-NET (Logo 3-class):")
        print(f"   Train | Loss: {avg_train_loss:.4f} | Macro F1: {train_macro_f1:.4f} | Mean IoU: {train_mean_iou:.4f}")
        print(f"          IoU - BG: {train_iou_bg:.4f} | Fill: {train_iou_fill:.4f} | Satin: {train_iou_satin:.4f}")
        print(f"   Val   | Loss: {avg_val_loss:.4f} | Macro F1: {val_macro_f1:.4f} | Mean IoU: {val_mean_iou:.4f}")
        print(f"          IoU - BG: {val_iou_bg:.4f} | Fill: {val_iou_fill:.4f} | Satin: {val_iou_satin:.4f}\n")

        wandb.log({
            "epoch": epoch + 1, 
            "learning_rate": current_lr,
            "Loss/Train": avg_train_loss, 
            "Loss/Val": avg_val_loss,
            "F1_Macro/Train": train_macro_f1,
            "F1_Macro/Val": val_macro_f1,
            "IoU_Mean/Train": train_mean_iou,
            "IoU_Mean/Val": val_mean_iou,
            "IoU_Background/Train": train_iou_bg,
            "IoU_Background/Val": val_iou_bg,
            "IoU_Fill/Train": train_iou_fill,
            "IoU_Fill/Val": val_iou_fill,
            "IoU_Satin/Train": train_iou_satin,
            "IoU_Satin/Val": val_iou_satin,
            "Train_Images": train_wandb_images if len(train_rgb_samples) > 0 else None,
            "Validation_Images": wandb_log_images
        })

        checkpoint_last = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': best_val_f1
        }
        torch.save(checkpoint_last, LAST_CHECKPOINT_PATH)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Da luu ky luc moi (Best Val Macro F1: {best_val_f1:.4f})")
            wandb.save(BEST_MODEL_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation Macro F1 khong tang. Canh bao: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nMO HINH DA HOI TU TAI EPOCH {epoch + 1}! Da kich hoat Dung Som de tiet kiem thoi gian.")
            break

    if os.path.exists(LAST_CHECKPOINT_PATH):
        os.remove(LAST_CHECKPOINT_PATH)

    print("\nHOAN THANH HUAN LUYEN LOGO 3-CLASS!")
    wandb.finish() 

if __name__ == "__main__":
    main()
