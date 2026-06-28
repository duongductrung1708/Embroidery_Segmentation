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

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import EmbroideryDataset
from src.model import U2NET 

# Kéo toàn bộ "đồ nghề" từ hộp utils
from src.utils import GeneralizedDiceLoss, FocalLoss, get_boundary_mask, calculate_metrics, calculate_metrics_torchmetrics, seed_everything

def main():
    seed_everything(42)

    # ==========================================
    # 1. CẤU HÌNH HỆ THỐNG
    # ==========================================
    TEMP_IMAGE_SIZE = 768  # Tăng từ 512 lên 768 để giữ chi tiết nhỏ
    TEMP_CROPS = 20
    BATCH_SIZE = 4
    NUM_CLASSES = 3  # Background, Fill, Satin

    # --- TẬP TRAIN: Cắt nhỏ & Data Augmentation Hạng nặng ---
    train_transform = A.Compose([
        A.OneOf([
            A.CropNonEmptyMaskIfExists(width=TEMP_IMAGE_SIZE, height=TEMP_IMAGE_SIZE, p=0.8),
            A.RandomCrop(width=TEMP_IMAGE_SIZE, height=TEMP_IMAGE_SIZE, p=0.2),
        ], p=1.0),
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5),   
        A.Affine(translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)}, scale=(0.85, 1.15), rotate=(-180, 180), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7),
        A.ElasticTransform(alpha=1, sigma=50, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.3),
        A.CLAHE(p=0.5), 
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5), 
        A.CoarseDropout(num_holes_range=(2, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, fill_mask=0, p=0.3),
        ToTensorV2()             
    ])

    # --- TẬP VAL: Cắt nhỏ để tính Loss cho nhẹ VRAM ---
    val_transform = A.Compose([
        A.CropNonEmptyMaskIfExists(width=TEMP_IMAGE_SIZE, height=TEMP_IMAGE_SIZE),
        ToTensorV2()
    ])

    # --- TẬP TRACKING: Kính lúp Góc Rộng (Tuyệt đối không cắt) ---
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
    train_dataset = EmbroideryDataset(image_dir="data/lineart/train/images", mask_dir="data/lineart/train/masks", transform=train_transform, resize_factor=0.5, crops_per_image=TEMP_CROPS)
    val_dataset = EmbroideryDataset(image_dir="data/lineart/val/images", mask_dir="data/lineart/val/masks", transform=val_transform, resize_factor=0.5, crops_per_image=max(1, TEMP_CROPS // 2))
    tracking_dataset = EmbroideryDataset(image_dir="data/lineart/val/images", mask_dir="data/lineart/val/masks", transform=tracking_transform, resize_factor=0.5, crops_per_image=1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True) 
    tracking_loader = DataLoader(tracking_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_train_images = len(train_dataset) // TEMP_CROPS 
    num_val_images = len(val_dataset) // max(1, TEMP_CROPS // 2)

    print(f"Number of training images: {num_train_images}")
    print(f"Number of validation images: {num_val_images}")

    # ==========================================
    # 3. KHỞI TẠO WANDB & THIẾT BỊ
    # ==========================================
    wandb.init(
        project="embroidery-segmentation", 
        name="v7-u2net-3class-improved",         
        config={                           
            "learning_rate": 1e-4,
            "architecture": "U2-Net",
            "dataset": "Embroidery_V3",
            "epochs": 50,
            "batch_size": BATCH_SIZE,
            "image_size": TEMP_IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
            "fill_weight": 2.5,
            "satin_weight": 2.5,
            "crops_per_image": TEMP_CROPS,
            "label_smoothing": 0.02
        }
    )
    config = wandb.config 

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Dang su dung thiet bi tinh toan: {device}")

    # --- Trích xuất 1 Lô Ảnh Toàn Cảnh (Góc rộng) cố định ---
    print("Dang trich xuat tap mau Toan Canh de log len WandB...")
    fixed_val_batch = next(iter(tracking_loader))
    fixed_val_images = fixed_val_batch[0].to(device)
    fixed_val_masks = fixed_val_batch[1].to(device)

    # ==========================================
    # 4. CHUẨN BỊ BỘ NÃO U-2-NET & LOSS FUNCTION
    # ==========================================
    model = U2NET(in_ch=1, out_ch=NUM_CLASSES).to(device)
    
    # Class weights cho 3 lớp: Background, Fill, Satin
    class_weights = torch.tensor([1.0, config.fill_weight, config.satin_weight]).to(device)
    
    focal_loss_fn = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.02)
    dice_loss_fn = GeneralizedDiceLoss(num_classes=NUM_CLASSES, weights=[1.0, 2.0, 2.0])
    bce_boundary_fn = nn.BCEWithLogitsLoss()
    
    # Deep supervision weights: cao hơn cho output chính (d0), giảm dần cho các nhánh phụ
    deep_supervision_weights = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1] 
    
    LAST_CHECKPOINT_PATH = "checkpoints/lineart/u2net_last.pth"
    BEST_MODEL_PATH = "checkpoints/lineart/u2net_best.pth"
    
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
        print("\n[TRAIN MOI] Bat dau huan luyen U2-Net...")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    EARLY_STOPPING_PATIENCE = 12
    epochs_no_improve = 0

    # ==========================================
    # 5. VÒNG LẶP HUẤN LUYỆN
    # ==========================================
    print("\nBAT DAU HUAN LUYEN V6 (U2-NET DEEP SUPERVISION)...")
    for epoch in range(start_epoch, config.epochs):
        
        # --- PHA TRAIN ---
        model.train()
        running_train_loss = 0.0
        all_train_preds = []
        all_train_masks = []

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.epochs}] Train")

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)  # U2-Net trả về 7 bản đồ
            boundary_targets = get_boundary_mask(masks, device, num_classes=NUM_CLASSES)
            
            loss = 0.0
            # Deep Supervision: Áp trọng số khác nhau cho từng output
            for idx, d in enumerate(outputs):
                weight = deep_supervision_weights[idx]
                seg_loss = focal_loss_fn(d, masks) + dice_loss_fn(d, masks)
                # Multi-class boundary loss: tính cho tất cả các lớp
                boundary_loss = 0.0
                for class_idx in range(NUM_CLASSES):
                    boundary_loss += bce_boundary_fn(d[:, class_idx, :, :], boundary_targets[:, class_idx, :, :])
                boundary_loss /= NUM_CLASSES  # Average over classes
                loss += weight * (seg_loss + 0.5 * boundary_loss)
            
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=loss.item(), lr=current_lr)
            
            with torch.no_grad():
                preds = torch.argmax(outputs[0], dim=1) # Chỉ đánh giá độ chính xác trên d0
                all_train_preds.append(preds.cpu())
                all_train_masks.append(masks.cpu()) 

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

        # --- PHA VALIDATION (CẮT NHỎ TÍNH LOSS) ---
        model.eval()
        running_val_loss = 0.0
        all_val_preds = []
        all_val_masks = []
        
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)
                
                val_boundary_targets = get_boundary_mask(val_masks, device, num_classes=NUM_CLASSES)
                
                val_loss = 0.0
                for idx, d in enumerate(val_outputs):
                    weight = deep_supervision_weights[idx]
                    v_seg_loss = focal_loss_fn(d, val_masks) + dice_loss_fn(d, val_masks)
                    # Multi-class boundary loss
                    v_boundary_loss = 0.0
                    for class_idx in range(NUM_CLASSES):
                        v_boundary_loss += bce_boundary_fn(d[:, class_idx, :, :], val_boundary_targets[:, class_idx, :, :])
                    v_boundary_loss /= NUM_CLASSES
                    val_loss += weight * (v_seg_loss + 0.5 * v_boundary_loss)
                    
                running_val_loss += val_loss.item()

                preds = torch.argmax(val_outputs[0], dim=1)
                all_val_preds.append(preds.cpu())
                all_val_masks.append(val_masks.cpu())

            # --- DỰ ĐOÁN ẢNH TOÀN CẢNH (GÓC RỘNG) CHO WANDB ---
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
                            "class_labels": {0: "Background", 1: "Fill", 2: "Satin"}
                        },
                        "predictions": {
                            "mask_data": pred_mask_np,
                            "class_labels": {0: "Background", 1: "Fill", 2: "Satin"}
                        }
                    }
                )
                wandb_log_images.append(wandb_img)

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

        scheduler.step(avg_val_loss)

        print(f"\n[Epoch {epoch+1}] Bao cao U-2-NET (3-class):")
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

    print("\nHOAN THANH HUAN LUYEN V6!")
    wandb.finish() 

if __name__ == "__main__":
    main()
