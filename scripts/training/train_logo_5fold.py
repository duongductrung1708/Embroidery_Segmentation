import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler
from tqdm import tqdm
import wandb
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import xml.etree.ElementTree as ET
import shutil

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Xác định đường dẫn gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_svg import EmbroideryDatasetSVG
from src.model import U2NET
from src.utils_logo import GeneralizedDiceLoss, FocalLoss, get_boundary_mask, calculate_metrics_torchmetrics, seed_everything

# SVG namespace for parsing
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"


def count_satin_fill_paths(svg_path: Path) -> tuple:
    """Count satin and fill paths in an SVG file."""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        fill_count = 0
        satin_count = 0
        
        for elem in root.findall(".//svg:path", SVG_NS):
            label = elem.attrib.get(INKSCAPE_LABEL, "").strip().lower()
            if label == "fill":
                fill_count += 1
            elif label == "satin":
                satin_count += 1
        
        return fill_count, satin_count
    except Exception as e:
        print(f"Error parsing {svg_path}: {e}")
        return 0, 0


def calculate_satin_ratio(fill_count: int, satin_count: int) -> float:
    """Calculate satin ratio (satin / total paths)."""
    total = fill_count + satin_count
    if total == 0:
        return 0.0
    return satin_count / total


def bucket_satin_ratio(ratio: float, n_buckets: int = 4) -> int:
    """Bucket satin ratio into discrete bins for stratification."""
    return min(int(ratio * n_buckets), n_buckets - 1)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def train_one_epoch(model, train_loader, optimizer, scaler, device, focal_loss_fn, dice_loss_fn, 
                    bce_boundary_fn, deep_supervision_weights, num_classes, epoch, config):
    """Train for one epoch and return metrics."""
    model.train()
    running_train_loss = 0.0
    all_train_preds = []
    all_train_masks = []
    train_rgb_samples = []
    train_mask_samples = []
    train_pred_samples = []

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.epochs}] Train")

    for batch_idx, (images, masks, rgb_images) in enumerate(loop):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        if batch_idx == 0:
            train_rgb_samples = rgb_images[:min(4, rgb_images.size(0))]
            train_mask_samples = masks[:min(4, masks.size(0))]
        
        with torch.autocast(device_type=device.type):
            outputs = model(images)
            boundary_targets = get_boundary_mask(masks, device, num_classes=num_classes)
            
            loss = 0.0
            for idx, d in enumerate(outputs):
                weight = deep_supervision_weights[idx]
                seg_loss = focal_loss_fn(d, masks) + dice_loss_fn(d, masks)
                loss += weight * seg_loss
            
            boundary_loss = 0.0
            for class_idx in range(num_classes):
                boundary_loss += bce_boundary_fn(outputs[0][:, class_idx, :, :], boundary_targets[:, class_idx, :, :])
            boundary_loss /= num_classes
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
            if batch_idx == 0:
                train_pred_samples = preds[:min(4, preds.size(0))]

    avg_train_loss = running_train_loss / len(train_loader)
    all_train_preds = torch.cat(all_train_preds, dim=0)
    all_train_masks = torch.cat(all_train_masks, dim=0)
    
    # Calculate metrics
    train_metrics = calculate_metrics_torchmetrics(all_train_preds, all_train_masks, num_classes=num_classes)
    train_macro_f1 = train_metrics['macro_f1']
    train_mean_iou = train_metrics['mean_iou']
    train_iou_bg = train_metrics['iou_background']
    train_iou_fill = train_metrics['iou_fill']
    train_iou_satin = train_metrics['iou_satin']
    
    # Create wandb images
    train_wandb_images = []
    if len(train_rgb_samples) > 0 and len(train_pred_samples) > 0:
        for i in range(min(4, len(train_rgb_samples))):
            rgb_np = train_rgb_samples[i].numpy()
            true_mask_np = train_mask_samples[i].cpu().numpy().astype(np.uint8)
            pred_mask_np = train_pred_samples[i].cpu().numpy().astype(np.uint8)
            
            # Create colored masks
            true_mask_colored = np.zeros((true_mask_np.shape[0], true_mask_np.shape[1], 3), dtype=np.uint8)
            pred_mask_colored = np.zeros((pred_mask_np.shape[0], pred_mask_np.shape[1], 3), dtype=np.uint8)
            
            true_mask_colored[true_mask_np == 1] = [0, 255, 255]
            true_mask_colored[true_mask_np == 2] = [255, 0, 255]
            pred_mask_colored[pred_mask_np == 1] = [0, 255, 255]
            pred_mask_colored[pred_mask_np == 2] = [255, 0, 255]
            
            train_wandb_images.append(wandb.Image(rgb_np, caption=f"Train Input #{i+1}"))
            train_wandb_images.append(wandb.Image(true_mask_colored, caption=f"Train GT #{i+1}"))
            train_wandb_images.append(wandb.Image(pred_mask_colored, caption=f"Train Pred #{i+1}"))

    return {
        'loss': avg_train_loss,
        'macro_f1': train_macro_f1,
        'mean_iou': train_mean_iou,
        'iou_bg': train_iou_bg,
        'iou_fill': train_iou_fill,
        'iou_satin': train_iou_satin,
        'wandb_images': train_wandb_images
    }


def validate_one_epoch(model, val_loader, device, focal_loss_fn, dice_loss_fn, 
                       bce_boundary_fn, deep_supervision_weights, num_classes, fixed_val_batch):
    """Validate for one epoch and return metrics."""
    model.eval()
    running_val_loss = 0.0
    all_val_preds = []
    all_val_masks = []
    
    fixed_val_images = fixed_val_batch[0].to(device)
    fixed_val_masks = fixed_val_batch[1].to(device)
    fixed_val_rgb = fixed_val_batch[2]
    
    with torch.no_grad():
        for val_images, val_masks, _ in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)
            val_boundary_targets = get_boundary_mask(val_masks, device, num_classes=num_classes)
            
            val_loss = 0.0
            for idx, d in enumerate(val_outputs):
                weight = deep_supervision_weights[idx]
                val_loss += weight * (focal_loss_fn(d, val_masks) + dice_loss_fn(d, val_masks))
            
            v_boundary_loss = 0.0
            for class_idx in range(num_classes):
                v_boundary_loss += bce_boundary_fn(val_outputs[0][:, class_idx, :, :], val_boundary_targets[:, class_idx, :, :])
            val_loss += 0.5 * (v_boundary_loss / num_classes)
                
            running_val_loss += val_loss.item()
            preds = torch.argmax(val_outputs[0], dim=1)
            all_val_preds.append(preds.cpu())
            all_val_masks.append(val_masks.cpu())

        # Fixed batch for visualization
        fixed_outputs = model(fixed_val_images)
        fixed_preds = torch.argmax(fixed_outputs[0], dim=1)

        wandb_log_images = []
        num_images = min(4, fixed_val_images.size(0))
        for i in range(num_images):
            rgb_np = fixed_val_rgb[i].numpy()
            
            img_np = fixed_val_images[i].cpu().permute(1, 2, 0).numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
                
            rgb_fg = img_np[:, :, :3]
            alpha = img_np[:, :, 3:4] / 255.0 
            
            bg_color = np.full_like(rgb_fg, 128)
            img_display = (rgb_fg * alpha + bg_color * (1 - alpha)).astype(np.uint8)
            
            true_mask_np = fixed_val_masks[i].cpu().numpy().astype(np.uint8)
            pred_mask_np = fixed_preds[i].cpu().numpy().astype(np.uint8)
            
            # Create colored masks
            true_mask_colored = np.zeros((true_mask_np.shape[0], true_mask_np.shape[1], 3), dtype=np.uint8)
            pred_mask_colored = np.zeros((pred_mask_np.shape[0], pred_mask_np.shape[1], 3), dtype=np.uint8)
            
            true_mask_colored[true_mask_np == 1] = [0, 255, 255]
            true_mask_colored[true_mask_np == 2] = [255, 0, 255]
            pred_mask_colored[pred_mask_np == 1] = [0, 255, 255]
            pred_mask_colored[pred_mask_np == 2] = [255, 0, 255]
            
            wandb_log_images.append(wandb.Image(rgb_np, caption=f"Val Input #{i+1}"))
            wandb_log_images.append(wandb.Image(true_mask_colored, caption=f"Val GT #{i+1}"))
            wandb_log_images.append(wandb.Image(pred_mask_colored, caption=f"Val Pred #{i+1}"))

    avg_val_loss = running_val_loss / len(val_loader)
    all_val_preds = torch.cat(all_val_preds, dim=0)
    all_val_masks = torch.cat(all_val_masks, dim=0)
    
    # Calculate metrics
    val_metrics = calculate_metrics_torchmetrics(all_val_preds, all_val_masks, num_classes=num_classes)
    val_macro_f1 = val_metrics['macro_f1']
    val_mean_iou = val_metrics['mean_iou']
    val_iou_bg = val_metrics['iou_background']
    val_iou_fill = val_metrics['iou_fill']
    val_iou_satin = val_metrics['iou_satin']

    return {
        'loss': avg_val_loss,
        'macro_f1': val_macro_f1,
        'mean_iou': val_mean_iou,
        'iou_bg': val_iou_bg,
        'iou_fill': val_iou_fill,
        'iou_satin': val_iou_satin,
        'wandb_images': wandb_log_images
    }


def train_one_fold(fold_idx, train_indices, val_indices, svg_files, train_transform, val_transform, config, device, 
                   checkpoint_dir, wandb_run):
    """Train one fold and return best validation metrics from best epoch."""
    
    # Create separate datasets for train and val with different transforms
    train_dataset = EmbroideryDatasetSVG(
        svg_dir_or_paths=[svg_files[i] for i in train_indices],
        transform=train_transform,
        crops_per_image=config.crops,
        augment_color=True,
        target_size=config.image_size,
        supersample_factor=config.supersample_factor
    )
    
    val_dataset = EmbroideryDatasetSVG(
        svg_dir_or_paths=[svg_files[i] for i in val_indices],
        transform=val_transform,
        crops_per_image=config.crops,
        augment_color=False,
        target_size=config.image_size,
        supersample_factor=config.supersample_factor
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                             num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                           num_workers=4, persistent_workers=True)
    
    # Create tracking dataset for visualization (use val dataset)
    tracking_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{config.n_folds}")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Get fixed batch for visualization
    fixed_val_batch = next(iter(tracking_loader))
    
    # Initialize model
    model = U2NET(in_ch=4, out_ch=config.num_classes).to(device)
    class_weights = torch.tensor([1.0, config.fill_weight, config.satin_weight]).to(device)
    
    focal_loss_fn = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0)
    dice_loss_fn = GeneralizedDiceLoss(num_classes=config.num_classes)
    bce_boundary_fn = nn.BCEWithLogitsLoss()
    
    deep_supervision_weights = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1]
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scaler = GradScaler(device.type)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    best_val_f1 = 0.0
    best_epoch_metrics = None
    best_model_state = None
    epochs_no_improve = 0
    EARLY_STOPPING_PATIENCE = 100
    
    for epoch in range(config.epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            focal_loss_fn, dice_loss_fn, bce_boundary_fn,
            deep_supervision_weights, config.num_classes, epoch, config
        )
        
        # Validate
        val_metrics = validate_one_epoch(
            model, val_loader, device,
            focal_loss_fn, dice_loss_fn, bce_boundary_fn,
            deep_supervision_weights, config.num_classes, fixed_val_batch
        )
        
        scheduler.step()
        
        # Print logs
        print(f"\n[Epoch {epoch+1}] Fold {fold_idx+1}:")
        print(f"   Train | Loss: {train_metrics['loss']:.4f} | Macro F1: {train_metrics['macro_f1']:.4f} | Mean IoU: {train_metrics['mean_iou']:.4f}")
        print(f"          IoU - BG: {train_metrics['iou_bg']:.4f} | Fill: {train_metrics['iou_fill']:.4f} | Satin: {train_metrics['iou_satin']:.4f}")
        print(f"   Val   | Loss: {val_metrics['loss']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f} | Mean IoU: {val_metrics['mean_iou']:.4f}")
        print(f"          IoU - BG: {val_metrics['iou_bg']:.4f} | Fill: {val_metrics['iou_fill']:.4f} | Satin: {val_metrics['iou_satin']:.4f}")
        
        # Log to wandb with fold field
        wandb_run.log({
            "fold": fold_idx + 1,
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "Loss/Train": train_metrics['loss'],
            "Loss/Val": val_metrics['loss'],
            "F1_Macro/Train": train_metrics['macro_f1'],
            "F1_Macro/Val": val_metrics['macro_f1'],
            "IoU_Mean/Train": train_metrics['mean_iou'],
            "IoU_Mean/Val": val_metrics['mean_iou'],
            "IoU_Background/Train": train_metrics['iou_bg'],
            "IoU_Background/Val": val_metrics['iou_bg'],
            "IoU_Fill/Train": train_metrics['iou_fill'],
            "IoU_Fill/Val": val_metrics['iou_fill'],
            "IoU_Satin/Train": train_metrics['iou_satin'],
            "IoU_Satin/Val": val_metrics['iou_satin'],
            "Train_Images": train_metrics['wandb_images'],
            "Validation_Images": val_metrics['wandb_images']
        })
        
        
        # Save best model and metrics
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch_metrics = val_metrics.copy()
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"   *** NEW BEST MODEL: Val Macro F1 = {best_val_f1:.4f} ***")
        else:
            epochs_no_improve += 1
            print(f"   F1 no improve: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
        
        # Early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}!")
            print(f"Best Val Macro F1: {best_val_f1:.4f}")
            break
    
    # Return metrics from best epoch and best model state
    return best_epoch_metrics if best_epoch_metrics else val_metrics, best_model_state


def main():
    seed_everything(42)
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    class Config:
        image_size = 768
        crops = 1
        batch_size = 2
        num_classes = 3
        epochs = 50
        learning_rate = 1e-4
        fill_weight = 2
        satin_weight = 5
        supersample_factor = 2
        n_folds = 5
        source_dirs = ["data/logo/easy", "data/logo/medium", "data/logo/hard"]
    
    config = Config()
    
    # ==========================================
    # TRANSFORMS
    # ==========================================
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=config.image_size),
        A.PadIfNeeded(
            min_height=config.image_size,
            min_width=config.image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.85, 1.15),
            rotate=(-30, 30),
            interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.7
        ),
        A.CoarseDropout(
            num_holes_range=(4, 8), hole_height_range=(10, 30),
            hole_width_range=(10, 30), fill=0, p=0.3
        ),
        A.GaussNoise(std_range=(0.01, 0.02), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.LongestMaxSize(max_size=config.image_size),
        A.PadIfNeeded(
            min_height=config.image_size, min_width=config.image_size,
            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0
        ),
        ToTensorV2()
    ])
    
    # ==========================================
    # LOAD ALL SVG FILES AND CALCULATE SATIN RATIOS
    # ==========================================
    all_svg_files = []
    for source_dir in config.source_dirs:
        source_path = os.path.join(PROJECT_ROOT, source_dir)
        if os.path.exists(source_path):
            svg_files = list(Path(source_path).rglob("*.svg"))
            all_svg_files.extend(svg_files)
            print(f"Found {len(svg_files)} SVG files in {source_dir}")
    
    print(f"\nTotal SVG files: {len(all_svg_files)}")
    
    if len(all_svg_files) == 0:
        print("No SVG files found!")
        return
    
    # Calculate satin ratios for stratification
    print("\nCalculating satin ratios for stratification...")
    satin_ratios = []
    for svg_file in all_svg_files:
        fill_count, satin_count = count_satin_fill_paths(svg_file)
        ratio = calculate_satin_ratio(fill_count, satin_count)
        satin_ratios.append(ratio)
    
    # Bucket satin ratios for StratifiedKFold
    n_buckets = 4
    satin_ratio_buckets = [bucket_satin_ratio(r, n_buckets) for r in satin_ratios]
    
    print(f"Satin ratio distribution:")
    for i in range(n_buckets):
        count = satin_ratio_buckets.count(i)
        print(f"  Bucket {i}: {count} files ({count/len(all_svg_files)*100:.1f}%)")
    
    # ==========================================
    # 5-FOLD CROSS VALIDATION WITH STRATIFICATION
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"\nUsing device: {device}")
    
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints/logo")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create StratifiedKFold splitter based on satin ratio buckets
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
    
    # ==========================================
    # INITIALIZE SINGLE WANDB RUN
    # ==========================================
    wandb.init(
        project="embroidery-segmentation",
        name="logo-5fold-stratified",
        config={
            "learning_rate": config.learning_rate,
            "architecture": "U2-Net",
            "dataset": "Logo_3Class_5Fold_Stratified",
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "num_classes": config.num_classes,
            "input_channels": 4,
            "fill_weight": config.fill_weight,
            "satin_weight": config.satin_weight,
            "supersample_factor": config.supersample_factor,
            "n_folds": config.n_folds,
            "stratification": "satin_ratio_buckets"
        }
    )
    
    # Store results for each fold
    fold_results = []
    best_overall_f1 = 0.0
    best_overall_model_state = None
    
    # Train each fold
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(range(len(all_svg_files)), satin_ratio_buckets)):
        print(f"\n{'='*60}")
        print(f"Starting Fold {fold_idx + 1}/{config.n_folds}")
        print(f"{'='*60}")
        
        # Train this fold
        fold_result, fold_model_state = train_one_fold(
            fold_idx, train_indices, val_indices, all_svg_files, train_transform, val_transform, config, device,
            checkpoint_dir, wandb.run
        )
        
        fold_results.append(fold_result)
        print(f"\nFold {fold_idx+1} completed. Best Val Macro F1: {fold_result['macro_f1']:.4f}")
        
        # Track best overall model
        if fold_result['macro_f1'] > best_overall_f1:
            best_overall_f1 = fold_result['macro_f1']
            best_overall_model_state = fold_model_state
    
    # ==========================================
    # CALCULATE FINAL RESULTS
    # ==========================================
    print(f"\n{'='*60}")
    print(f"5-FOLD CROSS VALIDATION RESULTS")
    print(f"{'='*60}")
    
    # Calculate mean and std for each metric
    macro_f1s = [r['macro_f1'] for r in fold_results]
    mean_ious = [r['mean_iou'] for r in fold_results]
    iou_bgs = [r['iou_bg'] for r in fold_results]
    iou_fills = [r['iou_fill'] for r in fold_results]
    iou_satins = [r['iou_satin'] for r in fold_results]
    
    print(f"\nMacro F1:")
    print(f"  Mean ± Std: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    
    print(f"\nMean IoU:")
    print(f"  Mean ± Std: {np.mean(mean_ious):.4f} ± {np.std(mean_ious):.4f}")
    
    print(f"\nBackground IoU:")
    print(f"  Mean ± Std: {np.mean(iou_bgs):.4f} ± {np.std(iou_bgs):.4f}")
    
    print(f"\nFill IoU:")
    print(f"  Mean ± Std: {np.mean(iou_fills):.4f} ± {np.std(iou_fills):.4f}")
    
    print(f"\nSatin IoU:")
    print(f"  Mean ± Std: {np.mean(iou_satins):.4f} ± {np.std(iou_satins):.4f}")
    
    # ==========================================
    # SAVE BEST OVERALL MODEL
    # ==========================================
    best_overall_path = os.path.join(checkpoint_dir, "best_overall.pth")
    
    if best_overall_model_state is not None:
        torch.save(best_overall_model_state, best_overall_path)
        wandb.save(best_overall_path)
        print(f"\nBest overall model saved to: {best_overall_path}")
    else:
        print("\nWarning: No best model state found. Skipping save.")
    
    wandb.finish()


if __name__ == "__main__":
    main()
