import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Giả sử bạn đã có 2 file này trong thư mục src
from src.dataset import EmbroideryDataset
from src.model import UNet

# ==========================================
# HÀM TÍNH TOÁN METRICS
# ==========================================
def calculate_metrics(tp, fp, fn, tn):
    epsilon = 1e-7 # Chống chia cho 0
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return accuracy, precision, recall, f1

# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    # 1. Khởi tạo Wandb và Cấu hình
    wandb.init(
        project="embroidery-segmentation", 
        name="v2-weighted-loss-scheduler-test",         
        config={                           
            "learning_rate": 1e-4,
            "architecture": "U-Net",
            "dataset": "Embroidery_V2",
            "epochs": 30, 
            "batch_size": 4,
            "image_size": 512,
            "fill_weight": 5.0 # Trọng số phạt nếu đoán sai class Fill
        }
    )
    config = wandb.config 

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Đang sử dụng thiết bị tính toán: {device}")

    # 2. Khai báo 3 tập dữ liệu riêng biệt (Tránh Data Leakage)
    train_dataset = EmbroideryDataset(image_dir="data/train/images", mask_dir="data/train/masks")
    val_dataset = EmbroideryDataset(image_dir="data/val/images", mask_dir="data/val/masks")
    test_dataset = EmbroideryDataset(image_dir="data/test/images", mask_dir="data/test/masks")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, persistent_workers=True) 
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, persistent_workers=True) 

    # 3. Khởi tạo Model, Loss, Optimizer và Scheduler
    model = UNet(in_channels=1, out_channels=2).to(device)
    
    # Loss có trọng số phạt
    class_weights = torch.tensor([1.0, config.fill_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Scheduler không có tham số verbose (tránh lỗi PyTorch mới)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_f1 = 0.0

    print("\nBẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN...")
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
            loss = criterion(outputs, masks)
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
                val_loss = criterion(val_outputs, val_masks)
                running_val_loss += val_loss.item()

                preds = torch.argmax(val_outputs, dim=1)
                val_tp += ((preds == 1) & (val_masks == 1)).sum().item()
                val_fp += ((preds == 1) & (val_masks == 0)).sum().item()
                val_fn += ((preds == 0) & (val_masks == 1)).sum().item()
                val_tn += ((preds == 0) & (val_masks == 0)).sum().item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_acc, val_prec, val_recall, val_f1 = calculate_metrics(val_tp, val_fp, val_fn, val_tn)

        # Cập nhật Scheduler và in ra cảnh báo nếu LR bị giảm
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"\nCảnh báo: Val Loss đi ngang, Learning Rate giảm xuống: {new_lr}")

        # ------------------------------------------
        # BÁO CÁO VÀ GHI LOG WANDB
        # ------------------------------------------
        print(f"\n[Epoch {epoch+1}] Báo cáo:")
        print(f"   Train | Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(f"   Val   | Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}\n")

        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "Loss/Train": avg_train_loss,
            "Loss/Val": avg_val_loss,
            "Accuracy/Train": train_acc,
            "Accuracy/Val": val_acc,
            "Recall/Train": train_recall,
            "Recall/Val": val_recall,
            "F1_Score/Train": train_f1,
            "F1_Score/Val": val_f1
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "unet_binary_best.pth")
            print(f"Đã lưu kỷ lục mới (Best Val F1: {best_val_f1:.4f})")
            wandb.save("unet_binary_best.pth")

    # ==========================================
    # PHA 3: THI ĐẠI HỌC (TEST TRÊN DỮ LIỆU MỚI TINH)
    # ==========================================
    print("\n==============================================")
    print("HOÀN THÀNH HUẤN LUYỆN! BƯỚC VÀO KỲ THI TEST...")
    print("==============================================")
    
    # Tải lại bộ não có phong độ cao nhất trong quá trình thi thử
    model.load_state_dict(torch.load("unet_binary_best.pth"))
    model.eval()
    
    test_tp = test_fp = test_fn = test_tn = 0 
    
    with torch.no_grad():
        for test_images, test_masks in tqdm(test_loader, desc="Đang chấm bài Test..."):
            test_images, test_masks = test_images.to(device), test_masks.to(device)
            
            test_outputs = model(test_images)
            preds = torch.argmax(test_outputs, dim=1)
            
            test_tp += ((preds == 1) & (test_masks == 1)).sum().item()
            test_fp += ((preds == 1) & (test_masks == 0)).sum().item()
            test_fn += ((preds == 0) & (test_masks == 1)).sum().item()
            test_tn += ((preds == 0) & (test_masks == 0)).sum().item()
            
    test_acc, test_prec, test_recall, test_f1 = calculate_metrics(test_tp, test_fp, test_fn, test_tn)

    print("\nKẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (UNSEEN DATA):")
    print(f"   Accuracy : {test_acc:.4f}")
    print(f"   Precision: {test_prec:.4f}")
    print(f"   Recall   : {test_recall:.4f}")
    print(f"   F1-Score : {test_f1:.4f}\n")
    
    wandb.log({
        "Test/Accuracy": test_acc,
        "Test/Precision": test_prec,
        "Test/Recall": test_recall,
        "Test/F1_Score": test_f1
    })
    
    wandb.finish() 

# -------------------------------------------------------------
# DÒNG NÀY LÀ CÁI NÚT BẤM KÍCH HOẠT TẤT CẢ - TUYỆT ĐỐI KHÔNG BỎ
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
