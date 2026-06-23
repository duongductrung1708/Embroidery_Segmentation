import torch
import torch.nn as nn
import random
import numpy as np
import os
import cv2

# ==========================================
# 1. CỐ ĐỊNH MÔI TRƯỜNG (REPRODUCIBILITY)
# ==========================================
def seed_everything(seed=42):
    """Giúp cố định các yếu tố ngẫu nhiên để kết quả train luôn giống nhau ở mọi lần chạy"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. HÀM TÍNH TOÁN METRICS
# ==========================================
def calculate_metrics(tp, fp, fn, tn):
    """Tính toán 4 chỉ số quan trọng nhất của phân loại điểm ảnh"""
    epsilon = 1e-7 
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return accuracy, precision, recall, f1

# ==========================================
# 3. CÁC HÀM LOSS NÂNG CAO (TRỊ TRÀN VIỀN & MŨI MẮT)
# ==========================================
class DiceLoss(nn.Module):
    """Trừng phạt AI khi vẽ sai hình dáng tổng thể của mảng thêu"""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)[:, 1] 
        targets_float = targets.float()
        
        intersection = (probs * targets_float).sum(dim=(1,2))
        union = probs.sum(dim=(1,2)) + targets_float.sum(dim=(1,2))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    """Trị lỗi Model bỏ qua các chi tiết khó (như biên giới, vùng nhỏ)"""
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # inputs: [Batch, Class, H, W], targets: [Batch, H, W]
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, 
            reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean()

def get_boundary_mask(masks, device):
    """Trích xuất đường biên bằng thuật toán Canny để ép Model học ranh giới"""
    masks_np = masks.cpu().numpy().astype(np.uint8)
    boundaries = np.zeros_like(masks_np, dtype=np.float32)
    for i in range(masks_np.shape[0]):
        # Chuyển mask (0, 1) sang (0, 255) cho hàm cv2.Canny
        mask_255 = (masks_np[i] * 255).astype(np.uint8)
        edges = cv2.Canny(mask_255, 100, 200)
        boundaries[i] = (edges > 0).astype(np.float32)
    return torch.from_numpy(boundaries).to(device)
