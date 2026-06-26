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
        # Multi-class Dice Loss
        probs = torch.softmax(inputs, dim=1)  # [Batch, Class, H, W]
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_one_hot).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
        
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
        # Multi-class mask: detect boundaries for non-background pixels (class > 0)
        mask_binary = (masks_np[i] > 0).astype(np.uint8)
        mask_255 = (mask_binary * 255).astype(np.uint8)
        edges = cv2.Canny(mask_255, 100, 200)
        boundaries[i] = (edges > 0).astype(np.float32)
    return torch.from_numpy(boundaries).to(device)
