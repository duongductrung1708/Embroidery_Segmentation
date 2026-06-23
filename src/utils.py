import torch
import torch.nn as nn
import random
import numpy as np
import os

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
# 3. HÀM TÍNH LOSS KÉP (DICE LOSS)
# ==========================================
class DiceLoss(nn.Module):
    """Trừng phạt AI khi vẽ sai hình dáng của mảng thêu"""
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
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: [Batch, Class, H, W], targets: [Batch, H, W]
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()