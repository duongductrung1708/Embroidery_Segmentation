import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import cv2
from torchmetrics.functional import f1_score, jaccard_index

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
# 2. HÀM TÍNH TOÁN METRICS (Torchmetrics-based)
# ==========================================
def calculate_metrics(tp, fp, fn, tn):
    """Tính toán 4 chỉ số quan trọng nhất của phân loại điểm ảnh (legacy)"""
    epsilon = 1e-7 
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return accuracy, precision, recall, f1

def calculate_metrics_torchmetrics(preds, targets, num_classes=3):
    """
    Tính toán metrics sử dụng torchmetrics cho multi-class segmentation
    Args:
        preds: [Batch, H, W] - predicted class indices
        targets: [Batch, H, W] - ground truth class indices
        num_classes: số lượng lớp (3 cho Background, Fill, Satin)
    Returns:
        dict: chứa macro_f1, per_class_iou, mean_iou
    """
    device = preds.device
    
    # Macro F1 Score
    macro_f1 = f1_score(
        preds, 
        targets, 
        task='multiclass', 
        num_classes=num_classes, 
        average='macro'
    ).item()
    
    # Per-class IoU (Jaccard Index)
    per_class_iou = jaccard_index(
        preds,
        targets,
        task='multiclass',
        num_classes=num_classes,
        average=None
    )  # Returns tensor of shape [num_classes]
    
    per_class_iou_list = per_class_iou.cpu().tolist()
    mean_iou = per_class_iou.mean().item()
    
    return {
        'macro_f1': macro_f1,
        'iou_background': per_class_iou_list[0],
        'iou_fill': per_class_iou_list[1],
        'iou_satin': per_class_iou_list[2] if num_classes == 3 else None,
        'mean_iou': mean_iou
    }

# ==========================================
# 3. CÁC HÀM LOSS NÂNG CAO (TRỊ TRÀN VIỀN & MŨI MẮT)
# ==========================================
class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss với trọng số cho từng lớp (hỗ trợ multi-class)"""
    def __init__(self, num_classes=3, smooth=1e-5, weights=None):
        super(GeneralizedDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
        if weights is None:
            # Mặc định: trọng số cao hơn cho các lớp thiểu số (Fill, Satin)
            self.weights = torch.tensor([1.0, 2.0, 2.0]) if num_classes == 3 else torch.tensor([1.0, 2.0])
        else:
            self.weights = torch.tensor(weights)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [Batch, num_classes, H, W] - raw logits
            targets: [Batch, H, W] - class indices
        """
        probs = torch.softmax(inputs, dim=1)  # [Batch, num_classes, H, W]
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Apply weights
        weights = self.weights.to(inputs.device)
        weights = weights.view(1, self.num_classes, 1, 1)
        
        # Calculate weighted intersection and union
        intersection = (probs * targets_one_hot * weights).sum(dim=(2, 3))
        union = (probs * weights).sum(dim=(2, 3)) + (targets_one_hot * weights).sum(dim=(2, 3))
        
        # Generalized Dice
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    """Trị lỗi Model bỏ qua các chi tiết khó (như biên giới, vùng nhỏ)"""
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.02):
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

def get_boundary_mask(masks, device, num_classes=3):
    """
    Trích xuất đường biên cho multi-class segmentation bằng thuật toán Canny
    Args:
        masks: [Batch, H, W] - class indices
        device: torch device
        num_classes: số lượng lớp (3 cho Background, Fill, Satin)
    Returns:
        boundaries: [Batch, num_classes, H, W] - boundary masks cho từng lớp
    """
    masks_np = masks.cpu().numpy().astype(np.uint8)
    batch_size = masks_np.shape[0]
    h, w = masks_np.shape[1], masks_np.shape[2]
    boundaries = np.zeros((batch_size, num_classes, h, w), dtype=np.float32)
    
    for i in range(batch_size):
        for class_idx in range(num_classes):
            # Tạo binary mask cho lớp hiện tại
            class_mask = (masks_np[i] == class_idx).astype(np.uint8)
            
            # Chuyển sang (0, 255) cho Canny
            mask_255 = (class_mask * 255).astype(np.uint8)
            edges = cv2.Canny(mask_255, 100, 200)
            boundaries[i, class_idx] = (edges > 0).astype(np.float32)
    
    return torch.from_numpy(boundaries).to(device)
