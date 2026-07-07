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
    """Generalized Dice Loss với trọng số tính theo tần suất pixel của từng batch"""
    def __init__(self, num_classes=3, smooth=1e-5):
        super(GeneralizedDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [Batch, num_classes, H, W] - raw logits
            targets: [Batch, H, W] - class indices
        """
        probs = torch.softmax(inputs, dim=1)  # [Batch, num_classes, H, W]
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate class frequencies in the batch
        # Sum over spatial dimensions and batch
        class_counts = targets_one_hot.sum(dim=(0, 2, 3))  # [num_classes]
        
        # Calculate weights inversely proportional to frequency
        # w = 1 / (class_count + epsilon) to avoid division by zero
        epsilon = 1e-6
        weights = 1.0 / (class_counts + epsilon)
        
        # Normalize weights so they sum to num_classes (optional, keeps scale reasonable)
        weights = weights * self.num_classes / weights.sum()
        
        # Apply weights
        weights = weights.to(inputs.device)
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

    def forward(self, inputs, targets, spatial_weights=None):
        # inputs: [Batch, Class, H, W], targets: [Batch, H, W]
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, 
            reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        # --- NHÂN TRỌNG SỐ KHÔNG GIAN (SPATIAL BOUNDARY PENALTY) ---
        if spatial_weights is not None:
            focal_loss = focal_loss * spatial_weights
            
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
            
            # Dilate boundary để làm dày đường biên (Đã hạ xuống 3x3 để giữ chữ nét mảnh)
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            
            boundaries[i, class_idx] = (edges_dilated > 0).astype(np.float32)
    
    return torch.from_numpy(boundaries).to(device)

def compute_spatial_weight_map(masks, base_weight=1.0, edge_weight=5.0,
                                gap_weight=10.0, gap_radius=5,
                                thin_weight=8.0, thin_radius=3):
    """
    Bản đồ trọng số không gian, phạt nặng hơn ở 3 loại vùng khó, ưu tiên
    theo thứ tự MẠNH NHẤT trước (lấy max nếu 1 pixel rơi vào nhiều loại):

    1. gap_weight -- NỀN (background) nằm trong bán kính gap_radius tính từ
       Satin gần nhất. Xử lý đúng trường hợp "2 nét satin sát nhau": nền ở
       khe hẹp luôn nằm trong bán kính này từ CẢ HAI phía nét satin, nên bị
       phạt nặng nhất trong 3 loại -- ép mô hình phải giữ đúng dải nền mỏng
       đó thay vì "lấp" nó thành satin.

    2. thin_weight -- SATIN thuộc phần "lõi mỏng": pixel satin biến mất khi
       ăn mòn (erode) bán kính thin_radius, tức bề dày thực tế của nét tại
       đó nhỏ hơn 2*thin_radius. Xử lý trường hợp "nét satin quá mảnh" và
       "lỗ nhỏ bên trong chữ bold" (viền quanh lỗ cũng là 1 dạng nét mỏng).

    3. edge_weight -- biên giữa 2 class bất kỳ nói chung (như bản gốc,
       giữ lại để không mất tác dụng chống tràn viền ở các vùng biên bình
       thường, không thuộc 2 loại đặc biệt trên).

    QUAN TRỌNG: đây là loss weighting, không phải chỉnh kiến trúc mạng --
    giúp mô hình "cố gắng hơn" ở đúng vùng khó, nhưng KHÔNG đảm bảo giải
    quyết triệt để nếu bản thân downsampling của kiến trúc đã xoá mất
    thông tin không gian trước khi loss kịp tác động (net quá mảnh so với
    độ phân giải sau pooling). Nếu tăng gap_weight/thin_weight mà vẫn
    không cải thiện, nhiều khả năng cần tăng độ phân giải ảnh đầu vào
    hoặc giảm số tầng downsampling, không phải chỉnh tiếp trọng số.
    """
    device = masks.device
    masks_np = masks.cpu().numpy().astype(np.uint8)
    batch_size, h, w = masks_np.shape

    weight_map = np.full((batch_size, h, w), base_weight, dtype=np.float32)

    edge_kernel = np.ones((5, 5), np.uint8)
    thin_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * thin_radius + 1, 2 * thin_radius + 1)
    )

    for i in range(batch_size):
        m = masks_np[i]

        # --- 1. Biên chung (giữ như bản gốc) ---
        grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, edge_kernel)
        edge_pixels = grad > 0
        weight_map[i][edge_pixels] = np.maximum(weight_map[i][edge_pixels], edge_weight)

        satin_mask = (m == 2).astype(np.uint8)
        if satin_mask.any():
            # --- 2. Nền kẹp giữa 2 vùng satin (khe hẹp) ---
            # distanceTransform(1 - satin_mask): với mỗi pixel KHÔNG phải
            # satin, cho khoảng cách tới satin gần nhất. Nền ở khe hẹp luôn
            # có khoảng cách nhỏ tới satin ở cả 2 phía -> chắc chắn lọt vào
            # gap_radius, dù satin 2 bên có thuộc 2 blob tách biệt hay không.
            dist_to_satin = cv2.distanceTransform(1 - satin_mask, cv2.DIST_L2, 5)
            gap_pixels = (m == 0) & (dist_to_satin <= gap_radius)
            weight_map[i][gap_pixels] = np.maximum(weight_map[i][gap_pixels], gap_weight)

            # --- 3. Satin thuộc phần lõi mỏng (nét mảnh / viền lỗ nhỏ) ---
            eroded_satin = cv2.erode(satin_mask, thin_kernel, iterations=1)
            thin_pixels = (satin_mask == 1) & (eroded_satin == 0)
            weight_map[i][thin_pixels] = np.maximum(weight_map[i][thin_pixels], thin_weight)

    return torch.from_numpy(weight_map).to(device)