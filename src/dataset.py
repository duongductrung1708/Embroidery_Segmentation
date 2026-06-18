import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class EmbroideryDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform # MỚI: Nhận bộ công cụ nhào nặn ảnh
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # ======================
        # 1. ĐỌC ẢNH VÀ MASK BẰNG PILLOW -> CHUYỂN SANG NUMPY
        # ======================
        rgba_img = Image.open(img_path).convert("RGBA")
        
        # Albumentation thích làm việc với mảng Numpy (H, W)
        image = np.array(rgba_img.getchannel("A"), dtype=np.float32) / 255.0

        mask_img = Image.open(mask_path).convert("L")
        mask = np.array(mask_img)
        mask = (mask > 128).astype(np.float32) # [0.0, 1.0]

        # ======================
        # 2. ÁP DỤNG DATA AUGMENTATION (NẾU CÓ)
        # ======================
        if self.transform is not None:
            # Albumentation sẽ tự động lật/xoay đồng bộ cả Image lẫn Mask!
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        # ======================
        # 3. ÉP KIỂU CHUẨN CHO PYTORCH
        # ======================
        # Đảm bảo image có shape (1, H, W) cho U-Net
        if not torch.is_tensor(image):
            image = torch.tensor(image)
        if len(image.shape) == 2:
            image = image.unsqueeze(0) # (H, W) -> (1, H, W)
            
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask)
            
        # Mask BẮT BUỘC phải là số nguyên (LongTensor) để dùng hàm CrossEntropyLoss
        mask = mask.long()

        return image, mask