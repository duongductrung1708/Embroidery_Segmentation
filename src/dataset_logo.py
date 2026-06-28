import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class EmbroideryDatasetLogo(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, crops_per_image=1):
        self.image_paths = sorted(glob.glob(f"{image_dir}/*.png"))
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.transform = transform
        
        # HACK CHÍ MẠNG: Nhân bản danh sách để 1 ảnh gốc được load nhiều lần trong 1 Epoch
        # Giúp hàm RandomCrop có cơ hội cắt được nhiều vị trí khác nhau trên cùng 1 bức ảnh to
        self.image_paths = self.image_paths * crops_per_image
        self.mask_paths = self.mask_paths * crops_per_image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 1. Đọc ảnh gốc (Giữ nguyên kênh Alpha trong suốt)
        img_rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if len(img_rgba.shape) == 2 or img_rgba.shape[2] == 3:
            img_rgba = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2BGRA)

        # Giữ nguyên 3-class mask (0=background, 1=fill, 2=satin)
        mask_binary = mask_gray.astype(np.float32)

        # Trích xuất kênh Alpha (Nét vẽ gốc) làm Input cho AI
        alpha_channel = img_rgba[:, :, 3].astype(np.float32) if img_rgba.shape[2] == 4 else img_rgba.mean(axis=2).astype(np.float32)

        # Đẩy qua Transform (Albumentations sẽ resize/pad ở bước này)
        if self.transform is not None:
            augmented = self.transform(image=alpha_channel, mask=mask_binary)
            image_tensor = augmented['image'] / 255.0 # Chuẩn hóa về [0, 1]
            mask_tensor = augmented['mask'].long()    # Bắt buộc là LongTensor cho CrossEntropy
        else:
            # Code an toàn nếu quên truyền Transform
            image_tensor = torch.tensor(alpha_channel / 255.0).unsqueeze(0)
            mask_tensor = torch.tensor(mask_binary).long()

        return image_tensor, mask_tensor
