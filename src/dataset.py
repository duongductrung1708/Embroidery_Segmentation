import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class EmbroideryDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, resize_factor=0.5, crops_per_image=20):
        self.image_paths = sorted(glob.glob(f"{image_dir}/*.png"))
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.transform = transform
        self.resize_factor = resize_factor
        
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

        # 2. Thu nhỏ ảnh động (On-the-fly Resize)
        if self.resize_factor != 1.0:
            new_w = int(img_rgba.shape[1] * self.resize_factor)
            new_h = int(img_rgba.shape[0] * self.resize_factor)
            img_rgba = cv2.resize(img_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Mask bắt buộc dùng INTER_NEAREST để viền không bị mờ nhòe (0 hoặc 255)
            mask_gray = cv2.resize(mask_gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Ép nhãn Mask về 3 lớp: Background (0), Fill (1), Satin (2)
        # Giữ nguyên giá trị gốc từ mask (nếu đã có 3 lớp)
        # Nếu mask chỉ có 2 lớp (0 và 255), chuyển về 0 và 1
        unique_vals = np.unique(mask_gray)
        if len(unique_vals) == 2 and 255 in unique_vals:
            # Binary mask: chuyển 255 -> 1
            mask_binary = (mask_gray > 127).astype(np.float32)
        else:
            # Multi-class mask: giữ nguyên giá trị (0, 1, 2)
            mask_binary = mask_gray.astype(np.float32)

        # Trích xuất kênh Alpha (Nét vẽ gốc) làm Input cho AI
        alpha_channel = img_rgba[:, :, 3].astype(np.float32)

        # 3. Kỹ thuật Padding
        # Đề phòng trường hợp ảnh gốc sau khi thu nhỏ lại bé hơn 512x512
        h, w = alpha_channel.shape
        if h < 512 or w < 512:
            pad_h = max(0, 512 - h)
            pad_w = max(0, 512 - w)
            # Lấp đầy phần thiếu bằng số 0 (Màu đen / Trong suốt)
            alpha_channel = np.pad(alpha_channel, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            mask_binary = np.pad(mask_binary, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        # 4. Đẩy qua Transform (Albumentations sẽ RandomCrop 512x512 ở bước này)
        if self.transform is not None:
            augmented = self.transform(image=alpha_channel, mask=mask_binary)
            image_tensor = augmented['image'] / 255.0 # Chuẩn hóa về [0, 1]
            mask_tensor = augmented['mask'].long()    # Bắt buộc là LongTensor cho CrossEntropy
        else:
            # Code an toàn nếu quên truyền Transform
            image_tensor = torch.tensor(alpha_channel / 255.0).unsqueeze(0)
            mask_tensor = torch.tensor(mask_binary).long()

        return image_tensor, mask_tensor
