import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class EmbroideryDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # ======================
        # 1. XỬ LÝ INPUT (ẢNH GỐC VIỀN ĐEN, NỀN TRONG SUỐT)
        # ======================
        rgba_img = Image.open(img_path).convert("RGBA")
        
        # TRÍCH XUẤT KÊNH ALPHA: Bắt trọn vẹn nét vẽ!
        # Nét vẽ sẽ thành 255 (Sáng), Nền trong suốt sẽ thành 0 (Tối)
        alpha_channel = np.array(rgba_img.getchannel("A"), dtype=np.float32)
        
        # Thêm chiều (1, H, W) và đưa về khoảng [0, 1]
        image = torch.tensor(alpha_channel).unsqueeze(0) / 255.0

        # ======================
        # 2. XỬ LÝ TARGET (MASK ĐÃ TÔ TAY)
        # ======================
        mask_img = Image.open(mask_path).convert("L")
        mask = np.array(mask_img)
        
        # Đảm bảo Mask chỉ chứa giá trị 0 (Nền) và 1 (Vùng Fill)
        mask = (mask > 128).astype(np.uint8)
        mask = torch.tensor(mask).long()

        return image, mask