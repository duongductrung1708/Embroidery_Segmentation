import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class EmbroideryDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '.png')
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Ảnh xám

        if self.transform:
            image = self.transform(image)

        # CHUYỂN MÀU TRẮNG (255) THÀNH NHÃN 1 (VÙNG FILL)
        mask_np = np.array(mask)
        mask_np[mask_np > 0] = 1 
        mask_tensor = torch.from_numpy(mask_np).long()

        return image, mask_tensor