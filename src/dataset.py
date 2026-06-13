import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class EmbroideryDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Khởi tạo băng chuyền
        :param image_dir: Đường dẫn tới thư mục chứa ảnh RGB
        :param mask_dir: Đường dẫn tới thư mục chứa ảnh Mask PNG
        :param transform: Các hàm biến đổi ảnh (như ToTensor, Normalize)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Lấy danh sách toàn bộ tên file ảnh
        self.images = os.listdir(image_dir)

    def __len__(self):
        # Trả về tổng số lượng ảnh đang có
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Lấy tên file
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '.png') # Mask đuôi .png
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 2. Đọc ảnh bằng PIL
        image = Image.open(img_path).convert("RGB") # Ảnh màu
        mask = Image.open(mask_path).convert("L")   # Ảnh xám (Grayscale)

        # 3. Biến đổi ảnh Input (Ví dụ: Chuyển thành Tensor Float 0-1)
        if self.transform:
            image = self.transform(image)

        # 4. Ép kiểu cho Mask (Cực kỳ quan trọng: LongTensor 0, 1, 2)
        # Không dùng self.transform cho mask vì nó sẽ làm hỏng nhãn ID
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
