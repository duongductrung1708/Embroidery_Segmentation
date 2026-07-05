import cv2
import numpy as np
from src.dataset_svg import EmbroideryDatasetSVG

# 1. Khởi tạo Dataset và load ảnh (với supersample_factor=2 để phóng to)
ds = EmbroideryDatasetSVG(
    svg_dir_or_paths=["data/svg/logo/23.svg"], 
    transform=None, 
    target_size=768, 
    supersample_factor=2
)

# dataset_svg trả về: image_tensor, mask_tensor, rgb_image
_, mask_tensor, rgb = ds[0]

# 2. Chuyển tensor về numpy array để xử lý bằng OpenCV
mask = mask_tensor.numpy().astype(np.uint8)

# 3. Tạo một bức ảnh trống để tô màu
colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

# Tô màu Cyan cho Fill (Nhãn 1) và Magenta cho Satin (Nhãn 2)
colored_mask[mask == 1] = [0, 255, 255]
colored_mask[mask == 2] = [255, 0, 255]

# 4. Lưu ra file để dễ dàng zoom bằng trình xem ảnh
cv2.imwrite("inspect_110_mask.png", cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
cv2.imwrite("inspect_110_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

print("✅ Đã lưu thành công 2 file: inspect_110_mask.png và inspect_110_rgb.png")
print("Hãy mở file mask lên và zoom thật to vào các khe hẹp để kiểm tra!")
