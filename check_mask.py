import cv2
import matplotlib.pyplot as plt

def check_overlay(img_path, mask_path):
    # 1. Đọc ảnh gốc (màu) và Mask (đen trắng)
    img = cv2.imread(img_path) 
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Tạo một lớp màu đỏ để chồng lên phần Fill (Mắt, Mũi)
    # Mask màu trắng (255) -> Màu đỏ (R=255, G=0, B=0)
    overlay = img.copy()
    overlay[mask == 255] = [0, 0, 255] # Gán màu đỏ cho pixel được chọn
    
    # 3. Trộn ảnh gốc và ảnh đỏ (độ trong suốt 0.5)
    result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    # 4. Hiển thị
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Kiểm tra Mask: Vùng Đỏ là phần bạn đã tô")
    plt.axis('off')
    plt.show()

# Thay tên file vào đây
check_overlay('./data/raw/images/6.png', './data/raw/masks/6.png')