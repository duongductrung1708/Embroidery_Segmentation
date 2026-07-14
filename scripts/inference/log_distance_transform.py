import cv2
import numpy as np
import matplotlib.pyplot as plt

def debug_shape_thickness_fixed(image_path, physical_width_mm=80.0):
    # 1. Đọc ảnh và tìm ngưỡng
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Không tìm thấy ảnh: {image_path}")
        return
        
    h, w = img.shape[:2]
    pixel_to_mm = physical_width_mm / max(float(w), 1.0)
    
    # 2. XỬ LÝ MÀU SẮC (SỬA LỖI Ở ĐÂY)
    # Dùng THRESH_OTSU để tự động phân mảng, kết hợp THRESH_BINARY_INV để chữ thành Trắng, nền thành Đen
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Kiểm tra an toàn: Nếu 4 góc của ảnh vẫn là màu Trắng -> nền chưa bị đảo đúng -> Đảo lại lần nữa
    corners = [binary[0,0], binary[0,-1], binary[-1,0], binary[-1,-1]]
    if sum(corners) > 255 * 2: 
        binary = cv2.bitwise_not(binary)
        
    # 3. Tìm contours và bóc tách từng hình khối
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        print("Không tìm thấy hình khối nào trong ảnh.")
        return
    hierarchy = hierarchy[0]
    
    canvas_absolute = np.zeros((h, w), dtype=np.float32)
    canvas_relative = np.zeros((h, w), dtype=np.float32)
    
    shape_annotations = []
    
    # Phân tích từng chữ cái độc lập
    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1: continue # Chỉ lấy viền ngoài cùng
        if cv2.contourArea(contour) < 4: continue # Bỏ qua rác
            
        holes = [contours[j] for j, hj in enumerate(hierarchy) if hj[3] == i]
        
        # Tạo mask Trắng cho 1 chữ cái duy nhất
        shape_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(shape_mask, [contour], -1, 255, thickness=cv2.FILLED)
        for hole in holes:
            cv2.drawContours(shape_mask, [hole], -1, 0, thickness=cv2.FILLED) # Đục lỗ chữ 8, A...
            
        # ĐO ĐẠC BÊN TRONG NÉT CHỮ
        dist = cv2.distanceTransform(shape_mask, cv2.DIST_L2, 5)
        max_dist_px = np.max(dist)
        
        if max_dist_px == 0: continue
            
        canvas_absolute = np.maximum(canvas_absolute, dist)
        
        # Chuẩn hóa dải màu cho từng chữ
        dist_norm_per_shape = dist / max_dist_px
        canvas_relative = np.maximum(canvas_relative, dist_norm_per_shape)
        
        # Tìm tọa độ điểm dày nhất của chữ đó
        _, _, _, max_loc = cv2.minMaxLoc(dist)
        max_thickness_mm = (max_dist_px * 2.0) * pixel_to_mm
        
        shape_annotations.append({'loc': max_loc, 'mm': max_thickness_mm})

    # 4. Hiển thị
    # Che đi phần nền ĐEN 0 để không bị đổ màu xanh lè
    masked_abs = np.ma.masked_where(canvas_absolute == 0, canvas_absolute)
    masked_rel = np.ma.masked_where(canvas_relative == 0, canvas_relative)
    
    plt.figure(figsize=(18, 8))
    
    # --- Cột 1: Mask ảnh (Đã sửa lại thành chữ trắng nền đen) ---
    plt.subplot(1, 3, 1)
    plt.title("1. Mask Chữ (Đã đảo nền Đen, chữ Trắng)")
    plt.imshow(binary, cmap='gray')
    plt.axis('off')
    
    # --- Cột 2: Heatmap Tuyệt đối ---
    plt.subplot(1, 3, 2)
    plt.title("2. Heatmap Tuyệt đối\n(Bên trong lòng chữ)")
    im1 = plt.imshow(masked_abs, cmap='jet')
    plt.colorbar(im1, fraction=0.046, pad=0.04, label='Độ dày (Pixels)')
    plt.axis('off')
    
    # --- Cột 3: Heatmap Tương đối (Quan trọng) ---
    plt.subplot(1, 3, 3)
    plt.title("3. Heatmap Tương đối\n(Phân tích lõi của từng Shape)")
    
    # Vẽ nền xám mờ chữ gốc để dễ hình dung
    plt.imshow(binary, cmap='gray', alpha=0.2)
    # Vẽ heatmap đè lên
    im2 = plt.imshow(masked_rel, cmap='jet')
    plt.colorbar(im2, fraction=0.046, pad=0.04, label='Tỷ lệ (0.0 -> 1.0)')
    
    # In thông số
    for ann in shape_annotations:
        x, y = ann['loc']
        plt.plot(x, y, 'rx', markersize=6) 
        plt.text(x + 5, y, f"{ann['mm']:.2f}mm", color='white', 
                 fontsize=9, weight='bold', bbox=dict(facecolor='black', alpha=0.6, pad=1))
                 
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Hãy thay đường dẫn ảnh của bạn vào đây
debug_shape_thickness_fixed('data/opencv_test/133.png', physical_width_mm=80.0)