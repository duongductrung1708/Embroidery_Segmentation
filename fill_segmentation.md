# Pipeline Giai đoạn 1 (V3 - Production): Binary Fill Segmentation & Satin Boundary Extraction

**Mục tiêu:** Xây dựng hệ thống Trí tuệ Nhân tạo có khả năng nhận thức sắc nét vùng mảng thêu (Fill) từ ảnh phác thảo, đồng thời tích hợp Toán thái học Hình học (Computational Geometry) để trích xuất trực tiếp các đường viền Vector Satin siêu mảnh, sẵn sàng cho việc sinh mã máy thêu.

---

## Khâu 1: Chuẩn bị Dữ liệu Vàng & Băm ảnh (`slice_image.py`)

Loại bỏ dữ liệu rác, tập trung vào số lượng nhỏ các "Mẫu Vàng" (Golden Samples) tự tô mask, và tối ưu hóa không gian ngữ nghĩa cho AI.

1. **Khởi tạo Dữ liệu Vàng:** Sử dụng các ảnh gốc đa dạng (Động vật, Typography/Chữ) được tô Mask thủ công cực kỳ cẩn thận (Mask nằm lọt thỏm trong viền đen) để tránh dạy AI thói quen "tràn viền".
2. **Kỹ thuật Đồng bộ Scale (Resize Factor):**
   - Thu nhỏ ảnh gốc và ảnh Mask (Ví dụ: `RESIZE_FACTOR = 0.5`) trước khi băm. Giúp khung quét `512x512` ôm trọn được ngữ cảnh lớn hơn của bức tranh.
   - *Lưu ý cơ học:* Thu nhỏ Mask bắt buộc dùng `cv2.INTER_NEAREST` để giữ viền nhị phân cứng cáp, không bị nội suy mờ nhòe.
3. **Bộ lọc Không gian trống:** Chỉ lưu các bản vá (patch) chứa nét vẽ gốc (Alpha > 0) hoặc có mảng Mask (> 0), vứt bỏ hoàn toàn nền trong suốt để tiết kiệm dung lượng.
4. **Chiến lược "Small Data Era":** Cắt bỏ hoàn toàn tập Test, dồn toàn lực dữ liệu theo tỷ lệ **Train (80%) - Val (20%)** để mô hình hội tụ tốt nhất trong bối cảnh dữ liệu khan hiếm.

---

## Khâu 2: Tiền xử lý & Tăng cường Dữ liệu (`dataset.py`)

Băng chuyền tiêu chuẩn hóa và "bơm" thêm dữ liệu (Data Augmentation) để chống học vẹt (Overfitting).

1. **Ép nhãn (Labeling):** Thuật toán `mask_np[mask_np > 0] = 1` ép toàn bộ pixel vùng Fill về class 1, Background về class 0.
2. **Albumentations (Tăng cường dữ liệu cường độ cao):** - Tập Train được áp dụng các phép biến đổi ngẫu nhiên: *HorizontalFlip (Lật ngang), VerticalFlip (Lật dọc), RandomRotate90 (Xoay 90 độ)*. Giúp 1 bức ảnh sinh ra nhiều biến thể.
   - Tập Val được giữ nguyên bản.
3. **Tối ưu hóa GPU:** Đẩy luồng tải dữ liệu qua 4 công nhân độc lập (`num_workers=4`) với `persistent_workers=True` để GPU không bao giờ bị thiếu data.

---

## Khâu 3: Kiến trúc Mạng & Huấn luyện (Loss Kép) (`train.py`)

Khắc phục triệt để bệnh "Tham lam" (Over-segmentation / Tràn viền) bằng cách thay đổi luật chơi của AI.

1. **Chữa bệnh Tham lam (Weighted Loss):** Hạ trọng số phạt `fill_weight` từ `5.0` xuống `2.0`. Mạng U-Net sẽ không còn ám ảnh việc "thà tô lố còn hơn bỏ sót", giúp ranh giới bám khít nét viền đen.
2. **Vũ khí tối thượng - Loss Kép (Hybrid Loss):**
   - **CrossEntropy Loss:** Chấm điểm cứng nhắc trên từng pixel.
   - **Dice Loss:** Chấm điểm dựa trên độ khớp (IoU) của cả một mảng diện tích lớn. Ép mảng trắng phải ôm sát ranh giới viền đen tuyệt đối. Tổng Loss = CE Loss + Dice Loss.
3. **Giám sát W&B:** Ghi log toàn bộ hệ thống (Loss, Accuracy, Precision, Recall, F1-Score) lên Cloud W&B theo thời gian thực. Lưu checkpoint `unet_binary_best.pth` tự động tại Epoch có Val F1-Score cao nhất.

---

## Khâu 4: Hậu xử lý Toán thái học & Trích xuất Vector (`inference_satin_border.py`)

Biến mảng Mask Trắng/Đen thô kệch thành các đường nét tinh tế, tách biệt chuẩn kỹ thuật thêu vi tính.

1. **Đồng bộ Scale & Padding:** Ảnh thực tế được Resize (`0.5`) giống hệt lúc Train, đệm viền (Padding) cho chia hết `512`, cho U-Net quét, sau đó phóng to Mask trả lại kích thước gốc nguyên bản.
2. **Xác suất tự tin (Softmax Thresholding):** Loại bỏ `argmax` dễ dãi. Dùng `torch.softmax`, AI phải tự tin `> 50%` (với model V3 đã rất chuẩn) mới được tô màu Fill.
3. **Chuỗi Kỹ thuật Trích xuất biên Satin (Advanced Morphology):**
   - **Vá lỗ (`MORPH_CLOSE`):** Đảm bảo mảng Fill đặc ruột, không đứt gãy.
   - **Cưỡng ép Tách nét (`Erosion`):** Đánh sập các "cây cầu" pixel đang dính lẹo giữa 2 mảng Fill nằm sát nhau, tạo khoảng hở không gian cho kim thêu.
   - **Trích xuất viền bên trong (`Boundary Extraction`):** Thu nhỏ mảng vừa tách thêm 1 lần nữa, lấy mảng ngoài TRỪ đi mảng trong. Phép toán này tạo ra các dải phân cách mảnh mai (2-3 pixel) mô phỏng chính xác đường chạy kim mũi Satin.
4. **Kết quả cuối cùng:** Trả ra file `AI_Satin_Borders.png` sắc lẹm, sẵn sàng đưa vào thuật toán tìm tọa độ `cv2.findContours` và nội suy đường cong `Shapely` ở Giai đoạn 2.
