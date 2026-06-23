# Pipeline Giai đoạn 1 (V5 Pro - Precision & Boundary Focus): Binary Fill Segmentation & Satin Boundary Extraction

**Mục tiêu:** Xây dựng hệ thống Trí tuệ Nhân tạo có khả năng nhận thức sắc nét vùng mảng thêu (Fill) từ ảnh phác thảo, đồng thời tích hợp Toán thái học Hình học (Computational Geometry) để trích xuất trực tiếp các đường viền Vector Satin siêu mảnh, sẵn sàng cho việc sinh mã máy thêu. Phiên bản V5 Pro tập trung vào trị dứt điểm lỗi "Tràn viền" (Bleeding) và "Mất chi tiết" ở các vùng khó (Mắt, Mũi) bằng cách ép AI học đường biên ranh giới.

---

## Khâu 1: Kiến trúc Dữ liệu & Học liên tục (Continuous Learning)

Thay vì băm nát ảnh ra hàng ngàn file nhỏ gây rác ổ cứng, hệ thống giữ nguyên ảnh gốc và ứng dụng cơ chế quản lý dữ liệu an toàn tuyệt đối.

1. **Khởi tạo Dữ liệu Vàng (`split_raw_data.py`):**
   - Quét toàn bộ ảnh trong kho tổng `data/raw/` và xáo trộn ngẫu nhiên.
   - Chia ảnh gốc trực tiếp vào 2 thư mục `train` (80%) và `val` (20%). KHÔNG BĂM ẢNH. Tập Validation là những bức tranh AI chưa từng được thấy, chống rò rỉ dữ liệu (Data Leakage) tuyệt đối.
2. **Bơm Dữ liệu mới (`append_new_data.py`):**
   - Khi có ảnh mới (do con người tô mask thêm), đưa vào trạm trung chuyển `data/raw_new/`.
   - Script sẽ phân loại 80/20 và **ghi nối (append)** vào tập Train/Val hiện tại mà không làm xáo trộn các đề thi cũ.
   - Sau khi nạp xong, tự động dọn dẹp và cất ảnh vào kho tổng `data/raw/` để lưu trữ.

---

## Khâu 2: Tiền xử lý Tensor & Data Augmentation Hạng nặng (`dataset.py`)

Băng chuyền tiêu chuẩn hóa và "bóp méo" dữ liệu trước khi đẩy vào GPU, tập trung vào việc làm rõ vân thêu và cấu trúc ranh giới.

1. **On-the-fly Cropping (Cắt ảnh động):**
   - Nhân bản danh sách ảnh trong 1 Epoch (Ví dụ: `crops_per_image = 20`). CPU sẽ tự động đọc ảnh gốc to, thu nhỏ (`Resize Factor = 0.5`) và cắt ngẫu nhiên 20 vùng khác nhau kích thước `512x512` rồi mới đẩy vào GPU.
2. **Negative Sampling & Cắt thông minh (Albumentations):**
   - **80% cơ hội:** Dùng `CropNonEmptyMaskIfExists` nhắm thẳng mảng Mask trắng để cắt, ép AI tập trung học chi tiết nét vẽ.
   - **20% cơ hội:** Dùng `RandomCrop` lia ngẫu nhiên vào các vùng nền đen để AI biết cách "giữ im lặng" khi gặp khoảng trống.
3. **Đột biến gen & Tăng cường Độ nét (Spatial & Pixel Augmentation):**
   - `Affine`: Xoay tự do (-180° đến 180°), phóng to thu nhỏ, dịch chuyển.
   - `ElasticTransform`: Kéo giãn, uốn éo hình dáng nét vẽ mô phỏng độ co giãn của sợi vải.
   - **(NEW) `CLAHE` & `Sharpen`:** Tăng cường độ tương phản cục bộ và làm sắc nét các đường ranh giới (đặc biệt là vùng mắt, mũi), giúp mô hình nhìn rõ sự khác biệt giữa Satin và Fill.
   - `CoarseDropout`: Khoét các lỗ vuông màu đen ngẫu nhiên để giả lập nét vẽ bị đứt gãy.

---

## Khâu 3: Kiến trúc Mạng & Huấn luyện Tự động (`train.py` & `utils.py`)

Bộ não U-Net được giám sát bởi **Bộ 3 Loss Hạng Nặng**, thiết kế chuyên biệt để phạt nặng các lỗi dự đoán sai đường biên và giảm ảo giác.

1. **Hộp đồ nghề (`utils.py`):** Cố định môi trường học (`seed_everything`), chứa các hàm tính toán Metrics và các class Loss nâng cao để source code `train.py` gọn gàng.
2. **Vũ khí 3 Loss Kết hợp (Hybrid Loss):**
   - **(NEW) Focal Loss (với Label Smoothing):** Thay thế CrossEntropy. Phạt cực nặng những pixel AI đoán sai (thường nằm ở ranh giới). Label Smoothing (0.1) giúp AI không quá bảo thủ với nhãn do người tô bằng tay (thường bị lem).
   - **Dice Loss:** Chấm điểm dựa trên độ khớp (IoU) của cả mảng diện tích. Ép mảng dự đoán phải ôm khít Ground Truth.
   - **(NEW) Boundary Loss (BCE trên Canny Edge):** Trích xuất đường biên của Mask thật bằng thuật toán Canny. Ép AI học thêm một nhiệm vụ phụ: "Hãy dự đoán chính xác đường viền này". Trị dứt điểm lỗi tràn viền (Bleeding).
3. **Cơ chế 2 trong 1 (Chống sập & Fine-Tuning):**
   - **Chống sập tuyệt đối:** Tự động lưu `unet_binary_last.pth`. Nếu sập nguồn, tự động khôi phục trí nhớ và số đếm Epoch để chạy tiếp.
   - **Fine-Tuning Tự động:** Phát hiện `best.pth` cũ khi train mới, tự động nạp trí nhớ và giảm Learning Rate (xuống `1e-5`) để gọt giũa an toàn, tránh Catastrophic Forgetting.

---

## Khâu 4: Dự đoán Thực chiến (`inference.py`)

Đưa mô hình vào kiểm thử trực quan với các cơ chế Hậu xử lý Toán thái học sắc bén và kiểm soát ranh giới khắt khe.

1. **(NEW) Xác suất tự tin khắt khe (Softmax Thresholding):**
   - Ép ngưỡng (Threshold) của class Fill lên **0.6 (hoặc cao hơn)**. AI phải chắc chắn `> 60%` mới được coi là vùng thêu, giúp dập tắt hoàn toàn các vùng "ảo giác" tràn ra ngoài lớp Satin.
2. **Kỹ thuật Padding & Vá viền:** Đệm viền chia hết cho 512, dự đoán qua Sliding Window. Phóng to nội suy `INTER_NEAREST` để giữ ranh giới nhị phân cứng cáp.
3. **Thợ mộc OpenCV (Advanced Morphology):**
   - Tối ưu hóa mảng dự đoán bằng `MORPH_OPEN` để loại bỏ các điểm rác thừa li ti (Noise removal).
   - Dùng `MORPH_CLOSE` để lấp đầy các khe nứt li ti bên trong mảng thêu (Hole filling).
4. **Kết quả:** Hiển thị trực quan song song ảnh gốc và ảnh Overlay màu xanh lá, ranh giới cắt gọn gàng, sẵn sàng xuất ra mảng Vector sắc cạnh phục vụ Giai đoạn 2.