# Pipeline Giai đoạn 1 (V4 Pro - Production): Binary Fill Segmentation & Satin Boundary Extraction

**Mục tiêu:** Xây dựng hệ thống Trí tuệ Nhân tạo có khả năng nhận thức sắc nét vùng mảng thêu (Fill) từ ảnh phác thảo, đồng thời tích hợp Toán thái học Hình học (Computational Geometry) để trích xuất trực tiếp các đường viền Vector Satin siêu mảnh, sẵn sàng cho việc sinh mã máy thêu. Phiên bản V4 Pro tập trung vào tối ưu hóa ổ cứng, học liên tục (Continuous Learning) và chống sập tuyệt đối.

---

## Khâu 1: Kiến trúc Dữ liệu & Học liên tục (Continuous Learning)

Thay vì băm nát ảnh ra hàng ngàn file nhỏ gây rác ổ cứng, V4 Pro giữ nguyên ảnh gốc và ứng dụng cơ chế quản lý dữ liệu an toàn tuyệt đối.

1. **Khởi tạo Dữ liệu Vàng (`split_raw_data.py`):** - Quét toàn bộ ảnh trong kho tổng `data/raw/` và xáo trộn ngẫu nhiên.
   - Chia ảnh gốc trực tiếp vào 2 thư mục `train` (80%) và `val` (20%). KHÔNG BĂM ẢNH. Tập Validation là những bức tranh AI chưa từng được thấy, chống rò rỉ dữ liệu (Data Leakage) tuyệt đối.
2. **Bơm Dữ liệu mới (`append_new_data.py`):**
   - Khi có ảnh mới (do con người tô mask thêm), đưa vào trạm trung chuyển `data/raw_new/`.
   - Script sẽ phân loại 80/20 và **ghi nối (append)** vào tập Train/Val hiện tại mà không làm xáo trộn các đề thi cũ.
   - Sau khi nạp xong, tự động dọn dẹp và cất ảnh vào kho tổng `data/raw/` để lưu trữ.

---

## Khâu 2: Tiền xử lý Tensor & Data Augmentation Hạng nặng (`dataset.py`)

Băng chuyền tiêu chuẩn hóa và "bóp méo" dữ liệu trước khi đẩy vào GPU để chống Overfitting (Học vẹt).

1. **On-the-fly Cropping (Cắt ảnh động):**
   - Nhân bản danh sách ảnh trong 1 Epoch (Ví dụ: `crops_per_image = 20`). CPU sẽ tự động đọc ảnh gốc to, thu nhỏ (`Resize Factor = 0.5`) và cắt ngẫu nhiên 20 vùng khác nhau kích thước `512x512` rồi mới đẩy vào GPU.
2. **Negative Sampling & Cắt thông minh (Albumentations):**
   - **80% cơ hội:** Dùng `CropNonEmptyMaskIfExists` nhắm thẳng mảng Mask trắng để cắt, ép AI tập trung học chi tiết nét vẽ (Đặc trị mất cân bằng lớp cực đoan).
   - **20% cơ hội:** Dùng `RandomCrop` lia ngẫu nhiên vào các vùng nền đen để AI biết cách "giữ im lặng" khi gặp khoảng trống (Chống ảo giác).
3. **Đột biến gen Không gian (Spatial Augmentation):**
   - `Affine`: Xoay tự do (-180° đến 180°), phóng to thu nhỏ, dịch chuyển. Lấp đầy viền hở bằng màu đen (`fill=0`).
   - `ElasticTransform`: Kéo giãn, uốn éo hình dáng nét vẽ mô phỏng độ co giãn của sợi vải.
   - `CoarseDropout`: Khoét các lỗ vuông màu đen ngẫu nhiên để giả lập nét vẽ bị đứt gãy, ép AI học theo ngữ cảnh tổng thể.

---

## Khâu 3: Kiến trúc Mạng & Huấn luyện Tự động (`train.py` & `utils.py`)

Bộ não U-Net được giám sát bởi Loss Kép, tách biệt các hàm tiện ích và trang bị cơ chế tự phục hồi.

1. **Hộp đồ nghề (`utils.py`):** Cố định môi trường học (`seed_everything`), cô lập hàm tính toán Metrics (Accuracy, Precision, Recall, F1) và `DiceLoss` để source code gọn gàng, có thể tái sử dụng cho khâu Inference.
2. **Vũ khí Loss Kép (Hybrid Loss):**
   - **CrossEntropy Loss:** Chấm điểm cứng nhắc trên từng pixel. Trọng số phạt `fill_weight` linh hoạt (thường set cực cao lên `10.0 - 20.0` nếu vùng Fill quá nhỏ như mắt/mũi).
   - **Dice Loss:** Chấm điểm dựa trên độ khớp (IoU) của cả mảng diện tích. Ép mảng trắng phải ôm khít ranh giới.
3. **Cơ chế 2 trong 1 (Chống sập & Fine-Tuning):**
   - **Chống sập tuyệt đối:** Cuối mỗi Epoch tự động lưu file `unet_binary_last.pth` (chứa tạ Model, trí nhớ Optimizer, và số đếm Epoch). Nếu sập nguồn, chạy lại file sẽ tự động khôi phục và chạy tiếp chính xác tại Epoch dang dở.
   - **Fine-Tuning Tự động:** Khi nạp thêm data mới và bắt đầu train lại, hệ thống sẽ phát hiện bộ não `best.pth` cũ, tự động nạp trí nhớ và **bóp phanh Learning Rate** (từ `1e-4` xuống `1e-5`) để gọt giũa an toàn, tránh hội chứng Catastrophic Forgetting.

---

## Khâu 4: Dự đoán Thực chiến (`inference.py`)

Đưa mô hình vào kiểm thử trực quan với các cơ chế Hậu xử lý Toán thái học sắc bén.

1. **Xác suất tự tin (Softmax Thresholding):**
   - Loại bỏ `argmax` dễ dãi. Dùng `torch.softmax`, AI phải tự tin `> 50%` mới được tô màu Fill.
2. **Kỹ thuật Padding & Vá viền:** Đệm viền chia hết cho 512, dự đoán qua Sliding Window, và thu dọn phần đệm thừa. Khi phóng to trả về kích thước gốc luôn dùng nội suy `INTER_NEAREST` để giữ ranh giới nhị phân cứng cáp.
3. **Thợ mộc OpenCV (Advanced Morphology):**
   - Dùng `MORPH_CLOSE` (Kernel 3x3) quét qua lớp Mask AI trả về để lấp đầy các khe nứt li ti.
   - Bỏ mài mòn (`Erosion`) để giữ nguyên độ béo của vùng Mask, bám sát mép bản vẽ.
4. **Kết quả:** Hiển thị trực quan song song ảnh gốc và ảnh Overlay màu xanh lá, sẵn sàng xuất ra mảng Vector sắc cạnh phục vụ Giai đoạn 2.
