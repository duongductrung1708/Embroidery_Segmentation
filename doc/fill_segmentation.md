# Pipeline Giai đoạn 1 (V6 Pro - U-2-Net & Deep Supervision): Binary Fill Segmentation & Satin Boundary Extraction

**Mục tiêu:** Xây dựng hệ thống Trí tuệ Nhân tạo có khả năng nhận thức sắc nét vùng mảng thêu (Fill) từ ảnh phác thảo, đồng thời tích hợp Toán thái học Hình học (Computational Geometry) để trích xuất trực tiếp các đường viền Vector Satin siêu mảnh, sẵn sàng cho việc sinh mã máy thêu. Phiên bản V6 Pro là một cuộc đại tu về kiến trúc bộ não, chuyển từ U-Net truyền thống sang **U-2-Net (Nested U-Structure)** kết hợp **Deep Supervision** để vắt kiệt khả năng nhận diện ranh giới li ti.

---

## Khâu 1: Kiến trúc Dữ liệu Phân tầng (Stratified Data Engineering)

Thay vì băm nát ảnh ra hàng ngàn file nhỏ hay xáo trộn mù quáng, hệ thống áp dụng chiến lược quản lý dữ liệu an toàn và phân tầng độ khó để đánh giá đúng thực lực của AI.

1. **Khởi tạo Dữ liệu Vàng (`split_raw_data.py`):**
   - Kho dữ liệu gốc `data/raw/` được phân loại thủ công thành 3 mức độ: `easy` (mảng to, rõ), `medium` (đan xen), `hard` (chân dung, mắt mũi nhỏ).
   - Áp dụng **Stratified Split (Chia theo Phân tầng):** Rút đều 80% Train và 20% Val từ _từng nhóm độ khó_. Đảm bảo tập Validation luôn có đủ các bài test "khoai" nhất để chống rò rỉ dữ liệu và ảo tưởng sức mạnh (Over-confident).
2. **Cập Nhật Dữ liệu mới (`append_new_data.py`):**
   - Khi có ảnh mới, đưa vào trạm trung chuyển `data/raw_new/` theo đúng thư mục độ khó tương ứng.
   - Script tự động chia 80/20 và **ghi nối (append)** vào tập Train/Val, sau đó cất gọn gàng ảnh mới vào kho `data/raw/` mà không làm xáo trộn các đề thi cũ.

---

## Khâu 2: Tiền xử lý Tensor & Data Augmentation Hạng nặng (`dataset.py`)

Băng chuyền tiêu chuẩn hóa và "bóp méo" dữ liệu trước khi đẩy vào GPU, tập trung vào việc làm rõ vân thêu và cấu trúc ranh giới.

1. **On-the-fly Cropping (Cắt ảnh động):**
   - Nhân bản danh sách ảnh trong 1 Epoch (Ví dụ: `crops_per_image = 20`). Thu nhỏ ảnh gốc (`Resize Factor = 0.5`) và cắt ngẫu nhiên 20 vùng `512x512` rồi mới đẩy vào GPU (Tiết kiệm VRAM triệt để).
2. **Negative Sampling & Cắt thông minh:**
   - **80% cơ hội:** Dùng `CropNonEmptyMaskIfExists` nhắm thẳng mảng Mask trắng để cắt, ép AI học chi tiết nét vẽ.
   - **20% cơ hội:** Dùng `RandomCrop` lia vào các vùng nền đen để AI biết cách "giữ im lặng" khi gặp khoảng trống.
3. **Đột biến gen & Tăng cường Độ nét (Spatial & Pixel Augmentation):**
   - `Affine` & `ElasticTransform`: Xoay, co giãn, uốn éo hình dáng nét vẽ mô phỏng độ co dãn của sợi vải.
   - **`CLAHE` & `Sharpen`:** Tăng tương phản cục bộ và biến ranh giới mờ thành "vách đá dựng đứng" để AI dễ cắt viền.
   - `CoarseDropout`: Khoét lỗ đen ngẫu nhiên để giả lập đứt nét.

---

## Khâu 3: Kiến trúc Mạng & Huấn luyện Tự động (`train.py` & `utils.py`)

Nâng cấp "bộ não" từ U-Net lên **U-2-Net (Kiến trúc lồng nhau đa quy mô)**, được giám sát nghiêm ngặt bởi cơ chế Deep Supervision và Fixed Tracking.

1. **Kiến trúc Nested U-Structure (U-2-Net):**
   - Mỗi Block Convolution được thay thế bằng một mạng U-Net thu nhỏ (RSU), giúp bắt được cả chi tiết cực nhỏ lẫn ngữ cảnh mảng lớn mà không làm phình to bộ nhớ.
2. **Deep Supervision & 3 Loss Kết hợp (Hybrid Loss):**
   - U-2-Net sinh ra **7 bản đồ dự đoán (outputs)** từ các chặng khác nhau.
   - **Mỗi bản đồ** đều bị tính tổng Loss bằng bộ 3 vũ khí:
     - **Focal Loss (Label Smoothing=0.1):** Phạt nặng các pixel ranh giới đoán sai, lờ đi vùng nền dễ đoán.
     - **Dice Loss:** Ép mảng dự đoán phải ôm khít diện tích Ground Truth.
     - **Boundary Loss (Canny Edge):** Ép AI hoàn thành nhiệm vụ phụ: Kẻ đúng chính xác đường viền ngoài cùng.
   - Quá trình Backpropagation lan truyền đạo hàm của cả 7 Loss này, ép mô hình phải "chuẩn từ trong trứng".
3. **Giám sát Thị giác Tuyệt đối (Fixed Batch Tracking trên WandB):**
   - Trích xuất riêng một lô ảnh Validation (Batch) **cố định** ở ngay đầu chương trình.
   - Mỗi Epoch đều dự đoán lại đúng lô ảnh này để xuất lên Weights & Biases (W&B). Kỹ sư có thể kéo thanh trượt Epoch để xem quá trình AI "mọc rễ" và nắn nót ranh giới qua từng ngày trên cùng một bức ảnh.
4. **Cơ chế Chống sập & Fine-Tuning Tự động:** Lưu checkpoit liên tục và tự động bóp Learning Rate khi Fine-tune data mới.

---

## Khâu 4: Dự đoán Thực chiến (`inference.py`)

Đưa mô hình vào kiểm thử trực quan với các cơ chế Hậu xử lý Toán thái học sắc bén và kiểm soát ranh giới khắt khe.

1. **Trích xuất d0 (Fusion Map):**
   - Nạp ảnh qua U-2-Net, loại bỏ các kết quả phụ, chỉ lấy kết quả chắt lọc tinh túy nhất `d0` ở lớp cuối cùng.
2. **Xác suất tự tin khắt khe (Softmax Thresholding):**
   - Ép ngưỡng Threshold của class Fill lên **> 0.65**. AI phải cực kỳ tự tin mới được tô màu, dập tắt ảo giác lem màu ra vùng Satin.
3. **Kỹ thuật Padding & Vá viền:** Đệm viền chia hết cho `512`, dự đoán qua Sliding Window. Phóng to nội suy `INTER_LINEAR` kết hợp Gaussian Blur nhẹ để khử răng cưa.
4. **Thợ mộc OpenCV (Advanced Morphology):**
   - Dùng `MORPH_CLOSE` để lấp đầy các khe nứt li ti bên trong mảng thêu.
   - Dùng `Erosion` nhẹ (3x3) để gọt giũa lại độ dày nét vẽ cho chuẩn xác.
5. **Kết quả:** Hiển thị trực quan song song ảnh gốc và ảnh Overlay màu xanh lá (Nền trong suốt RGBA), ranh giới cắt sắc lẹm, xuất trực tiếp ra mask Đen/Trắng phục vụ Giai đoạn 2 (Vectorization).
