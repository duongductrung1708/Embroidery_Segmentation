# AI Embroidery Segmentation: Automatic Satin & Tatami Recognition using U-Net

## 1. Giới thiệu dự án

Trong ngành thêu vi tính, hai kiểu mũi thêu phổ biến nhất là:

- **Satin Stitch**: mũi dài, chạy song song, thường dùng cho viền, chữ và các chi tiết nhỏ.
- **Tatami Stitch**: mũi ngắn, đan xen theo từng lớp, thường dùng để phủ kín các vùng lớn.

Trong quá trình thiết kế hoặc kiểm tra chất lượng file thêu, việc xác định chính xác vùng nào là Satin và vùng nào là Tatami thường phải thực hiện thủ công bởi kỹ thuật viên có kinh nghiệm.

Mục tiêu của dự án là xây dựng một hệ thống AI có khả năng:

```text
Ảnh bề mặt thêu
        ↓
     U-Net
        ↓
Mask phân vùng
        ↓
Satin / Tatami
```

Hệ thống giúp tự động phân tích cấu trúc mũi thêu, hỗ trợ:

- Kiểm tra chất lượng file thêu.
- Phân tích cấu trúc thiết kế.
- Chuyển đổi dữ liệu thiết kế.
- Hỗ trợ sinh file thêu tự động trong tương lai.

---

# 2. Ý tưởng cốt lõi của hệ thống

## Vấn đề của bài toán Segmentation

Thông thường để huấn luyện một mô hình Segmentation cần có:

```text
Ảnh
+
Mask gán nhãn
```

Ví dụ:

```text
Ảnh chó
+
Mask con chó
```

Tuy nhiên việc tạo Mask thủ công tốn rất nhiều thời gian.

---

## Lợi thế đặc biệt của ngành thêu

Khác với ảnh thông thường, file thêu `.dst` đã chứa:

```text
Toàn bộ đường đi của kim thêu
```

Mỗi file DST lưu:

```text
(X1, Y1)
(X2, Y2)
(X3, Y3)
...
```

Tức là hệ thống đã biết chính xác vị trí từng mũi kim.

Nhờ đó có thể:

```text
DST
↓
Phân tích mũi thêu
↓
Tự động gán nhãn
↓
Sinh Mask
```

mà không cần con người ngồi vẽ nhãn.

Đây chính là ý tưởng trung tâm của dự án.

---

# 3. Luồng hoạt động tổng thể

```text
DST Design File
        ↓
Phân tích tọa độ mũi thêu
        ↓
Tự động phân loại Satin/Tatami
        ↓
Render thành ảnh và mask
        ↓
Patch Extraction
        ↓
Tensor Dataset
        ↓
U-Net Training
        ↓
Pixel Classification
        ↓
Segmentation Map
        ↓
Visualization
```

---

# 4. Khâu 1: Phân tích file DST và tự động gán nhãn

## 4.1 Đọc dữ liệu thêu

File `.dst` được đọc bằng thư viện:

```python
pyembroidery
```

Sau khi đọc, hệ thống thu được:

```text
(X1, Y1)
(X2, Y2)
(X3, Y3)
...
```

là danh sách toàn bộ tọa độ kim thêu.

---

## 4.2 Tính khoảng cách giữa các mũi kim

Đối với mỗi cặp điểm liên tiếp:

```text
Pi
Pi+1
```

hệ thống tính khoảng cách Euclid:

Δd = √((X₂ − X₁)² + (Y₂ − Y₁)²)

Mục đích:

```text
Tọa độ
↓
Đặc trưng hình học
↓
Loại mũi thêu
```

---

## 4.3 Tự động nhận diện Satin và Tatami

### Satin

Đặc điểm:

- Mũi dài
- Song song
- Khoảng cách lớn

```text
Δd > Threshold
```

→ Gán nhãn Satin

---

### Tatami

Đặc điểm:

- Mũi ngắn
- Đan xen
- Mật độ cao

```text
Δd ≤ Threshold
```

→ Gán nhãn Tatami

---

## 4.4 Tạo ảnh và Mask

U-Net không hiểu:

```text
Vector
Tọa độ
DST
```

U-Net chỉ hiểu:

```text
Tensor ảnh
```

Do đó cần:

```text
DST
↓
Vector
↓
Raster Image
↓
Tensor
```

Hệ thống render toàn bộ đường thêu lên Canvas kích thước lớn.

---

# 5. Khâu 2: Patch Extraction

## Tại sao không Resize?

Giả sử ảnh gốc:

```text
15000 × 12000 px
```

Nếu ép:

```text
15000
↓
512
```

sẽ làm:

```text
Mất chi tiết mũi thêu
Mất texture
Mất cấu trúc sợi chỉ
```

---

## Giải pháp

Áp dụng Sliding Window:

```text
Ảnh lớn
↓
Patch 512×512
↓
Patch 512×512
↓
Patch 512×512
```

Thông số:

```python
PATCH_SIZE = 512
STRIDE = 256
```

---

## Lợi ích

### Giữ nguyên độ nét

AI nhìn được từng mũi thêu.

### Tăng dữ liệu

Một file DST có thể tạo ra hàng trăm patch.

### Học ngữ cảnh tốt hơn

Overlap 50%.

```text
Stride = 256
Patch = 512
```

---

# 6. Khâu 3: Data Pipeline

## Ảnh đầu vào

```python
transforms.ToTensor()
```

Biến đổi:

```text
[0,255]
↓
[0,1]
```

Kích thước:

```text
[B,3,512,512]
```

---

## Mask đầu ra

Giữ nguyên:

```text
0 = Background
1 = Satin
2 = Tatami
```

Kích thước:

```text
[B,512,512]
```

---

## DataLoader

Chức năng:

- Batch dữ liệu
- Shuffle dữ liệu
- Tăng tốc huấn luyện

---

# 7. Khâu 4: U-Net Architecture

## Mục tiêu

Thực hiện:

```text
Ảnh
↓
Pixel Classification
↓
Mask
```

---

## Encoder

Nhiệm vụ:

```text
Trích xuất đặc trưng
```

Các tầng đầu học:

```text
Edge
Line
Corner
```

---

Các tầng giữa học:

```text
Texture
Pattern
```

---

Các tầng sâu học:

```text
Satin Region
Tatami Region
```

---

## Bottleneck

Đây là nơi chứa:

```text
Thông tin ngữ nghĩa cao nhất
```

Mô tả:

```text
Ảnh này có cấu trúc gì?
Đây là Satin hay Tatami?
```

---

## Decoder

Nhiệm vụ:

```text
Khôi phục kích thước ảnh
```

Từ:

```text
32×32
```

quay lại:

```text
512×512
```

---

## Skip Connections

Nếu chỉ Encoder + Decoder:

```text
Mất vị trí chi tiết
```

Đặc biệt với:

```text
Sợi chỉ nhỏ
Biên Satin
```

Skip Connection giúp:

```text
Encoder
──────► Decoder
```

Khôi phục:

- Biên sắc nét
- Chi tiết không gian
- Texture nhỏ

Đây là yếu tố tạo nên sức mạnh của U-Net.

---

# 8. Khâu 5: Huấn luyện

## Forward Pass

Đầu ra:

```text
[B,3,512,512]
```

Trong đó:

```text
Channel 0 → Background
Channel 1 → Satin
Channel 2 → Tatami
```

---

## CrossEntropy Loss

Tại mỗi pixel:

Ví dụ:

```text
Ground Truth = Satin
```

Model dự đoán:

```text
Background = 0.1
Satin = 0.7
Tatami = 0.2
```

CrossEntropy sẽ tính mức sai lệch.

Mục tiêu:

```text
Satin → 1.0
Tatami → 0.0
Background → 0.0
```

---

## Backpropagation

```python
loss.backward()
```

Tính gradient bằng:

```text
Chain Rule
```

---

## Adam Optimizer

Nhiệm vụ:

```text
Điều chỉnh trọng số
Giảm Loss
Tăng Accuracy
```

---

# 9. Suy luận thực tế (Inference)

Sau khi huấn luyện:

```python
model.eval()
torch.no_grad()
```

---

## Dự đoán

Đầu ra:

```text
[B,3,512,512]
```

Chọn lớp có xác suất cao nhất:

```python
torch.argmax()
```

---

## Sinh Mask

```text
0 = Background
1 = Satin
2 = Tatami
```

---

## Tô màu trực quan

```text
Background → Đen
Satin      → Đỏ
Tatami     → Xanh lá
```

Kết quả:

```text
Ảnh gốc
+
Mask dự đoán
```

được hiển thị song song để kỹ sư dễ kiểm tra.

---

# 10. Kết quả đầu ra cuối cùng

Đầu vào:

```text
Ảnh bề mặt thêu
```

Đầu ra:

```text
Segmentation Mask
```

cho biết chính xác:

- Vùng Satin
- Vùng Tatami
- Vùng nền

ở cấp độ từng pixel.

---

# 11. Định hướng phát triển

Trong tương lai hệ thống có thể mở rộng:

```text
DST
↓
AI Segmentation
↓
Vector Reconstruction
↓
SVG
↓
Ink/Stitch
↓
DST/PES tự động
```

hoặc:

```text
Ảnh thêu thực tế
↓
AI Inspection
↓
Kiểm tra lỗi sản xuất
↓
Cảnh báo tự động
```

để xây dựng hệ thống kiểm tra chất lượng thêu hoàn toàn tự động trong môi trường công nghiệp.

### Dataset Assumption

Trong giai đoạn hiện tại, nhãn Satin và Tatami được sinh tự động dựa trên khoảng cách Euclid giữa các mũi kim liên tiếp.

Do đó chất lượng nhãn phụ thuộc trực tiếp vào giá trị ngưỡng THRESHOLD.

Hệ thống hiện tại được xem là:

Weakly Supervised Segmentation

thay vì Fully Supervised Segmentation.

Trong tương lai cần được đối chứng bởi chuyên gia thêu để hiệu chỉnh nhãn.
