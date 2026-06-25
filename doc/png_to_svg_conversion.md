# Giai đoạn 2 - Bước 1: Chuyển đổi PNG sang SVG

## Tổng quan

Bước này chuyển đổi các ảnh PNG đã phân đoạn từ mô hình U-2-Net sang định dạng vector SVG tương thích với máy thêu. Pipeline hiện tại tích hợp fal.ai NAFNet để tăng chất lượng ảnh trước khi vector hóa.

## Mục đích

- Chuyển đổi ảnh PNG raster sang định dạng vector SVG
- Giữ nguyên độ trong suốt từ ảnh gốc
- Duy trì độ chính xác màu cho mẫu thêu
- Tạo đường dẫn sạch, mượt cho máy thêu
- Tăng chất lượng ảnh bằng AI (NAFNet) trước khi vector hóa

## Triển khai

### File: `data_prep/png_to_svg.py`

#### Các thành phần chính

**1. Tăng chất lượng ảnh với fal.ai (`enhance_with_fal`)**

```python
def enhance_with_fal(img_path, original_alpha=None):
    """
    Sử dụng fal.ai NAFNet model để làm mượt và khử nhiễu ảnh.
    Giữ lại alpha channel gốc nếu có.
    """
```

- **NAFNet Deblur**: Sử dụng model fal.ai để làm mượt và khử nhiễu ảnh
- **Giữ Alpha Channel**: Trích xuất alpha channel gốc trước khi xử lý, dán lại sau khi nhận ảnh RGB từ API
- **Fallback**: Nếu fal.ai không phản hồi, sử dụng ảnh gốc

**2. Xử lý kênh Alpha (`clean_rgba_image`)**

```python
def clean_rgba_image(img):
    """
    Làm sạch kênh Alpha (trong suốt) bằng cách khử nhiễu,
    giữ nguyên format RGBA để vtracer có thể tạo SVG với nền trong suốt.
    Đồng thời làm mượt cả RGB channels để giảm nhọn nhô từ ảnh gốc.
    """
```

- **Threshold thay vì Blur**: Sử dụng `cv2.threshold` để làm SẮC LẸM alpha channel (binary 0/255)
  - Tránh Gaussian Blur tạo gradient gây vtracer tạo hàng ngàn vector path
  - vtracer cần alpha binary để tránh file SVG nặng hàng chục MB
- **Bilateral Filter**: Làm mượt RGB channels để giữ cạnh nhưng giảm nhiễu
- **Morphology Operations**: Open/Close để loại bỏ nhiễu alpha channel
- **Giữ nguyên RGBA**: Duy trì độ trong suốt thay vì ghép lên nền trắng

**3. Pipeline Vector hóa (`process_pipeline`)**

- Quét nhiều định dạng ảnh (PNG, JPG, JPEG)
- Trích xuất alpha channel gốc trước khi xử lý với fal.ai
- Xử lý ảnh qua fal.ai NAFNet để tăng chất lượng
- Xử lý từng ảnh qua làm sạch và vector hóa
- Sử dụng thư viện `vtracer` để tạo SVG

#### Tham số VTracer

```python
vtracer.convert_image_to_svg_py(
    clean_path,
    svg_path,
    colormode="color",        # Giữ nguyên màu gốc
    hierarchical="stacked",   # Tách layer
    mode="spline",            # Đường cong Bezier mượt
    filter_speckle=4,         # Tăng ngưỡng khử nhiễu để làm mượt hơn
    color_precision=12,       # Độ chính xác màu cao
    layer_difference=12,      # Phân tách màu tốt hơn
    corner_threshold=20,      # Giảm ngưỡng góc để đường cong mượt hơn
    length_threshold=4.0,     # Tăng ngưỡng độ dài đường cong để loại bỏ các đoạn ngắn
    max_iterations=25,        # Số lần lặp hội tụ
    splice_threshold=20,      # Giảm ngưỡng nối đường để đường cong mượt hơn
    path_precision=12         # Độ chính xác đường cong
)
```

## Input/Output

### Input
- **Thư mục**: `data_prep/data/dirty_png/`
- **Định dạng**: PNG, JPG, JPEG (không phân biệt hoa thường)
- **Yêu cầu**: Ảnh có hoặc không có kênh alpha

### Output
- **PNG đã làm sạch**: `data_prep/data/clean_png/` - Ảnh đã tiền xử lý với alpha sắc lẹm (binary)
- **File SVG**: `data_prep/data/svg/` - Mẫu thêu đã vector hóa

## Thách thức & Giải pháp

### Thách thức 1: Mất độ trong suốt khi qua fal.ai API
**Vấn đề**: NAFNet API trả về ảnh RGB và vứt bỏ alpha channel gốc.

**Giải pháp**: 
- Trích xuất alpha channel gốc trước khi gửi ảnh qua fal.ai
- Sau khi nhận ảnh RGB đã deblur, dán lại alpha channel gốc vào
- Đảm bảo transparency được giữ nguyên xuyên suốt pipeline

### Thách thức 2: Alpha Channel bị mờ gây SVG nặng
**Vấn đề**: Gaussian Blur trên alpha tạo gradient (50, 100, 150) thay vì binary (0, 255). vtracer cố gắng mô phỏng gradient bằng hàng ngàn vector path, file SVG nặng hàng chục MB.

**Giải pháp**:
- Thay Gaussian Blur bằng `cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)`
- Alpha giờ chỉ có giá trị 0 hoặc 255 (sắc lẹm/binary)
- Tránh vtracer tạo quá nhiều vector path mô phỏng gradient

### Thách thức 3: Cạnh răng cưa từ ảnh gốc U-2-Net
**Vấn đề**: Ảnh gốc từ U-2-Net có nhiều chỗ nhọn nhô, không mượt.

**Giải pháp**:
- Tích hợp fal.ai NAFNet để làm mượt và khử nhiễu ảnh trước khi vector hóa
- Sử dụng bilateral filter trên RGB channels để giữ cạnh nhưng giảm nhiễu
- Điều chỉnh tham số vtracer để đường cong mượt hơn

### Thách thức 4: Giảm màu (Color Quantization)
**Vấn đề**: Bảng màu bị giảm gây banding và biểu diễn màu không chính xác.

**Giải pháp**:
- Giữ `color_precision=12` để độ chính xác màu cao
- Giữ `layer_difference=12` để phân tách vùng màu tốt
- Duy trì `colormode="color"` để giữ đầy đủ màu

## Cách sử dụng

### Cài đặt Dependencies

```bash
pip install opencv-python numpy vtracer fal-client requests pillow tqdm
```

### Cấu hình API Key

```bash
export FAL_KEY="your-api-key-here"
```

### Chạy Pipeline

```bash
cd data_prep
python3 png_to_svg.py
```

Hoặc sử dụng script:

```bash
sh run_prep.sh
```

## Dependencies

- `opencv-python` - Xử lý ảnh
- `numpy` - Thao tác mảng
- `vtracer` - Chuyển đổi raster sang vector
- `fal-client` - Client cho fal.ai API
- `requests` - HTTP requests để tải ảnh từ fal.ai
- `pillow` - Xử lý ảnh PIL
- `tqdm` - Thanh tiến trình

## Các bước tiếp theo

Sau khi chuyển đổi SVG, giai đoạn tiếp theo bao gồm:
- Tối ưu hóa SVG cho máy thêu
- Đơn giản hóa đường dẫn và tạo mũi thêu
- Chuyổi đổi định dạng sang format riêng của máy (PES, DST, v.v.)

## Ghi chú

- Thời gian xử lý phụ thuộc vào độ phức tạp và kích thước ảnh
- fal.ai NAFNet thêm khoảng 20-30s mỗi ảnh nhưng cải thiện chất lượng đáng kể
- Alpha channel binary giúp file SVG nhẹ hơn nhiều so với gradient alpha
- Nếu fal.ai không phản hồi, pipeline tự động fallback về ảnh gốc