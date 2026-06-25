# Giai đoạn 2 - Bước 1: Chuyển đổi PNG sang SVG

## Tổng quan

Bước này chuyển đổi các ảnh PNG đã phân đoạn từ mô hình U-2-Net sang định dạng vector SVG tương thích với máy thêu.

## Mục đích

- Chuyển đổi ảnh PNG raster sang định dạng vector SVG
- Giữ nguyên độ trong suốt từ ảnh gốc
- Duy trì độ chính xác màu cho mẫu thêu
- Tạo đường dẫn sạch, mượt cho máy thêu

## Triển khai

### File: `data_prep/png_to_svg.py`

#### Các thành phần chính

**1. Xử lý kênh Alpha (`clean_rgba_image`)**

```python
def clean_rgba_image(img):
    """
    Làm sạch kênh Alpha (trong suốt) bằng cách khử nhiễu,
    giữ nguyên format RGBA để vtracer có thể tạo SVG với nền trong suốt.
    """
```

- **Gaussian Blur**: Áp dụng cho kênh alpha để làm mượt các cạnh răng cưa
- **Thao tác Morphology**: Các thao tác Open/Close để loại bỏ nhiễu nhưng vẫn giữ được cạnh
- **Giữ nguyên RGBA**: Duy trì độ trong suốt thay vì ghép lên nền trắng

**2. Pipeline Vector hóa (`process_pipeline`)**

- Quét nhiều định dạng ảnh (PNG, JPG, JPEG)
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
    filter_speckle=1,         # Loại bỏ nhiễu nhỏ
    color_precision=12,       # Độ chính xác màu cao
    layer_difference=12,      # Phân tách màu tốt hơn
    corner_threshold=50,      # Ngưỡng đường cong mượt
    length_threshold=1.5,     # Ngưỡng bắt chi tiết
    max_iterations=25,        # Số lần lặp hội tụ
    splice_threshold=40,      # Ngưỡng nối đường
    path_precision=12         # Độ chính xác đường cong
)
```

## Input/Output

### Input
- **Thư mục**: `data_prep/data/dirty_png/`
- **Định dạng**: PNG, JPG, JPEG (không phân biệt hoa thường)
- **Yêu cầu**: Ảnh có hoặc không có kênh alpha

### Output
- **PNG đã làm sạch**: `data_prep/data/clean_png/` - Ảnh đã tiền xử lý với alpha được làm mượt
- **File SVG**: `data_prep/data/svg/` - Mẫu thêu đã vector hóa

## Thách thức & Giải pháp

### Thách thức 1: Mất độ trong suốt
**Vấn đề**: Bản triển khai ban đầu ghép ảnh lên nền trắng, làm mất độ trong suốt.

**Giải pháp**: 
- Giữ nguyên format RGBA xuyên suốt pipeline
- Sử dụng các thao tác morphology thay vì ghép lên nền trắng
- Vtracer tự động xử lý độ trong suốt từ input RGBA

### Thách thức 2: Cạnh răng cưa
**Vấn đề**: Output vector hóa hiển thị cạnh răng cưa, pixelated dọc theo đường cong và viền.

**Giải pháp**:
- Thêm Gaussian blur vào kênh alpha trước khi vector hóa
- Điều chỉnh tham số vtracer để đường cong mượt hơn:
  - Tăng `path_precision` và `color_precision`
  - Điều chỉnh `corner_threshold` và `splice_threshold`
  - Cân bằng `filter_speckle` để loại bỏ nhiễu mà không mất chi tiết

### Thách thức 3: Giảm màu (Color Quantization)
**Vấn đề**: Bảng màu bị giảm gây banding và biểu diễn màu không chính xác.

**Giải pháp**:
- Tăng `color_precision` lên 12
- Tăng `layer_difference` lên 12
- Duy trì `colormode="color"` để giữ đầy đủ màu

## Cách sử dụng

```bash
sh run_prep.sh
```