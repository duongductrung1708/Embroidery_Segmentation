# Giai đoạn 2 - Bước 1: Chuyển đổi PNG sang SVG

## Tổng quan

Bước này chuyển đổi các ảnh PNG đã phân đoạn từ mô hình U-2-Net sang định dạng vector SVG tương thích với máy thêu. Pipeline hiện tại tích hợp:
- fal.ai NAFNet để tăng chất lượng ảnh trước khi vector hóa
- Manual boolean cutout operations sử dụng shapely để tạo true single-layer SVG

## Mục đích

- Chuyển đổi ảnh PNG raster sang định dạng vector SVG
- Giữ nguyên độ trong suốt từ ảnh gốc
- Duy trì độ chính xác màu cho mẫu thêu
- Tạo đường dẫn sạch, mượt cho máy thêu
- Tăng chất lượng ảnh bằng AI (NAFNet) trước khi vector hóa
- Tạo true cutout SVG với zero overlap giữa các layers

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

**3. Manual Boolean Cutout (`StrictCutoutProcessor`)**

```python
class StrictCutoutProcessor:
    """Processes geometries sequentially from top to bottom to guarantee zero overlap."""
```

- **Top-to-Bottom Processing**: Xử lý từ layer trên cùng xuống dưới cùng
- **Cumulative Upper Mask**: Tích hợp dần các layer trên vào mask để trừ từ layer dưới
- **Zero Overlap Guarantee**: Đảm bảo mỗi coordinate space chỉ có 1 layer
- **Micro-Buffer**: Sử dụng buffer nhỏ (1e-8) để tránh slivers từ shared edges
- **Area Tolerance**: Lọc bỏ các phần có diện tích quá nhỏ (< 1e-5)

**4. High-Precision SVG Parsing (`HighPrecisionSVGParser`)**

```python
class HighPrecisionSVGParser:
    """High-precision SVG parser generating clean geometries."""
```

- **Multi-Element Support**: Parse rect, circle, ellipse, polygon, polyline, path
- **Transform Parsing**: Hỗ trợ translate và matrix transforms
- **Bezier Approximation**: 20-step approximation cho curves chính xác hơn
- **Geometry Cleaning**: Buffer operations để fix self-intersections

**5. Pipeline Vector hóa (`process_pipeline`)**

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
    hierarchical="stack",     # Xếp chồng các vùng màu (sau đó xử lý cutout thủ công)
    mode="spline",            # Đường cong Bezier mượt
    filter_speckle=2,         # Giảm ngưỡng để giữ nhiều chi tiết màu hơn
    color_precision=12,       # Độ chính xác màu cao
    layer_difference=4,       # Giảm thêm để gộp các vùng màu tương tự lại với nhau
    corner_threshold=20,      # Giảm ngưỡng góc để đường cong mượt hơn
    length_threshold=4.0,     # Tăng ngưỡng độ dài đường cong để loại bỏ các đoạn ngắn
    max_iterations=25,        # Số lần lặp hội tụ
    splice_threshold=20,      # Giảm ngưỡng nối đường để đường cong mượt hơn
    path_precision=12         # Độ chính xác đường cong
)
```

**6. Manual Cutout Conversion (`convert_stack_to_cutout`)**

Sau khi vtracer tạo SVG với `hierarchical="stack"`, pipeline áp dụng manual boolean operations:

```python
def convert_stack_to_cutout(svg_path: str):
    """Convert stacked SVG to cutout SVG by removing overlaps."""
```

- **Parse SVG**: Sử dụng `HighPrecisionSVGParser` để parse SVG paths sang Shapely geometries
- **Apply Cutout**: Sử dụng `StrictCutoutProcessor` để thực hiện boolean difference operations
- **Generate Output**: Export lại SVG với paths đã được đục lỗ (zero overlap)
- **Preserve Attributes**: Giữ nguyên viewBox và các attributes từ SVG gốc

## Input/Output

### Input
- **Thư mục**: `data/lineart/train/images` (hoặc custom path)
- **Định dạng**: PNG, JPG, JPEG (không phân biệt hoa thường)
- **Yêu cầu**: Ảnh có hoặc không có kênh alpha

### Output
- **PNG đã làm sạch**: `data/lineart/train/clean/` - Ảnh đã tiền xử lý với alpha sắc lẹm (binary)
- **File SVG**: `data/svg/logo/` - Mẫu thêu đã vector hóa

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
- Giữ `layer_difference=4` để phân tách vùng màu tốt
- Duy trì `colormode="color"` để giữ đầy đủ màu

### Thách thức 5: SVG Layers Overlap
**Vấn đề**: vtracer's built-in `hierarchical="cutout"` không đủ chính xác, vẫn còn overlap giữa layers.

**Giải pháp**:
- Sử dụng `hierarchical="stack"` để tạo layered SVG
- Áp dụng manual boolean operations với shapely
- `StrictCutoutProcessor` xử lý top-to-bottom với cumulative mask
- Đảm bảo zero overlap giữa các layers cho laser cutting/embroidery

## Cách sử dụng

### Cài đặt Dependencies

```bash
pip install opencv-python numpy vtracer fal-client requests pillow tqdm shapely
```

### Cấu hình API Key

```bash
export FAL_KEY="your-api-key-here"
```

### Chạy Pipeline

```bash
cd scripts/data_prep
python3 png_to_svg.py
```

## Dependencies

- `opencv-python` - Xử lý ảnh
- `numpy` - Thao tác mảng và transform matrices
- `vtracer` - Chuyển đổi raster sang vector
- `fal-client` - Client cho fal.ai API
- `requests` - HTTP requests để tải ảnh từ fal.ai
- `pillow` - Xử lý ảnh PIL
- `tqdm` - Thanh tiến trình
- `shapely` - Computational geometry cho boolean operations
- `cairosvg` - SVG rendering (optional)

## Các bước tiếp theo

Sau khi chuyển đổi SVG, giai đoạn tiếp theo bao gồm:
- Tối ưu hóa SVG cho máy thêu
- Đơn giản hóa đường dẫn và tạo mũi thêu
- Chuyổi đổi định dạng sang format riêng của máy (PES, DST, v.v.)

---

# Giai đoạn 2 - Bước 2: Chuyển đổi SVG sang PNG với Labels cho Training

## Tổng quan

Bước này chuyển đổi các file SVG có metadata (inkscape:label) sang ảnh PNG và label masks để huấn luyện model segmentation. Pipeline hỗ trợ:
- Đọc metadata từ `inkscape:label` attribute trong SVG paths
- Render SVG thành PNG với nền trong suốt
- Tạo label mask dựa trên stitch type (fill/satin)
- Hỗ trợ 3-class segmentation: background (0), fill (1), satin (2)

## Mục đích

- Tạo dataset training từ SVG có metadata
- Giữ nguyên kích thước gốc SVG
- Tạo label mask chính xác dựa trên alpha channel
- Hỗ trợ fine-tune model với dataset logo màu

## Triển khai

### File: `data_prep/svg_to_png_with_labels.py`

#### Các thành phần chính

**1. Parse SVG Metadata (`parse_svg_with_metadata`)**

```python
def parse_svg_with_metadata(svg_path: str) -> Tuple[ET.Element, Dict[str, str]]:
    """Parse SVG file and extract stitch type metadata from paths using inkscape:label."""
```

- Đọc `inkscape:label` attribute từ mỗi path
- Normalize metadata với `strip().lower()`
- Trả về dictionary mapping path_id → stitch_type

**2. Get SVG Dimensions (`get_svg_dimensions`)**

```python
def get_svg_dimensions(svg_path: str) -> Tuple[int, int]:
    """Get original SVG dimensions from viewBox or width/height attributes."""
```

- Đọc kích thước từ viewBox hoặc width/height attributes
- Hỗ trợ nhiều đơn vị (px, mm, cm, %) với regex
- Fallback về 512x512 nếu không tìm thấy

**3. Create Label Mask (`create_label_mask`)**

```python
def create_label_mask(svg_path: str, width: int, height: int, metadata: Dict[str, str]) -> np.ndarray:
    """Create label mask from SVG metadata by rendering paths in SVG order."""
```

- Render từng path theo thứ tự SVG để giữ đúng layer order
- Sử dụng RGBA alpha channel để xác định pixel thuộc path
- Map stitch type sang label: fill → 1, satin → 2
- Overwrite mask theo thứ tự SVG để xử lý vùng chồng lấp

**4. Render SVG to PNG (`render_svg_to_png`)**

```python
def render_svg_to_png(svg_path: str, output_path: str, width: int = None, height: int = None):
    """Render SVG to PNG image with transparent background."""
```

- Sử dụng cairosvg với `background_color=None` để giữ transparency
- Thêm `unsafe=True` để tăng tương thích với Inkscape SVG
- Giữ nguyên kích thước gốc nếu không chỉ định width/height

**5. Process SVG Folder (`process_svg_folder`)**

- Duyệt đệ quy thư mục với `rglob("*.svg")`
- Tạo dataset từ SVG với metadata
- Lưu images và masks vào thư mục output

## Input/Output

### Input
- **Thư mục**: `data/svg/logo/` (hoặc custom path)
- **Định dạng**: SVG với `inkscape:label` attribute
- **Metadata**: `inkscape:label="fill"` hoặc `inkscape:label="satin"`

### Output
- **PNG Images**: `data/logo/easy/images/` - Ảnh SVG đã render với nền trong suốt
- **Label Masks**: `data/logo/easy/masks/` - Mask grayscale với 3 giá trị (0, 1, 2)

## Label Mapping

- **0**: Background (trong suốt)
- **1**: Fill stitch
- **2**: Satin stitch

## Cách sử dụng

### Cài đặt Dependencies

```bash
pip install opencv-python numpy xml.etree.ElementTree cairosvg pillow tqdm
```

### Chạy Pipeline

```bash
cd scripts/data_prep
python3 svg_to_png_with_labels.py
```

Hoặc với custom paths:

```bash
python3 svg_to_png_with_labels.py --svg-dir path/to/svg --output-img path/to/images --output-mask path/to/masks
```

### Tham số

- `--svg-dir`: Thư mục chứa SVG (default: `data/svg/logo`)
- `--output-img`: Thư mục output images (default: `data/logo/easy/images`)
- `--output-mask`: Thư mục output masks (default: `data/logo/easy/masks`)
- `--width`: Kích thước output width (default: original SVG size)
- `--height`: Kích thước output height (default: original SVG size)

## Training với Dataset Logo

### Phiên bản Train

**1. train.py - Dataset Line-art (2-class)**
- Dataset: `data/lineart/train/images`, `data/lineart/train/masks`
- Model: `U2NET(in_ch=1, out_ch=2)`
- Labels: 0=background, 1=fill
- Checkpoint: `checkpoints/lineart/u2net_last.pth`, `checkpoints/lineart/u2net_best.pth`

**2. train_logo.py - Dataset Logo (3-class)**
- Dataset: `data/logo/easy/images`, `data/logo/easy/masks`
- Model: `U2NET(in_ch=1, out_ch=3)`
- Labels: 0=background, 1=fill, 2=satin
- Checkpoint: `checkpoints/logo/u2net_logo_last.pth`, `checkpoints/logo/u2net_logo_best.pth`
- Sử dụng: `dataset_logo.py`, `utils_logo.py`

### Chạy Training

```bash
# Train line-art (2-class)
python scripts/training/train.py

# Train logo (3-class)
python scripts/training/train_logo.py
```

## Ghi chú

**SVG to PNG Pipeline:**
- Pipeline render từng path theo thứ tự SVG để giữ đúng layer order
- Alpha channel được sử dụng thay vì grayscale threshold để độ chính xác cao hơn
- Model 3-class train từ đầu, không load checkpoint từ model 2-class
- Learning rate khuyến nghị: 1e-4 hoặc 5e-5 cho fine-tune

**PNG to SVG Pipeline:**
- Thời gian xử lý phụ thuộc vào độ phức tạp và kích thước ảnh
- fal.ai NAFNet thêm khoảng 20-30s mỗi ảnh nhưng cải thiện chất lượng đáng kể
- Alpha channel binary giúp file SVG nhẹ hơn nhiều so với gradient alpha
- Nếu fal.ai không phản hồi, pipeline tự động fallback về ảnh gốc
- Manual boolean cutout đảm bảo zero overlap giữa layers cho laser cutting/embroidery
- Pipeline tự động xử lý từ dirty_png → clean_png → svg với cutout