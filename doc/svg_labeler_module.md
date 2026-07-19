# SVG Stitch Labeler Module

Module dùng để nhận một hoặc nhiều SVG và trả về SVG đã gắn nhãn `fill` / `satin`
cho từng `<path>`. Module này dùng lại rule trong
`scripts/inference/opencv_stitch_classifier.py`, nên khi rule OpenCV được cập
nhật thì API đóng gói cũng đi theo cùng logic.

## API chính

```python
from embroidery_labeler import LabelingConfig, label_svg_bytes, label_svg_file, label_svg_files
```

Module này được import từ root của repo. Nếu gọi từ service/web app bên ngoài,
hãy đảm bảo repo nằm trong `PYTHONPATH` hoặc được mount/copy vào project backend.

### Label một file SVG

```python
from embroidery_labeler import LabelingConfig, label_svg_file

config = LabelingConfig(
    fallback_physical_width_mm=80.0,
    enable_context_refinement=True,
)

result = label_svg_file(
    "input.svg",
    output_svg="input_labeled.svg",
    config=config,
)

print(result.counts)
print(result.labels)
```

### Label SVG upload dạng bytes

```python
from embroidery_labeler import label_svg_bytes

svg_bytes = uploaded_file.read()
result = label_svg_bytes(svg_bytes, source_name="upload.svg")

return result.svg_bytes
```

Khi trả HTTP response, nên dùng content type:

```text
image/svg+xml; charset=utf-8
```

### Label nhiều SVG upload dạng bytes

```python
from embroidery_labeler import label_svg_bytes

results = []
for uploaded_file in uploaded_files:
    results.append(
        label_svg_bytes(
            uploaded_file.read(),
            source_name=uploaded_file.filename,
        )
    )
```

### Label nhiều file

```python
from embroidery_labeler import label_svg_files

results = label_svg_files(
    ["a.svg", "b.svg"],
    output_dir="labeled_svgs",
)
```

## Output

Mỗi path được classify sẽ có thêm:

```xml
inkscape:label="satin"
data-stitch="satin"
data-label="satin"
```

hoặc:

```xml
inkscape:label="fill"
data-stitch="fill"
data-label="fill"
```

`LabelingResult` gồm:

- `svg_text`: SVG đã gắn label dạng string.
- `svg_bytes`: SVG đã gắn label dạng bytes, tiện để trả HTTP response.
- `labels`: map `{path_id: "fill" | "satin"}`.
- `counts`: thống kê số path fill/satin/unclassified.
- `output_path`: path đã ghi ra nếu dùng `output_svg`.

## Config thường dùng

```python
config = LabelingConfig(
    fallback_physical_width_mm=80.0,
    classify_scale=1400.0 / 4200.0,
    enable_context_refinement=True,
    use_existing_svg_labels=False,
    write_data_attrs=True,
    write_css_class=False,
)
```

- `fallback_physical_width_mm`: dùng khi SVG không khai báo width rõ ràng.
- `classify_scale`: scale rasterize để classify path; thấp hơn thì nhanh hơn,
  cao hơn thì chính xác hơn nhưng chậm hơn.
- `enable_context_refinement`: bật rule context chống chồng satin/fill sai.
- `use_existing_svg_labels`: nếu `True`, SVG có label sẵn sẽ được ưu tiên.
- `write_data_attrs`: ghi thêm `data-stitch` và `data-label`.
- `write_css_class`: ghi thêm class `stitch-fill` hoặc `stitch-satin`.

## Ghi chú tích hợp

- Cần chạy trong environment có các dependency hiện tại của repo:
  `opencv-python`, `numpy`, `pillow`, `cairosvg`, `scikit-image`.
- Với SVG upload dạng bytes, module tạo temp file nội bộ để CairoSVG render ổn
  định.
- Nếu SVG có tham chiếu file ngoài bằng path tương đối như image/font local,
  nên dùng `label_svg_file` để CairoSVG có base path đúng. Với upload bytes,
  backend cần tự xử lý hoặc inline các asset đó trước.
- Nếu SVG đã có label và muốn ưu tiên label cũ, bật:

```python
LabelingConfig(use_existing_svg_labels=True)
```

## Kiểm tra nhanh

```bash
venv/bin/python - <<'PY'
from embroidery_labeler import label_svg_file

result = label_svg_file("data/svg/logo/111.svg", "/tmp/111_labeled.svg")
print(result.counts)
print("/tmp/111_labeled.svg")
PY
```
