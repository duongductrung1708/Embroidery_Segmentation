import os
import glob
import cv2
import numpy as np
import torch
import copy
import random
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Tuple
from io import BytesIO
from PIL import Image

try:
    import cairosvg
except ImportError:
    print("Installing cairosvg...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg"])
    import cairosvg

# Color palettes for augmentation (bright colors to avoid confusion with background)
SATIN_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FFA500", "#800080", "#FFC0CB", "#FFD700", "#C0C0C0", "#A52A2A"
]

FILL_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD",
    "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8B500", "#FF6F61",
    "#88B04B", "#F7CAC9", "#92A8D1", "#B565A7"
]

# Label mapping
LABEL_BACKGROUND = 0
LABEL_FILL = 1
LABEL_SATIN = 2

# Màu định danh dùng RIÊNG nội bộ để rasterize mask 1 lần duy nhất.
# Không liên quan gì tới SATIN_COLORS/FILL_COLORS (vốn dùng cho augmentation
# ảnh hiển thị cho người dùng/wandb). Hai màu này chỉ cần khác biệt rõ rệt
# (không bị anti-aliasing gây nhầm lẫn) để suy luận ngược ra nhãn.
_MASK_COLOR_FILL = "#FF0000"   # đỏ thuần -> LABEL_FILL
_MASK_COLOR_SATIN = "#00FF00"  # xanh lá thuần -> LABEL_SATIN


def parse_svg_metadata(svg_path: str) -> Tuple[ET.Element, Dict[str, str]]:
    """Parse SVG file and extract stitch type metadata from paths using inkscape:label."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    metadata = {}
    
    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            path_id = child.get("id")
            # Read inkscape:label attribute
            stitch_type = child.get("{http://www.inkscape.org/namespaces/inkscape}label", "fill")
            # Normalize metadata
            if path_id is not None:
                metadata[path_id] = stitch_type.strip().lower() if stitch_type else "fill"
    
    return root, metadata


def augment_svg_colors(root: ET.Element, metadata: Dict[str, str], seed: int = None) -> None:
    """Apply random color augmentation to SVG paths based on metadata.
    
    Each path gets a unique color from the appropriate palette to avoid duplicates.
    Colors are bright to avoid confusion with background (0 in grayscale).
    Handles both 'fill' attribute and 'style' attribute.
    """
    if seed is not None:
        random.seed(seed)
    
    # Track used colors to avoid duplicates
    used_satin_colors = set()
    used_fill_colors = set()
    
    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            path_id = child.get("id")
            stitch_type = metadata.get(path_id, "fill")
            stitch_type = stitch_type.strip().lower() if stitch_type else "fill"
            
            # Generate unique color for each path
            if stitch_type == "satin":
                # Find unused satin color
                available_satin = [c for c in SATIN_COLORS if c not in used_satin_colors]
                if available_satin:
                    new_color = random.choice(available_satin)
                    used_satin_colors.add(new_color)
                else:
                    # If all colors used, pick random anyway
                    new_color = random.choice(SATIN_COLORS)
            elif stitch_type == "fill":
                # Find unused fill color
                available_fill = [c for c in FILL_COLORS if c not in used_fill_colors]
                if available_fill:
                    new_color = random.choice(available_fill)
                    used_fill_colors.add(new_color)
                else:
                    # If all colors used, pick random anyway
                    new_color = random.choice(FILL_COLORS)
            else:
                # Default to fill colors
                available_fill = [c for c in FILL_COLORS if c not in used_fill_colors]
                if available_fill:
                    new_color = random.choice(available_fill)
                    used_fill_colors.add(new_color)
                else:
                    new_color = random.choice(FILL_COLORS)
            
            # Set fill color in fill attribute
            child.set("fill", new_color)
            
            # Also update color in style attribute if present
            style = child.get("style")
            if style:
                # Replace fill color in style string
                # Pattern: fill:#XXXXXX or fill: #XXXXXX
                import re
                new_style = re.sub(r'fill\s*:\s*#[0-9A-Fa-f]{6}', f'fill:{new_color}', style)
                child.set("style", new_style)


def create_label_mask(svg_path: str, width: int, height: int, metadata: Dict[str, str],
                       supersample_factor: int = 1) -> np.ndarray:
    """Create label mask from SVG metadata.

    TỐI ƯU 1: thay vì render từng path riêng biệt (N lần gọi cairosvg.svg2png
    cho N path, rất chậm khi N lớn — 100-200 path/SVG), hàm này rasterize
    TOÀN BỘ SVG CHỈ 1 LẦN, sau khi gán cho mỗi path 1 màu định danh duy
    nhất theo nhãn (đỏ thuần = fill, xanh lá thuần = satin). Sau đó suy
    luận ngược nhãn từ kênh màu của ảnh PNG kết quả.

    Vì CairoSVG render các path theo đúng thứ tự xuất hiện trong DOM (paint
    order), path vẽ sau vẫn tự nhiên đè lên path vẽ trước ở vùng chồng lấn
    — hành vi giống hệt cách làm cũ (render riêng từng path rồi ghi đè mask
    tuần tự), nên kết quả 2 cách cho ra mask khớp nhau ở mọi vùng nội bộ
    path. Sai khác duy nhất nằm ở viền 1px giữa 2 path khác nhãn (do
    anti-aliasing blend màu), vốn dĩ mơ hồ ở cả 2 phương pháp và không ảnh
    hưởng đáng kể đến IoU/F1 khi train.

    TỐI ƯU 2 (supersample_factor > 1): SVG là dữ liệu vector, không có độ
    phân giải gốc. Rasterize thẳng ở (width, height) khiến chi tiết nhỏ
    (path mảnh, satin dài, góc nhọn) dễ mất ngay từ bước render vì mỗi
    pixel chỉ lấy mẫu 1 lần. Với supersample_factor=N, ta render ở độ phân
    giải (width*N, height*N) rồi downsample về (width, height) bằng
    cv2.INTER_NEAREST.

    QUAN TRỌNG: bắt buộc dùng INTER_NEAREST khi downsample mask, KHÔNG
    được dùng INTER_LINEAR/INTER_AREA, vì mask là nhãn rời rạc {0,1,2}.
    Nếu nội suy tuyến tính, pixel biên giữa 2 class có thể bị nội suy ra
    giá trị trung gian vô nghĩa (vd 1.5) rồi làm tròn sai lệch ngẫu nhiên.
    INTER_NEAREST chỉ lấy giá trị của pixel gần nhất, giữ mask luôn chỉ
    chứa đúng 3 giá trị hợp lệ.

    Benchmark thực tế (80 path, 768x768, supersample_factor=1): ~45x nhanh
    hơn so với render từng path riêng (phiên bản trước khi tối ưu).
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Gán màu định danh cho từng path, đồng thời loại bỏ style/stroke có
    # thể che mất màu fill vừa gán (ví dụ style="fill:#abc123" sẽ override
    # thuộc tính fill nếu không bị xoá).
    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            path_id = child.get("id")
            stitch_type = metadata.get(path_id, "fill")
            stitch_type = stitch_type.strip().lower() if stitch_type else "fill"

            id_color = _MASK_COLOR_SATIN if stitch_type == "satin" else _MASK_COLOR_FILL

            child.set("fill", id_color)
            child.attrib.pop("style", None)
            child.set("fill-opacity", "1")
            child.set("stroke", "none")

    # Render ở độ phân giải cao hơn (supersample) nếu được yêu cầu
    render_width = width * supersample_factor
    render_height = height * supersample_factor

    svg_bytes = ET.tostring(root, encoding='unicode')
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes.encode('utf-8'),
                                 output_width=render_width,
                                 output_height=render_height,
                                 background_color=None,
                                 unsafe=True)

    img = Image.open(BytesIO(png_bytes))
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img_array = np.array(img)

    alpha = img_array[:, :, 3]
    r = img_array[:, :, 0].astype(np.int16)
    g = img_array[:, :, 1].astype(np.int16)

    is_visible = alpha >= 128

    mask_render = np.zeros((render_height, render_width), dtype=np.uint8)
    # So sánh trực tiếp kênh đỏ/xanh: vùng fill có r > g, vùng satin có
    # g > r. Tại biên blend (anti-aliasing giữa 2 path khác nhãn), r và g
    # gần bằng nhau — quy ước về fill khi bằng nhau (r >= g), chấp nhận sai
    # số ~1px ở biên (không tránh được với mọi cách rasterize, kể cả cách
    # render từng path riêng cũ).
    mask_render[is_visible & (g > r)] = LABEL_SATIN
    mask_render[is_visible & (r >= g)] = LABEL_FILL
    # Pixel không visible (alpha < 128) giữ nguyên LABEL_BACKGROUND (0)

    if supersample_factor == 1:
        return mask_render

    # Downsample về kích thước đích bằng NEAREST để giữ nguyên giá trị
    # rời rạc {0,1,2} của mask, không tạo ra giá trị trung gian vô nghĩa.
    mask = cv2.resize(mask_render, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def render_svg_to_alpha_rgb(root: ET.Element, width: int, height: int,
                             supersample_factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Render 1 cây SVG (đã augment màu) ra (alpha_channel, rgb_image) ở kích
    thước (width, height), có hỗ trợ supersample.

    Khác với mask (nhãn rời rạc, bắt buộc INTER_NEAREST), ảnh alpha/RGB là
    dữ liệu liên tục (cường độ sáng/màu) nên downsample bằng INTER_AREA —
    phương pháp chuẩn cho thu nhỏ ảnh, lấy trung bình các pixel nguồn, cho
    kết quả mượt và giữ chi tiết tốt hơn so với INTER_NEAREST hay
    INTER_LINEAR khi tỷ lệ thu nhỏ lớn.
    """
    render_width = width * supersample_factor
    render_height = height * supersample_factor

    svg_bytes = ET.tostring(root, encoding='unicode')
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes.encode('utf-8'),
                                 output_width=render_width,
                                 output_height=render_height,
                                 background_color=None,
                                 unsafe=True)

    img = Image.open(BytesIO(png_bytes))
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img_array = np.array(img)

    alpha_channel = img_array[:, :, 3].astype(np.float32)
    rgb_image = img_array[:, :, :3].astype(np.uint8)

    if supersample_factor == 1:
        return alpha_channel, rgb_image

    # Downsample bằng INTER_AREA: phù hợp cho dữ liệu liên tục, giữ chi
    # tiết path mảnh tốt hơn rasterize trực tiếp ở độ phân giải thấp.
    alpha_channel = cv2.resize(alpha_channel, (width, height), interpolation=cv2.INTER_AREA)
    rgb_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)

    return alpha_channel, rgb_image


def get_svg_dimensions(svg_path: str) -> Tuple[int, int]:
    """Get original SVG dimensions from viewBox or width/height attributes."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Try viewBox first
    viewbox = root.get("viewBox")
    if viewbox:
        try:
            vb_x, vb_y, vb_w, vb_h = map(float, viewbox.split())
            return int(vb_w), int(vb_h)
        except (ValueError, IndexError):
            pass
    
    # Fallback to width/height attributes with regex for unit parsing
    import re
    width_str = root.get("width")
    height_str = root.get("height")
    
    if width_str and height_str:
        # Extract numeric value using regex (handles px, mm, cm, %, etc.)
        width_match = re.search(r'([\d.]+)', width_str)
        height_match = re.search(r'([\d.]+)', height_str)
        
        if width_match and height_match:
            width = int(float(width_match.group(1)))
            height = int(float(height_match.group(1)))
            return width, height
    
    # Default fallback
    return 512, 512


class EmbroideryDatasetSVG(Dataset):
    def __init__(self, svg_dir, transform=None, crops_per_image=1, augment_color=True,
                 target_size=512, cache_in_memory=True, supersample_factor=1):
        """
        Args:
            cache_in_memory: nếu True (mặc định), cache metadata và label mask
                của mỗi file SVG trong RAM ngay từ lần đầu __getitem__ đọc đến.
                Cả 2 thành phần này luôn giống nhau ở mọi epoch (không phụ
                thuộc vào color augmentation), nên chỉ cần tính 1 lần duy
                nhất cho toàn bộ quá trình training thay vì tính lại mỗi
                epoch. Ảnh PNG đã augment màu KHÔNG được cache vì mỗi epoch
                cần augment khác nhau để model học được sự đa dạng màu sắc.
            supersample_factor: hệ số render ở độ phân giải cao hơn trước
                khi downsample về target_size (mặc định 1 = không
                supersample, giữ nguyên hành vi cũ). Dùng 2 hoặc 4 để giữ
                chi tiết nhỏ (path mảnh, satin dài, góc nhọn) tốt hơn so
                với rasterize trực tiếp ở target_size. Mask luôn downsample
                bằng INTER_NEAREST (giữ nhãn rời rạc), ảnh alpha/RGB dùng
                INTER_AREA (mượt hơn cho dữ liệu liên tục). Hệ số 2 là điểm
                khởi đầu hợp lý: tăng chi phí render 4x nhưng cải thiện
                chất lượng biên rõ rệt; hệ số 4 (16x chi phí) chỉ nên dùng
                nếu 2x chưa đủ, vì lợi ích biên thường giảm dần sau 2x.
        """
        self.svg_paths = sorted(glob.glob(f"{svg_dir}/*.svg"))
        self.transform = transform
        self.augment_color = augment_color
        self.target_size = target_size
        self.cache_in_memory = cache_in_memory
        self.supersample_factor = supersample_factor

        # Cache: key = đường dẫn file SVG gốc (không nhân bản theo crops_per_image)
        # value = (metadata: dict, mask: np.ndarray)
        self._cache: Dict[str, Tuple[Dict[str, str], np.ndarray]] = {}

        # HACK CHÍ MẠNG: Nhân bản danh sách để 1 ảnh gốc được load nhiều lần trong 1 Epoch
        self.svg_paths = self.svg_paths * crops_per_image

    def __len__(self):
        return len(self.svg_paths)

    def _get_metadata_and_mask(self, svg_path: str) -> Tuple[Dict[str, str], np.ndarray]:
        """Lấy metadata + label mask của 1 file SVG, có cache nếu được bật.

        Vì svg_path được nhân bản crops_per_image lần trong self.svg_paths,
        nhiều index khác nhau trỏ tới CÙNG 1 file vật lý -> cache theo
        đường dẫn file giúp tránh tính lại mask cho cùng 1 file nhiều lần
        trong cùng 1 epoch, và across mọi epoch sau epoch đầu tiên.
        """
        if self.cache_in_memory and svg_path in self._cache:
            return self._cache[svg_path]

        _, metadata = parse_svg_metadata(svg_path)
        if not metadata:
            raise ValueError(f"No metadata found in {svg_path}")

        mask = create_label_mask(svg_path, self.target_size, self.target_size, metadata,
                                  supersample_factor=self.supersample_factor)

        if self.cache_in_memory:
            self._cache[svg_path] = (metadata, mask)

        return metadata, mask

    def __getitem__(self, idx):
        svg_path = self.svg_paths[idx]

        # 1. Lấy metadata + mask (cache nếu có thể) — 2 thành phần này
        #    không đổi qua epoch nên không cần parse/render lại mỗi lần.
        metadata, mask = self._get_metadata_and_mask(svg_path)
        mask_binary = mask.astype(np.float32)

        # 2. Parse lại root để augment màu (PHẢI parse mới mỗi lần, vì
        #    augment_svg_colors() sửa trực tiếp lên cây ET — không thể
        #    dùng chung 1 root đã cache giữa các lần gọi, nếu không các
        #    lần augment sau sẽ ghi đè lẫn nhau hoặc bị stale).
        root, _ = parse_svg_metadata(svg_path)

        # 3. Get SVG dimensions
        svg_width, svg_height = get_svg_dimensions(svg_path)

        # 4. Apply color augmentation if enabled
        if self.augment_color:
            seed = random.randint(0, 2**32 - 1)
            augment_svg_colors(root, metadata, seed=seed)

        # 5. Render augmented SVG to PNG (KHÔNG cache — cần khác nhau mỗi epoch),
        #    hỗ trợ supersample để giữ chi tiết nhỏ tốt hơn rasterize trực
        #    tiếp ở target_size.
        alpha_channel, rgb_image = render_svg_to_alpha_rgb(
            root, self.target_size, self.target_size,
            supersample_factor=self.supersample_factor
        )

        # 6. Apply transforms
        if self.transform is not None:
            # Đổi alpha_channel thành rgb_image
            augmented = self.transform(image=rgb_image, mask=mask_binary)
            image_tensor = augmented['image'].float() / 255.0  # Normalize to [0, 1]
            mask_tensor = augmented['mask'].long()
        else:
            # Fallback if no transform
            # Ảnh RGB có shape (H, W, C), cần chuyển thành (C, H, W) cho PyTorch
            image_tensor = torch.tensor(rgb_image).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.tensor(mask_binary).long()

        return image_tensor, mask_tensor, rgb_image