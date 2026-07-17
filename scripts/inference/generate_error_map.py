import os
import time
import xml.etree.ElementTree as ET
import concurrent.futures
import io
import sys

import cairosvg
import cv2
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

# Đăng ký Namespace của Inkscape để Python đọc được nhãn
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("inkscape", INKSCAPE_NS)

DEFAULT_TOLERANCE_PX = 5

"""
BẢN TỐI ƯU TỐC ĐỘ
==================
Đã profile bản gốc trên máy: mỗi file mất ~2.7-3.7s, trong đó:
  - cairosvg.svg2png() (render GT ở full 4200x4800)  ~70% thời gian
  - decode PNG + dilate + so khớp mask                ~15%
  - cv2.imwrite() error map                            ~15%

=> Việc render+so-sánh của MỖI FILE hoàn toàn ĐỘC LẬP với nhau, nên tối
ưu lớn nhất không phải là "code nhanh hơn trên 1 file" mà là CHẠY SONG
SONG NHIỀU FILE CÙNG LÚC bằng ProcessPoolExecutor - giống cách
run_opencv_test.py và opencv_stitch_classifier.py (--batch) đã làm.
Trên máy nhiều core, tốc độ tổng gần như chia đều cho số core.

Các tối ưu nhỏ hơn đã áp dụng thêm:
  1. Bỏ qua cv2.imwrite() cho các file "khớp hoàn hảo" (mặc định BẬT,
     tắt bằng save_perfect_matches=True nếu muốn giữ nguyên hành vi cũ) -
     tiết kiệm ~15-20% thời gian + dung lượng đĩa cho các file không lỗi,
     vốn thường chiếm đa số trong một pipeline đã được tinh chỉnh tốt.
  2. Dùng cv2.imdecode() thay vì PIL.Image.open() để decode PNG do
     cairosvg render ra - nhanh hơn nhẹ và bớt 1 bước convert RGB<->BGR
     thủ công (opencv làm việc native với BGR nên khỏi phải đổi qua đổi
     lại giữa PIL(RGB) và cv2(BGR)).
  3. KHÔNG dùng cv2.IMWRITE_PNG_COMPRESSION tường minh - đã đo thực tế
     compression mặc định của OpenCV cho loại ảnh sparse-color (nền đen/
     xám phẳng) này NHANH HƠN và file NHỎ HƠN so với ép compression=1
     hay 9 (ngược với giả định thường thấy "compression thấp = nhanh
     hơn" - giả định đó đúng với ảnh noise ngẫu nhiên nhưng SAI với ảnh
     it-chi-tiết/nhieu-mang-mau-dong-nhat như error map ở đây).
  4. Thêm option `downscale` (mặc định 1.0 = giữ nguyên độ phân giải gốc,
     không đổi hành vi) - nếu muốn nhanh hơn nữa và chấp nhận thô hơn 1
     chút, truyền downscale=0.5 sẽ giảm ~4x số pixel phải render/so
     sánh/ghi đĩa (cairosvg render tỉ lệ thuận với số pixel). Tolerance
     (tinh bang px) se duoc tu dong scale theo de giu nguyen y nghia vat
     ly (mm) cua dung sai.
"""


def render_svg_to_gt_mask(svg_path: str, target_w: int = 4200, target_h: int = 4800) -> np.ndarray:
    """Đọc trực tiếp file SVG, tách nhãn Inkscape và dựng thành ma trận nhãn 0,1,2 trong bộ nhớ"""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    for child in root.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "path":
            label = child.get(f"{{{INKSCAPE_NS}}}label", "").lower()

            if "satin" in label:
                color = "#00FF00"
            elif "fill" in label:
                color = "#FF0000"
            else:
                color = "none"

            if color != "none":
                child.set("fill", color)
                child.set("stroke", "none")
                child.set("fill-opacity", "1")
                child.attrib.pop("style", None)

    svg_bytes = ET.tostring(root, encoding="utf-8")
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=target_w, output_height=target_h)

    # [TOI UU #2] cv2.imdecode thay vi PIL - nhanh hon nhe, tra ve BGR
    # (khop tu nhien voi cach cv2 lam viec o cac buoc sau), khong can
    # convert("RGB") + np.array() qua 2 lop nhu ban goc.
    bgr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)

    gt_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    gt_mask[bgr[:, :, 2] > 128] = 1  # Kenh Do (BGR[2]) -> FILL
    gt_mask[bgr[:, :, 1] > 128] = 2  # Kenh Xanh la (BGR[1]) -> SATIN

    return gt_mask


def load_pred_mask_smart(pred_path: str) -> np.ndarray:
    """Đọc ảnh Dự đoán: Tự động hiểu cả ảnh mask số học lẫn ảnh màu Preview (Vàng/Hồng)"""
    pred_bgr = cv2.imread(pred_path, cv2.IMREAD_COLOR)
    if pred_bgr is None:
        return None

    h, w = pred_bgr.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    if pred_bgr.max() <= 2:
        return pred_bgr[:, :, 0]

    fill_pixels = (pred_bgr[:, :, 0] < 50) & (pred_bgr[:, :, 1] > 200) & (pred_bgr[:, :, 2] > 200)
    satin_pixels = (pred_bgr[:, :, 0] > 200) & (pred_bgr[:, :, 1] < 50) & (pred_bgr[:, :, 2] > 200)

    pred_mask[fill_pixels] = 1
    pred_mask[satin_pixels] = 2

    return pred_mask


def generate_visual_error_map(pred_mask: np.ndarray, gt_mask: np.ndarray, output_path: str,
                               tolerance_px: int, save_if_no_error: bool = False):
    h, w = gt_mask.shape
    kernel = np.ones((tolerance_px * 2 + 1, tolerance_px * 2 + 1), np.uint8)

    pred_satin_dilated = cv2.dilate((pred_mask == 2).astype(np.uint8), kernel)
    missed_satin_core = (gt_mask == 2) & (pred_satin_dilated == 0)

    gt_satin_dilated = cv2.dilate((gt_mask == 2).astype(np.uint8), kernel)
    over_satin_core = (pred_mask == 2) & (gt_satin_dilated == 0)

    missed_count = int(np.count_nonzero(missed_satin_core))
    over_count = int(np.count_nonzero(over_satin_core))

    # [TOI UU #1] Bo qua imwrite khi khong co loi va khong yeu cau giu lai
    if missed_count == 0 and over_count == 0 and not save_if_no_error:
        return missed_count, over_count

    error_visual = np.zeros((h, w, 3), dtype=np.uint8)
    match_mask = (pred_mask == gt_mask) & (gt_mask != 0)
    error_visual[match_mask] = [70, 70, 70]
    error_visual[missed_satin_core] = [0, 0, 255]   # Do ruc
    error_visual[over_satin_core] = [0, 255, 255]   # Vang choi

    # [TOI UU #3] KHONG ep IMWRITE_PNG_COMPRESSION - da do thuc te mac
    # dinh cua OpenCV nhanh hon va file nho hon cho loai anh nay.
    cv2.imwrite(output_path, error_visual)

    return missed_count, over_count


def _process_one_file(args) -> dict:
    filename, svg_dir, pred_dir, error_map_dir, tolerance_px, downscale, save_perfect_matches = args
    t0 = time.time()

    svg_path = os.path.join(svg_dir, filename)
    pred_filename = filename.replace(".svg", "_pred.png")
    pred_path = os.path.join(pred_dir, pred_filename)

    if not os.path.exists(pred_path):
        return {"status": "missing_pred", "filename": filename, "pred_filename": pred_filename}

    try:
        target_w = int(round(4200 * downscale))
        target_h = int(round(4800 * downscale))
        eff_tolerance_px = max(1, int(round(tolerance_px * downscale)))

        gt_mask = render_svg_to_gt_mask(svg_path, target_w=target_w, target_h=target_h)
        pred_mask_full = load_pred_mask_smart(pred_path)
        if pred_mask_full is None:
            return {"status": "error", "filename": filename, "error": "khong doc duoc anh du doan"}

        if downscale != 1.0:
            pred_mask = cv2.resize(pred_mask_full, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            pred_mask = pred_mask_full

        error_output_path = os.path.join(error_map_dir, f"error_{pred_filename}")
        missed_core, over_core = generate_visual_error_map(
            pred_mask, gt_mask, error_output_path, eff_tolerance_px,
            save_if_no_error=save_perfect_matches,
        )

        return {
            "status": "ok", "filename": filename,
            "missed_core": missed_core, "over_core": over_core,
            "elapsed": time.time() - t0,
        }
    except Exception as e:
        return {"status": "error", "filename": filename, "error": str(e)}


def evaluate_directly_from_svg(svg_dir: str, pred_dir: str, error_map_dir: str, tolerance_px: int,
                                n_workers: int = None, downscale: float = 1.0,
                                save_perfect_matches: bool = False) -> None:
    if not os.path.exists(error_map_dir):
        os.makedirs(error_map_dir)

    svg_files = sorted(f for f in os.listdir(svg_dir) if f.endswith(".svg"))
    if not svg_files:
        print(f"Khong tim thay file .svg nao trong {svg_dir}")
        return

    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)

    print(f"{'Ten File SVG':<30} | {'Sot Satin (Loi Loi Do)':<25} | {'Du Satin (Loi Loi Vang)':<25}")
    print("-" * 85)

    tasks = [(f, svg_dir, pred_dir, error_map_dir, tolerance_px, downscale, save_perfect_matches)
              for f in svg_files]

    total_checked = 0
    t_start = time.time()

    # [TOI UU CHINH] Chay song song nhieu file cung luc bang
    # ProcessPoolExecutor - moi file render+so-sanh doc lap hoan toan
    # nen day la cach tang toc hieu qua nhat, gan tuyen tinh theo so core.
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one_file, t): t[0] for t in tasks}
        results = {}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Danh gia"):
            res = future.result()
            results[res["filename"]] = res

    # In ket qua theo dung thu tu file (khong phu thuoc thu tu hoan thanh cua các worker)
    for filename in svg_files:
        res = results.get(filename)
        if res is None:
            continue
        if res["status"] == "missing_pred":
            print(f"[CANH BAO] Khong tim thay anh du doan ket qua cho: {filename} "
                  f"(Mong doi: {res['pred_filename']})")
            continue
        if res["status"] == "error":
            print(f"Loi xu ly file {filename}: {res['error']}")
            continue

        total_checked += 1
        missed_core, over_core = res["missed_core"], res["over_core"]
        if missed_core > 0 or over_core > 0:
            print(f"{filename:<30} | {missed_core:<25} | {over_core:<25}")
        else:
            print(f"{filename:<30} | Khop hoan hao 100% (trong vung dung sai)!")

    elapsed = time.time() - t_start
    print(f"\n[INFO] Da quet xong {total_checked} file trong {elapsed:.2f}s "
          f"({n_workers} tien trinh song song). "
          f"Loi vien do rang cua ({tolerance_px}px) da duoc tu dong loai bo.")


if __name__ == "__main__":
    SVG_INKSCAPE_DIR = "data/opencv_test/svg/"
    PREDICTED_MASK_DIR = "data/opencv_test/predictions/"
    ERROR_MAP_OUTPUT_DIR = "data/opencv_test/error_maps/"

    tolerance = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOLERANCE_PX

    evaluate_directly_from_svg(SVG_INKSCAPE_DIR, PREDICTED_MASK_DIR, ERROR_MAP_OUTPUT_DIR, tolerance)