import os
import time
import xml.etree.ElementTree as ET
import concurrent.futures
import sys
from pathlib import Path

import cairosvg
import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("inkscape", INKSCAPE_NS)

DEFAULT_TOLERANCE_PX = 5

# --- SUA: neo duong dan theo VI TRI FILE NAY, khong phu thuoc thu muc
# ban dang dung khi go `python generate_error_map.py` (VD chay tu VSCode
# "Run" hay tu terminal da `cd` vao scripts/inference/ deu cho cung 1
# ket qua - truoc day duong dan la relative theo cwd nen 2 truong hop
# nay ra 2 thu muc KHAC NHAU, day rat co the la ly do "khong thay file
# luu vao dung cho ban dang mo".
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/inference/ -> len 2 cap la root du an


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

    bgr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)

    gt_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    gt_mask[bgr[:, :, 2] > 128] = 1  # Kenh Do (BGR[2]) -> FILL
    gt_mask[bgr[:, :, 1] > 128] = 2  # Kenh Xanh la (BGR[1]) -> SATIN

    return gt_mask


def load_pred_mask_smart(pred_path: str) -> np.ndarray:
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

    if missed_count == 0 and over_count == 0 and not save_if_no_error:
        return missed_count, over_count, False  # <-- THÊM: co bit bao "co ghi file hay khong"

    error_visual = np.zeros((h, w, 3), dtype=np.uint8)
    match_mask = (pred_mask == gt_mask) & (gt_mask != 0)
    error_visual[match_mask] = [70, 70, 70]
    error_visual[missed_satin_core] = [0, 0, 255]
    error_visual[over_satin_core] = [0, 255, 255]

    ok = cv2.imwrite(output_path, error_visual)
    if not ok:
        # --- SUA: truoc day khong kiem tra gia tri tra ve cua imwrite,
        # neu ghi that bai se im lang khong bao gi, tuong nham la "khong
        # co loi nen khong can luu". Gio bao ro ra terminal.
        print(f"[LOI GHI FILE] cv2.imwrite tra ve False cho: {output_path}")

    return missed_count, over_count, ok


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
        missed_core, over_core, file_written = generate_visual_error_map(
            pred_mask, gt_mask, error_output_path, eff_tolerance_px,
            save_if_no_error=save_perfect_matches,
        )

        return {
            "status": "ok", "filename": filename,
            "missed_core": missed_core, "over_core": over_core,
            "file_written": file_written,  # <-- THEM
            "elapsed": time.time() - t0,
        }
    except Exception as e:
        return {"status": "error", "filename": filename, "error": str(e)}


def evaluate_directly_from_svg(svg_dir: str, pred_dir: str, error_map_dir: str, tolerance_px: int,
                                n_workers: int = None, downscale: float = 1.0,
                                save_perfect_matches: bool = False,
                                clear_before_run: bool = True) -> None:
    error_map_dir_abs = os.path.abspath(error_map_dir)
    if not os.path.exists(error_map_dir):
        os.makedirs(error_map_dir)

    # --- THEM: don sach error_*.png CU truoc khi chay, neu khong thu muc
    # se tich luy rac tu nhung lan chay truoc (model cu, tolerance cu, bo
    # SVG cu...) khien so file trong thu muc KHONG con phan anh dung ket
    # qua cua lan chay hien tai (VD: 100 file cu trong khi lan nay chi
    # thuc su phat hien 7 loi). Tat bang clear_before_run=False neu muon
    # GIU LAI cac lan chay truoc de so sanh (VD: dat ten thu muc theo
    # ngay/gio thay vi xoa).
    if clear_before_run:
        old_files = [f for f in os.listdir(error_map_dir) if f.startswith("error_") and f.endswith(".png")]
        for f in old_files:
            os.remove(os.path.join(error_map_dir, f))
        if old_files:
            print(f"[INFO] Da xoa {len(old_files)} file error map CU truoc khi chay lai "
                  f"(tat bang clear_before_run=False neu khong muon xoa).")

    svg_files = sorted(f for f in os.listdir(svg_dir) if f.endswith(".svg"))
    if not svg_files:
        print(f"Khong tim thay file .svg nao trong {os.path.abspath(svg_dir)}")
        return

    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)

    print(f"[INFO] Thu muc GHI ERROR MAP (duong dan tuyet doi): {error_map_dir_abs}")
    print(f"{'Ten File SVG':<30} | {'Sot Satin (Loi Loi Do)':<25} | {'Du Satin (Loi Loi Vang)':<25}")
    print("-" * 85)

    tasks = [(f, svg_dir, pred_dir, error_map_dir, tolerance_px, downscale, save_perfect_matches)
              for f in svg_files]

    total_checked = 0
    n_perfect_skipped = 0
    n_files_written = 0
    t_start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one_file, t): t[0] for t in tasks}
        results = {}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Danh gia"):
            res = future.result()
            results[res["filename"]] = res

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
        if res["file_written"]:
            n_files_written += 1
        elif missed_core == 0 and over_core == 0:
            n_perfect_skipped += 1

        if missed_core > 0 or over_core > 0:
            print(f"{filename:<30} | {missed_core:<25} | {over_core:<25}")
        else:
            print(f"{filename:<30} | Khop hoan hao 100% (trong vung dung sai)! -> KHONG ghi file "
                  f"(save_perfect_matches=False)")

    elapsed = time.time() - t_start

    # --- THEM: bao cao ro rang giup tu chan doan ngay lan sau, khong can
    # doan mo hinh do bug hay do thiet ke.
    n_actual_files_on_disk = len([f for f in os.listdir(error_map_dir) if f.startswith("error_")])
    print(f"\n[INFO] Da quet xong {total_checked} file trong {elapsed:.2f}s "
          f"({n_workers} tien trinh song song). "
          f"Loi vien do rang cua ({tolerance_px}px) da duoc tu dong loai bo.")
    print(f"[INFO] So file THUC SU duoc ghi ra dia: {n_files_written}")
    print(f"[INFO] So file BI BO QUA vi khop hoan hao (khong ghi, dung thiet ke): {n_perfect_skipped}")
    print(f"[INFO] Dem thuc te trong thu muc ({error_map_dir_abs}): {n_actual_files_on_disk} file error_*.png")
    if clear_before_run and n_actual_files_on_disk != n_files_written:
        print(f"[CANH BAO] Da bat clear_before_run nhung so file dem duoc ({n_actual_files_on_disk}) "
              f"KHONG khop voi so file vua ghi ({n_files_written}) — co the co tien trinh khac dang "
              f"ghi dong thoi vao cung thu muc nay, kiem tra lai.")
    if n_files_written == 0 and n_perfect_skipped == total_checked and total_checked > 0:
        print("[INFO] => Tat ca file deu khop hoan hao trong dung sai, nen thu muc error_maps "
              "TRONG LA DUNG NHU THIET KE, khong phai loi. Muon xem lai anh 'sach' cho tung file, "
              "goi ham voi save_perfect_matches=True.")


if __name__ == "__main__":
    SVG_INKSCAPE_DIR = str(_PROJECT_ROOT / "data/opencv_test/svg")
    PREDICTED_MASK_DIR = str(_PROJECT_ROOT / "data/opencv_test/predictions")
    ERROR_MAP_OUTPUT_DIR = str(_PROJECT_ROOT / "data/opencv_test/error_maps")

    tolerance = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOLERANCE_PX

    evaluate_directly_from_svg(SVG_INKSCAPE_DIR, PREDICTED_MASK_DIR, ERROR_MAP_OUTPUT_DIR, tolerance)