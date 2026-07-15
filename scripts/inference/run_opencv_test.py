#!/usr/bin/env python3
"""
run_opencv_test.py
- Chay batch + danh gia dung theo cau truc thu muc data/opencv_test/
- Doan nhan GT da tang (Hierarchical Guessing).
- TICH HOP DO THOI GIAN & THANH TIEN TRINH (tqdm).
- CHAY SONG SONG NHIEU ANH CUNG LUC (multiprocessing).
- LOG LEN WEIGHTS & BIASES (wandb): Ghep 3 anh thanh 1 cot duy nhat tren RAM.
- LỌC RÁC: Xóa các hạt nhiễu < 50px trước khi chấm điểm.
- CHI SO CHINH: PIXEL-LEVEL IOU & F1 (Do chinh xac tren tung diem anh).
- DU DOAN TRUC TIEP TREN FILE SVG (classify_svg) thay vi tren anh PNG
  da render + gop mau. Anh raster (neu co) chi con duoc dung de hien thi cot
  "Goc" trong bang so sanh, khong con anh huong den ket qua du doan.

===============================================================================
TOI UU TOC DO (SPEEDUP PATCH):
- classify_svg (trong opencv_stitch_classifier.py) gio chay phan loai hinh
  hoc (quyet dinh satin/fill) tren 1 canvas THU NHO cho nhanh, nhung anh
  label_mask CUOI CUNG van duoc RENDER VECTOR 1 LAN DUY NHAT o full-res
  (giong cach lam Ground-Truth) - nen bien hinh van MUOT, KHONG bi rang cua
  do resize pixel. Xem chi tiet trong file do.
- Moi file .svg TRUOC DAY bi ET.parse() 2 LAN doc lap (1 lan ben trong
  classify_svg, 1 lan ben trong render_gt_label_mask) -> gio chi parse 1 LAN
  DUY NHAT trong _process_one_svg, roi truyen thang ET.Element (root) da
  parse cho ca 2 ham (render_gt_label_mask duoc truyen 1 BAN SAO rieng vi no
  se MUTATE cay XML de gan mau nhan GT).
- save_preview/save_quantized (dung chung tu opencv_stitch_classifier.py)
  gio ghi PNG voi muc nen thap hon -> ghi file nhanh hon dang ke.
===============================================================================
"""

import copy
import glob
import os
import sys
import xml.etree.ElementTree as ET
import time
import concurrent.futures
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cairosvg
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

# =============================================================================
# SUA DUONG DAN GOC CHO DUNG MAY BAN
# =============================================================================
OPENCV_TEST_DIR = "data/opencv_test"
# =============================================================================

SVG_DIR = os.path.join(OPENCV_TEST_DIR, "svg")
PREDICTIONS_DIR = os.path.join(OPENCV_TEST_DIR, "predictions")
QUANTIZED_DIR = os.path.join(OPENCV_TEST_DIR, "quantized")
GT_PREVIEW_DIR = os.path.join(OPENCV_TEST_DIR, "gt_preview")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

N_WORKERS: Optional[int] = None

# Ty le canvas dung cho buoc phan loai hinh hoc nhanh cua classify_svg (xem
# SVG_CLASSIFY_SCALE trong opencv_stitch_classifier.py). Giam so nay xuong
# neu muon chay nhanh hon nua (danh doi mot chut chi tiet hinh hoc nho).
CLASSIFY_SCALE: Optional[float] = None  # None = dung mac dinh cua module

# ---------------------------------------------------------------------------
# Cau hinh Weights & Biases
# ---------------------------------------------------------------------------
WANDB_ENABLED = True
WANDB_PROJECT = "embroidery-stitch-classifier"
WANDB_ENTITY: Optional[str] = None
WANDB_RUN_NAME: Optional[str] = None
# Ghi chu toc do: moi anh comparison (goc/predict/GT ghep lai) va upload len
# wandb deu ton them thoi gian dang ke. Neu chi can xem chi so (khong can
# xem truc quan tung anh), dat WANDB_LOG_IMAGES = False de chay nhanh hon.
WANDB_LOG_IMAGES = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opencv_stitch_classifier import (
    classify_svg, save_preview, save_quantized,
    LABEL_BACKGROUND, LABEL_FILL, LABEL_SATIN,
    DEFAULT_PHYSICAL_WIDTH_MM, DEFAULT_THRESHOLD_MM,
    SVG_CLASSIFY_SCALE,
)

CLASS_NAMES = {LABEL_BACKGROUND: "background", LABEL_FILL: "fill", LABEL_SATIN: "satin"}


def _tag(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


# ---------------------------------------------------------------------------
# THUAT TOAN DOAN NHAN DA TANG (HIERARCHICAL GUESSING)
# ---------------------------------------------------------------------------
def check_element_label(elem: ET.Element) -> Optional[str]:
    """Quet thuoc tinh cua 1 element (label, id, class, style)."""
    for key, val in elem.attrib.items():
        if key.endswith("}label") or key == "inkscape:label":
            v = val.strip().lower()
            if "satin" in v:
                return "satin"
            if "fill" in v:
                return "fill"
    for attr_name in ("data-label", "data-stitch", "data-stitch-type", "class", "id"):
        if attr_name in elem.attrib:
            v = elem.attrib[attr_name].strip().lower()
            if "satin" in v:
                return "satin"
            if "fill" in v:
                return "fill"
    style_str = elem.attrib.get("style", "").lower()
    if "#ff0000" in style_str:
        return "satin"
    if "#0000ff" in style_str or "#00ff00" in style_str:
        return "fill"
    return None


def guess_label(path_elem: ET.Element, parent_map: dict) -> Optional[str]:
    """Doan nhan: kiem tra the hien tai -> do nguoc len cac the cha -> mac dinh Fill."""
    label = check_element_label(path_elem)
    if label:
        return label

    curr = path_elem
    while curr in parent_map:
        curr = parent_map[curr]
        label = check_element_label(curr)
        if label:
            return label

    return "fill"


def render_gt_label_mask(svg_path: str, width: int, height: int,
                          pre_parsed_root: Optional[ET.Element] = None) -> Optional[np.ndarray]:
    """Tinh mask nhan GT. Neu duoc truyen san `pre_parsed_root` (mot BAN SAO
    rieng, vi ham nay se MUTATE cay XML de gan mau theo nhan doan duoc), se
    dung luon thay vi doc + parse lai file tu dau (tranh I/O + parse XML
    thua khi da parse file nay o noi khac trong cung 1 lan xu ly)."""
    if pre_parsed_root is not None:
        root = pre_parsed_root
    else:
        tree = ET.parse(svg_path)
        root = tree.getroot()

    parent_map = {c: p for p in root.iter() for c in p}

    paths = [el for el in root.iter() if _tag(el) == "path"]
    if not paths:
        return None

    for p in paths:
        label = guess_label(p, parent_map)
        color = "#FF0000" if label == "satin" else "#0000FF"
        p.set("fill", color)
        p.set("fill-opacity", "1")
        p.set("stroke", "none")
        p.attrib.pop("style", None)

    png_bytes = cairosvg.svg2png(bytestring=ET.tostring(root), output_width=width,
                                  output_height=height, background_color=None, unsafe=True)
    img = np.array(Image.open(BytesIO(png_bytes)).convert("RGBA"))

    alpha = img[:, :, 3]
    is_visible = alpha >= 128
    r, g, b = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    is_satin = is_visible & (r > 150) & (g < 80) & (b < 80)
    is_fill = is_visible & (b > 150) & (r < 80) & (g < 80)

    label_mask = np.zeros((height, width), dtype=np.uint8)
    label_mask[is_fill] = LABEL_FILL
    label_mask[is_satin] = LABEL_SATIN
    return label_mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def confusion_matrix_3class(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for gt_c in range(3):
        for pred_c in range(3):
            cm[gt_c, pred_c] = int(np.sum((gt == gt_c) & (pred == pred_c)))
    return cm


def metrics_from_confusion(cm: np.ndarray) -> Dict[int, Dict[str, float]]:
    results = {}
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
              else float("nan"))
        results[c] = {"iou": iou, "precision": precision, "recall": recall, "f1": f1}
    return results


def find_matching_image(base_name: str) -> Optional[str]:
    """Chi dung de HIEN THI cot 'Goc' trong bang so sanh - khong con anh huong
    den ket qua du doan (du doan gio doc truc tiep tu file SVG)."""
    for ext in IMAGE_EXTS:
        candidate = os.path.join(OPENCV_TEST_DIR, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None

# ---------------------------------------------------------------------------
# Lọc rác (Loại bỏ các cụm pixel siêu nhỏ do nhiễu/render lỗi)
# ---------------------------------------------------------------------------
def remove_small_components(mask: np.ndarray, min_area_px: int = 4) -> np.ndarray:
    cleaned_mask = mask.copy()
    for label_val in (LABEL_FILL, LABEL_SATIN):
        class_mask = (cleaned_mask == label_val).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area_px:
                cleaned_mask[labels == i] = LABEL_BACKGROUND
    return cleaned_mask


# ---------------------------------------------------------------------------
# Ghep 3 anh (goc / predict / ground-truth) canh nhau de log len wandb
# ---------------------------------------------------------------------------
def _build_comparison_image(image_path: str, pred_path: str, gt_path: str,
                             panel_height: int = 700) -> np.ndarray:
    def load_and_resize(path: str, target_h: int) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.full((target_h, target_h, 3), 40, dtype=np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            bgr = img[:, :, :3].astype(np.float32)
            alpha = (img[:, :, 3:4].astype(np.float32)) / 255.0
            white_bg = np.full_like(bgr, 255.0)
            img = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        h, w = img.shape[:2]
        scale = target_h / float(h)
        new_w = max(1, int(round(w * scale)))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        img = img.copy()
        bar_h = max(32, img.shape[0] // 18)
        cv2.rectangle(img, (0, 0), (img.shape[1], bar_h), (30, 30, 30), -1)
        cv2.putText(img, text, (10, int(bar_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    bar_h / 42.0, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    orig = add_label(load_and_resize(image_path, panel_height), "Goc")
    pred = add_label(load_and_resize(pred_path, panel_height), "Predict")
    gt = add_label(load_and_resize(gt_path, panel_height), "Ground Truth")

    separator = np.full((panel_height, 6, 3), 255, dtype=np.uint8)
    return cv2.hconcat([orig, separator, pred, separator, gt])


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _process_one_svg(svg_path: str) -> dict:
    base = os.path.splitext(os.path.basename(svg_path))[0]
    # Anh raster (neu co) gio chi dung de hien thi cot "Goc", KHONG con dung
    # de du doan - du doan chay truc tiep tren svg_path ben duoi.
    raster_image_path = find_matching_image(base)

    start_img_time = time.perf_counter()

    # TOI UU: parse SVG DUY NHAT 1 LAN o day, roi tai su dung cho ca
    # classify_svg (du doan) lan render_gt_label_mask (GT) - thay vi de moi
    # ham tu doc + parse lai file tu dau (I/O + parse XML thua).
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception as e:
        return {"base": base, "status": "error", "error": f"Loi parse SVG: {e}"}

    try:
        pred_mask, rendered_img = classify_svg(
            svg_path, pre_parsed_root=root,
            classify_scale=CLASSIFY_SCALE or SVG_CLASSIFY_SCALE,
            verbose=False,
        )
    except Exception as e:
        return {"base": base, "status": "error", "error": str(e)}

    # Dọn rác cho Prediction
    pred_mask = remove_small_components(pred_mask, min_area_px=4)

    pred_path = os.path.join(PREDICTIONS_DIR, f"{base}_pred.png")
    save_preview(pred_mask, pred_path)
    quantized_path = os.path.join(QUANTIZED_DIR, f"{base}_quantized.png")
    save_quantized(rendered_img, quantized_path)

    h, w = pred_mask.shape[:2]
    # render_gt_label_mask MUTATE cay XML (gan mau theo nhan doan duoc) nen
    # truyen 1 BAN SAO rieng cua root, khong dung chung voi ban da dung cho
    # classify_svg o tren (de tranh xung dot trang thai).
    gt_mask = render_gt_label_mask(svg_path, width=w, height=h,
                                    pre_parsed_root=copy.deepcopy(root))
    if gt_mask is None:
        return {"base": base, "status": "no_gt"}

    # Dọn rác cho Ground Truth (nhiễu viền SVG)
    gt_mask = remove_small_components(gt_mask, min_area_px=4)

    cm = confusion_matrix_3class(pred_mask, gt_mask)
    per_class = metrics_from_confusion(cm)

    gt_has_fill = bool(np.any(gt_mask == LABEL_FILL))
    gt_has_satin = bool(np.any(gt_mask == LABEL_SATIN))
    applicable_iou, applicable_f1 = [], []
    if gt_has_fill:
        applicable_iou.append(per_class[LABEL_FILL]["iou"])
        applicable_f1.append(per_class[LABEL_FILL]["f1"])
    if gt_has_satin:
        applicable_iou.append(per_class[LABEL_SATIN]["iou"])
        applicable_f1.append(per_class[LABEL_SATIN]["f1"])

    mean_iou = float(np.nanmean(applicable_iou)) if applicable_iou else float("nan")
    mean_f1 = float(np.nanmean(applicable_f1)) if applicable_f1 else float("nan")

    gt_preview_path = os.path.join(GT_PREVIEW_DIR, f"{base}_gt.png")
    save_preview(gt_mask, gt_preview_path)

    # Cot "Goc": uu tien anh raster that neu tim thay, khong thi dung ban da
    # render tu SVG (da luu o quantized_path o tren).
    goc_image_path = raster_image_path if raster_image_path is not None else quantized_path

    # Tao anh comparison trong RAM va chuyen sang RGB cho wandb
    composite_bgr = _build_comparison_image(goc_image_path, pred_path, gt_preview_path)
    composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)

    img_time = time.perf_counter() - start_img_time

    return {
        "base": base,
        "status": "ok",
        "cm": cm,
        "mean_iou": mean_iou,
        "mean_f1": mean_f1,
        "gt_has_fill": gt_has_fill,
        "gt_has_satin": gt_has_satin,
        "img_time": img_time,
        "comparison_img": composite_rgb, # Truyen truc tiep array
    }


def main():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(QUANTIZED_DIR, exist_ok=True)
    os.makedirs(GT_PREVIEW_DIR, exist_ok=True)

    svg_files = sorted(glob.glob(os.path.join(SVG_DIR, "*.svg")))
    if not svg_files:
        print(f"Khong tim thay file .svg nao trong {SVG_DIR}")
        return

    all_cm = np.zeros((3, 3), dtype=np.int64)
    per_image_iou, per_image_f1 = [], []
    n_evaluated, n_skipped = 0, 0

    n_workers = N_WORKERS or max(1, (os.cpu_count() or 2) - 1)

    use_wandb = WANDB_ENABLED and _HAS_WANDB
    if WANDB_ENABLED and not _HAS_WANDB:
        print("[CANH BAO] WANDB_ENABLED=True nhung chua cai 'wandb' "
              "(pip install wandb) -> bo qua logging len wandb.")

    wandb_table = None
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_RUN_NAME,
            config={
                "n_images_found": len(svg_files),
                "n_workers": n_workers,
                "physical_width_mm": DEFAULT_PHYSICAL_WIDTH_MM,
                "threshold_mm": DEFAULT_THRESHOLD_MM,
                "classify_scale": CLASSIFY_SCALE or SVG_CLASSIFY_SCALE,
                "svg_dir": SVG_DIR,
                "primary_metric": "pixel-level",
                "prediction_source": "svg",
            },
        )
        wandb_table = wandb.Table(columns=[
            "image", "iou", "f1", "time_s", "note",
            "bg_iou", "bg_precision", "bg_recall", "bg_f1",
            "fill_iou", "fill_precision", "fill_recall", "fill_f1",
            "satin_iou", "satin_precision", "satin_recall", "satin_f1",
            "comparison",
        ])

    print("=" * 60)
    print("BAT DAU CHAY DANH GIA (BATCH MODE - SONG SONG)")
    print(f"So tien trinh song song: {n_workers}")
    print(f"Classify scale (canvas thu nho cho fast pass): {CLASSIFY_SCALE or SVG_CLASSIFY_SCALE:.3f}")
    print("CHI SO CHINH: PIXEL-LEVEL | NGUON DU DOAN: SVG (truc tiep tung <path>)")
    if use_wandb:
        print(f"Wandb: BAT ({WANDB_PROJECT})")
    print("=" * 60)

    start_total_time = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one_svg, svg_path): svg_path
                   for svg_path in svg_files}

        for future in tqdm(concurrent.futures.as_completed(futures),
                            total=len(svg_files), desc="Dang danh gia", unit="anh"):
            svg_path = futures[future]
            base_fallback = os.path.splitext(os.path.basename(svg_path))[0]

            try:
                result = future.result()
            except Exception as e:
                tqdm.write(f"[loi nghiem trong] '{base_fallback}': {e}")
                n_skipped += 1
                continue

            base = result["base"]
            status = result["status"]

            if status == "error":
                tqdm.write(f"[loi] classify_svg that bai cho '{base}': {result['error']}")
                n_skipped += 1
                continue
            if status == "no_gt":
                tqdm.write(f"[LOI] Khong co <path> nao trong SVG cua '{base}'")
                n_skipped += 1
                continue

            cm = result["cm"]
            mean_iou = result["mean_iou"]
            mean_f1 = result["mean_f1"]
            gt_has_fill = result["gt_has_fill"]
            gt_has_satin = result["gt_has_satin"]
            img_time = result["img_time"]

            per_class = metrics_from_confusion(cm)

            all_cm += cm
            if not np.isnan(mean_iou):
                per_image_iou.append(mean_iou)
            if not np.isnan(mean_f1):
                per_image_f1.append(mean_f1)
            n_evaluated += 1

            note = ""
            if not (gt_has_fill and gt_has_satin):
                only = "fill" if gt_has_fill else ("satin" if gt_has_satin else "khong co gi")
                note = f"chi co {only} trong GT"

            note_display = f"  [{note}]" if note else ""
            tqdm.write(f"{base:20s} | {img_time:5.2f}s | meanIoU={mean_iou:.3f} | meanF1={mean_f1:.3f}{note_display}")
            tqdm.write(f"    bg: IoU={per_class[LABEL_BACKGROUND]['iou']:.3f}  "
                       f"fill: IoU={per_class[LABEL_FILL]['iou']:.3f}  "
                       f"satin: IoU={per_class[LABEL_SATIN]['iou']:.3f}")

            if use_wandb:
                log_dict = {
                    "per_image/iou": mean_iou,
                    "per_image/f1": mean_f1,
                    "per_image/time_s": img_time,
                }
                for c in range(3):
                    cname = CLASS_NAMES[c]
                    m = per_class[c]
                    log_dict[f"per_image/{cname}_iou"] = m["iou"]
                    log_dict[f"per_image/{cname}_precision"] = m["precision"]
                    log_dict[f"per_image/{cname}_recall"] = m["recall"]
                    log_dict[f"per_image/{cname}_f1"] = m["f1"]
                wandb.log(log_dict)

                comparison_img_log = None
                if WANDB_LOG_IMAGES:
                    try:
                        comparison_img_log = wandb.Image(result["comparison_img"], caption=base)
                    except Exception as e:
                        tqdm.write(f"    [canh bao] khong log duoc anh cho '{base}': {e}")
                        comparison_img_log = None

                bg_m, fill_m, satin_m = per_class[LABEL_BACKGROUND], per_class[LABEL_FILL], per_class[LABEL_SATIN]
                wandb_table.add_data(
                    base, mean_iou, mean_f1,
                    img_time, note,
                    bg_m["iou"], bg_m["precision"], bg_m["recall"], bg_m["f1"],
                    fill_m["iou"], fill_m["precision"], fill_m["recall"], fill_m["f1"],
                    satin_m["iou"], satin_m["precision"], satin_m["recall"], satin_m["f1"],
                    comparison_img_log,
                )

    end_total_time = time.perf_counter()
    total_elapsed = end_total_time - start_total_time

    print("\n" + "=" * 60)
    print(f"DA DANH GIA: {n_evaluated} anh  (bo qua: {n_skipped})")
    print(f"TONG THOI GIAN: {total_elapsed:.2f} giay ({n_workers} tien trinh song song)")

    if n_evaluated == 0:
        if use_wandb:
            wandb.finish()
        return

    mean_iou_macro = float(np.nanmean(per_image_iou))
    mean_f1_macro = float(np.nanmean(per_image_f1))

    print("\n--- MACRO (trung binh cong moi anh) ---")
    print(f"Mean IoU (fill+satin): {mean_iou_macro:.3f}")
    print(f"Mean F1  (fill+satin): {mean_f1_macro:.3f}")

    print("\n--- MICRO (gop tat ca pixel lai tinh 1 lan) ---")
    micro = metrics_from_confusion(all_cm)
    for c in range(3):
        m = micro[c]
        print(f"  {CLASS_NAMES[c]:10s}: IoU={m['iou']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  F1={m['f1']:.3f}")

    print(f"\nPredictions -> {PREDICTIONS_DIR}/")
    print(f"Quantized   -> {QUANTIZED_DIR}/")

    if use_wandb:
        summary = {
            "summary/mean_iou_macro": mean_iou_macro,
            "summary/mean_f1_macro": mean_f1_macro,
            "summary/n_evaluated": n_evaluated,
            "summary/n_skipped": n_skipped,
            "summary/total_time_s": total_elapsed,
            "summary/avg_time_per_image_s": total_elapsed / len(svg_files) if svg_files else 0.0,
        }
        for c in range(3):
            m = micro[c]
            summary[f"summary/{CLASS_NAMES[c]}_iou"] = m["iou"]
            summary[f"summary/{CLASS_NAMES[c]}_precision"] = m["precision"]
            summary[f"summary/{CLASS_NAMES[c]}_recall"] = m["recall"]
            summary[f"summary/{CLASS_NAMES[c]}_f1"] = m["f1"]

        wandb.log(summary)
        wandb.log({"per_image_results": wandb_table})
        wandb.finish()
        print(f"\nDa log len Wandb (project: {WANDB_PROJECT})")


if __name__ == "__main__":
    main()