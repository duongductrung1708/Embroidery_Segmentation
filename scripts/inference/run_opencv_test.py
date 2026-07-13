#!/usr/bin/env python3
"""
run_opencv_test.py
- Chay batch + danh gia dung theo cau truc thu muc data/opencv_test/
- Doan nhan GT da tang (Hierarchical Guessing): tu dong thua ke nhan tu Group
  cha va Fallback theo mau Style (Red=Satin, Blue/Green=Fill), mac dinh Fill
  neu khong doan duoc gi ca.
- TICH HOP DO THOI GIAN & THANH TIEN TRINH (tqdm).
- CHAY SONG SONG NHIEU ANH CUNG LUC (multiprocessing) - moi anh xu ly doc lap
  hoan toan nen tan dung duoc toi da so CPU core.
- MOI: LOG LEN WEIGHTS & BIASES (wandb) - de trong doi (KHONG anh huong ket
  qua/toc do xu ly): metric theo tung anh (real-time), 1 bang tong hop de loc/
  sort trong UI, va tong ket cuoi cung (MACRO/MICRO/SHAPE-LEVEL). Chi log tu
  PROCESS CHINH (sau khi nhan ket qua tu worker qua future.result()) - KHONG
  khoi tao wandb ben trong worker de tranh xung dot run/auth giua cac process.
  Tu dong bo qua neu chua cai 'wandb' (pip install wandb), khong lam crash
  script.
- Log day du: MACRO (trung binh moi anh) + MICRO (gop tat ca pixel) +
  SHAPE-LEVEL (gop tat ca shape), giong dinh dang bao cao chuan cua ban.
"""

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
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# MOI: so tien trinh song song. None = tu dong dung (so CPU - 1, toi thieu 1).
N_WORKERS: Optional[int] = None

# ---------------------------------------------------------------------------
# MOI: cau hinh Weights & Biases
# ---------------------------------------------------------------------------
WANDB_ENABLED = True                              # dat False de tat han, khong can go import
WANDB_PROJECT = "embroidery-stitch-classifier"    # doi ten project cho dung workspace cua ban
WANDB_ENTITY: Optional[str] = None                # doi neu dung team/org rieng, None = mac dinh tai khoan
WANDB_RUN_NAME: Optional[str] = None              # None = de wandb tu dat ten (vd "different-cloud-12")
WANDB_LOG_IMAGES = False                          # Bat True de upload anh preview len wandb (cham hon, ton bang thong)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opencv_stitch_classifier import (
    classify_multicolor_image, save_preview, save_quantized,
    LABEL_BACKGROUND, LABEL_FILL, LABEL_SATIN,
    DEFAULT_PHYSICAL_WIDTH_MM, DEFAULT_THRESHOLD_MM,
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

    return "fill"  # Fallback cuoi cung


def render_gt_label_mask(svg_path: str, width: int, height: int) -> Optional[np.ndarray]:
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


def shape_level_stats(pred: np.ndarray, gt: np.ndarray, min_area_px: int = 4) -> Tuple[int, int, List]:
    total, correct, mismatches = 0, 0, []
    for gt_label in (LABEL_FILL, LABEL_SATIN):
        class_mask = (gt == gt_label).astype(np.uint8)
        n_labels, components = cv2.connectedComponents(class_mask, connectivity=8)
        for comp_id in range(1, n_labels):
            comp_mask = components == comp_id
            area = int(comp_mask.sum())
            if area < min_area_px:
                continue
            preds_in_shape = pred[comp_mask]
            n_fill = np.sum(preds_in_shape == LABEL_FILL)
            n_satin = np.sum(preds_in_shape == LABEL_SATIN)
            pred_label = (LABEL_BACKGROUND if n_fill == 0 and n_satin == 0
                          else (LABEL_SATIN if n_satin > n_fill else LABEL_FILL))
            total += 1
            if pred_label == gt_label:
                correct += 1
            else:
                mismatches.append((area, CLASS_NAMES[gt_label], CLASS_NAMES[pred_label]))
    return total, correct, mismatches


def find_matching_image(base_name: str) -> Optional[str]:
    for ext in IMAGE_EXTS:
        candidate = os.path.join(OPENCV_TEST_DIR, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# don vi cong viec cho 1 anh - chay trong WORKER PROCESS RIENG.
# Tra ve dict ket qua NHE (khong chua mang anh lon) de giam chi phi truyen
# du lieu giua cac tien trinh (IPC pickle overhead). KHONG goi wandb o day.
# ---------------------------------------------------------------------------
def _process_one_svg(svg_path: str) -> dict:
    base = os.path.splitext(os.path.basename(svg_path))[0]
    image_path = find_matching_image(base)

    if image_path is None:
        return {"base": base, "status": "no_image"}

    start_img_time = time.perf_counter()
    try:
        pred_mask, quantized_img = classify_multicolor_image(image_path, verbose=False)
    except Exception as e:
        return {"base": base, "status": "error", "error": str(e)}

    pred_path = os.path.join(PREDICTIONS_DIR, f"{base}_pred.png")
    save_preview(pred_mask, pred_path)
    save_quantized(quantized_img, os.path.join(QUANTIZED_DIR, f"{base}_quantized.png"))

    h, w = pred_mask.shape[:2]
    gt_mask = render_gt_label_mask(svg_path, width=w, height=h)
    if gt_mask is None:
        return {"base": base, "status": "no_gt"}

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
    s_total, s_correct, s_mismatches = shape_level_stats(pred_mask, gt_mask)

    img_time = time.perf_counter() - start_img_time

    return {
        "base": base,
        "status": "ok",
        "cm": cm,
        "mean_iou": mean_iou,
        "mean_f1": mean_f1,
        "s_total": s_total,
        "s_correct": s_correct,
        "s_mismatches": s_mismatches,
        "gt_has_fill": gt_has_fill,
        "gt_has_satin": gt_has_satin,
        "img_time": img_time,
        "pred_path": pred_path,  # MOI: de log anh len wandb neu can (doc lai tu dia, khong truyen mang qua IPC)
    }


def main():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(QUANTIZED_DIR, exist_ok=True)

    svg_files = sorted(glob.glob(os.path.join(SVG_DIR, "*.svg")))
    if not svg_files:
        print(f"Khong tim thay file .svg nao trong {SVG_DIR}")
        return

    all_cm = np.zeros((3, 3), dtype=np.int64)
    per_image_iou, per_image_f1 = [], []
    total_shapes, total_correct = 0, 0
    n_evaluated, n_skipped = 0, 0

    n_workers = N_WORKERS or max(1, (os.cpu_count() or 2) - 1)

    # MOI: khoi tao wandb (CHI trong process chinh)
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
                "svg_dir": SVG_DIR,
            },
        )
        wandb_table = wandb.Table(columns=[
            "image", "iou", "f1", "shape_acc", "shapes_total", "shapes_correct",
            "time_s", "note", "preview",
        ])

    print("=" * 60)
    print("BAT DAU CHAY DANH GIA (BATCH MODE - SONG SONG)")
    print(f"So tien trinh song song: {n_workers}")
    if use_wandb:
        print(f"Wandb: BAT ({WANDB_PROJECT})")
    print("=" * 60)

    start_total_time = time.perf_counter()

    # Chay song song nhieu anh cung luc bang ProcessPoolExecutor. Moi anh doc
    # lap hoan toan (khong chia se state) nen an toan de chay song song.
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

            if status == "no_image":
                tqdm.write(f"[bo qua] khong tim thay anh khop voi '{base}' "
                           f"truc tiep trong {OPENCV_TEST_DIR}")
                n_skipped += 1
                continue
            if status == "error":
                tqdm.write(f"[loi] classify that bai cho '{base}': {result['error']}")
                n_skipped += 1
                continue
            if status == "no_gt":
                tqdm.write(f"[LOI] Khong co <path> nao trong SVG cua '{base}'")
                n_skipped += 1
                continue

            # status == "ok"
            cm = result["cm"]
            mean_iou = result["mean_iou"]
            mean_f1 = result["mean_f1"]
            s_total = result["s_total"]
            s_correct = result["s_correct"]
            s_mismatches = result["s_mismatches"]
            gt_has_fill = result["gt_has_fill"]
            gt_has_satin = result["gt_has_satin"]
            img_time = result["img_time"]

            all_cm += cm
            if not np.isnan(mean_iou):
                per_image_iou.append(mean_iou)
            if not np.isnan(mean_f1):
                per_image_f1.append(mean_f1)
            total_shapes += s_total
            total_correct += s_correct
            n_evaluated += 1

            shape_acc = s_correct / s_total if s_total else float("nan")
            note = ""
            if not (gt_has_fill and gt_has_satin):
                only = "fill" if gt_has_fill else ("satin" if gt_has_satin else "khong co gi")
                note = f"chi co {only} trong GT"

            note_display = f"  [{note}]" if note else ""
            tqdm.write(f"{base:20s} | {img_time:5.2f}s | meanIoU={mean_iou:.3f} | meanF1={mean_f1:.3f} | "
                       f"shape_acc={shape_acc:.3f} ({s_correct}/{s_total}){note_display}")

            for area, gt_l, pred_l in sorted(s_mismatches, reverse=True)[:5]:
                tqdm.write(f"    -> shape sai: area={area:>8d}px  gt={gt_l:6s}  pred={pred_l}")

            # MOI: log metric theo tung anh len wandb (real-time, xem duoc ngay
            # tren dashboard trong khi batch dang chay)
            if use_wandb:
                log_dict = {
                    "per_image/iou": mean_iou,
                    "per_image/f1": mean_f1,
                    "per_image/shape_acc": shape_acc,
                    "per_image/time_s": img_time,
                }
                wandb.log(log_dict)

                preview_img = None
                if WANDB_LOG_IMAGES:
                    try:
                        preview_img = wandb.Image(result["pred_path"], caption=base)
                    except Exception:
                        preview_img = None

                wandb_table.add_data(
                    base, mean_iou, mean_f1, shape_acc, s_total, s_correct,
                    img_time, note, preview_img,
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

    print("\n--- SHAPE-LEVEL (gop tat ca shape cua moi anh) ---")
    shape_accuracy_total = None
    if total_shapes:
        shape_accuracy_total = total_correct / total_shapes
        print(f"Tong shape: {total_shapes}, dung: {total_correct}, "
              f"accuracy = {shape_accuracy_total:.3f}")
    else:
        print("khong co shape nao")

    print(f"\nPredictions -> {PREDICTIONS_DIR}/")
    print(f"Quantized   -> {QUANTIZED_DIR}/")

    # MOI: log tong ket cuoi cung + bang chi tiet len wandb
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
        if shape_accuracy_total is not None:
            summary["summary/shape_accuracy"] = shape_accuracy_total

        wandb.log(summary)
        wandb.log({"per_image_results": wandb_table})
        wandb.finish()
        print(f"\nDa log len Wandb (project: {WANDB_PROJECT})")


if __name__ == "__main__":
    main()