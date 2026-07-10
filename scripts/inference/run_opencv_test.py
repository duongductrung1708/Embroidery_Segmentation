#!/usr/bin/env python3
"""
run_opencv_test.py
- Chay batch + danh gia dung theo cau truc thu muc data/opencv_test/
- Doan nhan GT da tang (Hierarchical Guessing): tu dong thua ke nhan tu Group
  cha va Fallback theo mau Style (Red=Satin, Blue/Green=Fill), mac dinh Fill
  neu khong doan duoc gi ca.
- Log day du: MACRO (trung binh moi anh) + MICRO (gop tat ca pixel) +
  SHAPE-LEVEL (gop tat ca shape), giong dinh dang bao cao chuan cua ban.
"""

import glob
import os
import sys
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cairosvg
import cv2
import numpy as np
from PIL import Image

# =============================================================================
# SUA DUONG DAN GOC CHO DUNG MAY BAN
# =============================================================================
OPENCV_TEST_DIR = "data/opencv_test"
# =============================================================================

SVG_DIR = os.path.join(OPENCV_TEST_DIR, "svg")
PREDICTIONS_DIR = os.path.join(OPENCV_TEST_DIR, "predictions")
QUANTIZED_DIR = os.path.join(OPENCV_TEST_DIR, "quantized")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opencv_stitch_classifier import (
    classify_multicolor_image, save_preview, save_quantized,
    LABEL_BACKGROUND, LABEL_FILL, LABEL_SATIN,
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
    # Fallback mau sac (Do=Satin, Xanh duong/Xanh la=Fill)
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
        print(f"[LOI] Khong co <path> nao trong {svg_path}")
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

    for svg_path in svg_files:
        base = os.path.splitext(os.path.basename(svg_path))[0]
        image_path = find_matching_image(base)
        if image_path is None:
            print(f"[bo qua] khong tim thay anh khop voi '{base}' truc tiep trong {OPENCV_TEST_DIR}")
            n_skipped += 1
            continue

        try:
            pred_mask, quantized_img = classify_multicolor_image(image_path, verbose=False)
        except Exception as e:
            print(f"[loi] classify that bai cho '{base}': {e}")
            n_skipped += 1
            continue

        save_preview(pred_mask, os.path.join(PREDICTIONS_DIR, f"{base}_pred.png"))
        save_quantized(quantized_img, os.path.join(QUANTIZED_DIR, f"{base}_quantized.png"))

        h, w = pred_mask.shape[:2]
        gt_mask = render_gt_label_mask(svg_path, width=w, height=h)
        if gt_mask is None:
            n_skipped += 1
            continue

        cm = confusion_matrix_3class(pred_mask, gt_mask)
        per_class = metrics_from_confusion(cm)

        # Chi tinh mean_iou/mean_f1 tren lop THUC SU CO trong GT cua anh nay
        # (tranh anh chi co 1 loai stitch bi tru diem oan boi lop vang mat)
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
            note = f"  [chi co {only} trong GT]"
        print(f"{base:20s}  meanIoU={mean_iou:.3f}  meanF1={mean_f1:.3f}  "
              f"shape_acc={shape_acc:.3f} ({s_correct}/{s_total}){note}")
        for area, gt_l, pred_l in sorted(s_mismatches, reverse=True)[:5]:
            print(f"    -> shape sai: area={area:>8d}px  gt={gt_l:6s}  pred={pred_l}")

    print("\n" + "=" * 60)
    print(f"DA DANH GIA: {n_evaluated} anh  (bo qua: {n_skipped})")
    if n_evaluated == 0:
        return

    print("\n--- MACRO (trung binh cong moi anh) ---")
    print(f"Mean IoU (fill+satin): {np.nanmean(per_image_iou):.3f}")
    print(f"Mean F1  (fill+satin): {np.nanmean(per_image_f1):.3f}")

    print("\n--- MICRO (gop tat ca pixel lai tinh 1 lan) ---")
    micro = metrics_from_confusion(all_cm)
    for c in range(3):
        m = micro[c]
        print(f"  {CLASS_NAMES[c]:10s}: IoU={m['iou']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  F1={m['f1']:.3f}")

    print("\n--- SHAPE-LEVEL (gop tat ca shape cua moi anh) ---")
    if total_shapes:
        print(f"Tong shape: {total_shapes}, dung: {total_correct}, "
              f"accuracy = {total_correct/total_shapes:.3f}")
    else:
        print("khong co shape nao")

    print(f"\nPredictions -> {PREDICTIONS_DIR}/")
    print(f"Quantized   -> {QUANTIZED_DIR}/")


if __name__ == "__main__":
    main()