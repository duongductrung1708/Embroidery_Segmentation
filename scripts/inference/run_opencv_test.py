#!/usr/bin/env python3
"""
Chay batch + danh gia dung theo cau truc thu muc thuc te cua ban:

data/opencv_test/
  svg/              <- SVG co san nhan GT (vd 4.svg, 14.svg, 23.svg, ...)
  predictions/      <- (tu tao) luu preview du doan
  quantized/        <- (tu tao) luu anh da luong tu hoa mau
  4.png, 14.png, 23.png, 92.png, 127.png, 132.png, 144.png, 148.png, test.png
                    <- anh that can predict, TEN TRUNG voi file trong svg/

CHI CAN SUA DUONG DAN GOC (OPENCV_TEST_DIR) O DUOI RoI CHAY:
    python run_opencv_test.py
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


# ---------------------------------------------------------------------------
# Doan nhan GT tu 1 <path> SVG (sua ham nay neu file cua ban dung quy uoc khac)
# ---------------------------------------------------------------------------
def _tag(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def guess_label(path_elem: ET.Element) -> Optional[str]:
    for key, val in path_elem.attrib.items():
        if key.endswith("}label") or key == "inkscape:label":
            v = val.strip().lower()
            if "satin" in v:
                return "satin"
            if "fill" in v:
                return "fill"
    for attr_name in ("data-label", "data-stitch", "data-stitch-type"):
        if attr_name in path_elem.attrib:
            v = path_elem.attrib[attr_name].strip().lower()
            if "satin" in v:
                return "satin"
            if "fill" in v:
                return "fill"
    if "class" in path_elem.attrib:
        v = path_elem.attrib["class"].strip().lower()
        if "satin" in v:
            return "satin"
        if "fill" in v:
            return "fill"
    if "id" in path_elem.attrib:
        v = path_elem.attrib["id"].strip().lower()
        if "satin" in v:
            return "satin"
        if "fill" in v:
            return "fill"
    return None


def render_gt_label_mask(svg_path: str, width: int, height: int) -> Optional[np.ndarray]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    paths = [el for el in root.iter() if _tag(el) == "path"]
    if not paths:
        print(f"[LOI] Khong co <path> nao trong {svg_path}")
        return None

    unresolved = []
    for p in paths:
        label = guess_label(p)
        if label is None:
            unresolved.append(p)
        color = "#FF0000" if label == "satin" else "#0000FF"  # danh dau noi bo
        p.set("fill", color)
        p.set("fill-opacity", "1")
        p.set("stroke", "none")
        p.attrib.pop("style", None)

    if unresolved:
        sample = unresolved[0]
        print(f"[LOI] Khong doan duoc nhan GT cho {len(unresolved)}/{len(paths)} path trong {svg_path}")
        print("       Attribute thuc te cua path dau tien chua doan duoc:")
        for k, v in sample.attrib.items():
            print(f"         {k} = {v!r}")
        print("       -> Sua ham guess_label() o dau file nay cho dung quy uoc GT that.")
        return None

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

        # CHI tinh mean IoU/F1 tren cac lop THUC SU CO MAT trong GT cua anh nay.
        # Ly do: neu GT chi co fill (khong co satin nao ca), ma cu ep tinh
        # mean([iou_fill, iou_satin]), thi chi can pred lo du doan nham vai
        # pixel thanh satin (nhieu/rac) la iou_satin roi thang xuong 0.0 (thay
        # vi "khong ap dung"), keo diem trung binh xuong mot cach bat cong du
        # anh gan nhu hoan hao. Ngoai ra van CANH BAO rieng neu co hien tuong
        # du doan nham sang lop khong ton tai trong GT, de khong giau loi that.
        gt_has_fill = bool(np.any(gt_mask == LABEL_FILL))
        gt_has_satin = bool(np.any(gt_mask == LABEL_SATIN))

        applicable_iou, applicable_f1 = [], []
        if gt_has_fill:
            applicable_iou.append(per_class[LABEL_FILL]["iou"])
            applicable_f1.append(per_class[LABEL_FILL]["f1"])
        if gt_has_satin:
            applicable_iou.append(per_class[LABEL_SATIN]["iou"])
            applicable_f1.append(per_class[LABEL_SATIN]["f1"])

        mean_iou = np.nanmean(applicable_iou) if applicable_iou else float("nan")
        mean_f1 = np.nanmean(applicable_f1) if applicable_f1 else float("nan")

        # Canh bao rieng: pred co du doan nham sang lop KHONG ton tai trong GT
        warnings = []
        if not gt_has_satin and np.any(pred_mask == LABEL_SATIN):
            n_fp = int(np.sum(pred_mask == LABEL_SATIN))
            warnings.append(f"GT khong co satin nhung pred du doan nham {n_fp}px thanh satin")
        if not gt_has_fill and np.any(pred_mask == LABEL_FILL):
            n_fp = int(np.sum(pred_mask == LABEL_FILL))
            warnings.append(f"GT khong co fill nhung pred du doan nham {n_fp}px thanh fill")

        s_total, s_correct, s_mismatches = shape_level_stats(pred_mask, gt_mask)

        all_cm += cm
        per_image_iou.append(mean_iou)
        per_image_f1.append(mean_f1)
        total_shapes += s_total
        total_correct += s_correct
        n_evaluated += 1

        shape_acc = s_correct / s_total if s_total else float("nan")
        print(f"{base:20s}  meanIoU={mean_iou:.3f}  meanF1={mean_f1:.3f}  "
              f"shape_acc={shape_acc:.3f} ({s_correct}/{s_total})")
        for warning in warnings:
            print(f"    [CANH BAO] {warning}")
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