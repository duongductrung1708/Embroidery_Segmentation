#!/usr/bin/env python3
"""
So sanh SVG co nhan GT voi anh PREDICTED
==============================================================================

CHI CAN SUA 2 DUONG DAN BEN DUOI (SVG_DIR va PREVIEW_DIR) roi chay:
    python compare_svg_gt_with_preview.py

BẢN CẬP NHẬT:
1. Tích hợp SMART LOAD (chấp cả ảnh mask 0,1,2 lẫn ảnh Preview màu Vàng/Hồng).
2. Tich hop Boundary Tolerance de loai bo nhieu rang cua ra khoi chi so Metrics.
"""

import glob
import os
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cairosvg
import cv2
import numpy as np
from PIL import Image

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================================================================
SVG_DIR = "data/opencv_test/svg"
PREVIEW_DIR = "data/opencv_test/predictions"
# =============================================================================

LABEL_BACKGROUND, LABEL_FILL, LABEL_SATIN = 0, 1, 2
CLASS_NAMES = {LABEL_BACKGROUND: "background", LABEL_FILL: "fill", LABEL_SATIN: "satin"}
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("inkscape", INKSCAPE_NS)

# Dung sai bo qua vien (pixel)
DEFAULT_BOUNDARY_TOLERANCE_PX = 5


def _tag(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def guess_label(path_elem: ET.Element) -> Optional[str]:
    """Doan nhan GT cua 1 path."""
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


def render_gt_label_mask(svg_path: str, width: int, height: int) -> np.ndarray:
    """Render SVG -> label-mask (0/1/2) dung khop kich thuoc (width,height)."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    paths = [el for el in root.iter() if _tag(el) == "path"]
    if not paths:
        raise ValueError(f"Khong co <path> nao trong {svg_path}")

    unresolved = []
    for p in paths:
        label = guess_label(p)
        if label is None:
            unresolved.append(p)
        color = "#FF0000" if label == "fill" else "#00FF00"  # Đỏ -> Fill, Lục -> Satin
        p.set("fill", color)
        p.set("fill-opacity", "1")
        p.set("stroke", "none")
        p.attrib.pop("style", None)

    if unresolved:
        sample = unresolved[0]
        print(f"[LOI] Khong doan duoc nhan GT cho {len(unresolved)}/{len(paths)} path trong {svg_path}")
        return None

    png_bytes = cairosvg.svg2png(bytestring=ET.tostring(root), output_width=width,
                                  output_height=height, background_color=None, unsafe=True)
    img = np.array(Image.open(BytesIO(png_bytes)).convert("RGB"))

    label_mask = np.zeros((height, width), dtype=np.uint8)
    label_mask[img[:, :, 0] > 128] = LABEL_FILL   # Đỏ
    label_mask[img[:, :, 1] > 128] = LABEL_SATIN  # Xanh lục
    return label_mask


def load_preview_label_mask(pred_path: str) -> np.ndarray:
    """SMART LOAD: Tự động nhận diện ảnh xám (0,1,2) hoặc ảnh màu (Cyan/Magenta)"""
    pred_bgr = cv2.imread(pred_path, cv2.IMREAD_COLOR)
    if pred_bgr is None:
        raise FileNotFoundError(pred_path)

    h, w = pred_bgr.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    # KỊCH BẢN 1: Ảnh mask đen thui (toàn số 0, 1, 2)
    if pred_bgr.max() <= 2:
        return pred_bgr[:, :, 0]

    # KỊCH BẢN 2: Ảnh màu Preview 
    # Màu Vàng (Cyan BGR) -> Gán Fill (1)
    fill_pixels = (pred_bgr[:, :, 0] < 50) & (pred_bgr[:, :, 1] > 200) & (pred_bgr[:, :, 2] > 200)
    # Màu Hồng (Magenta BGR) -> Gán Satin (2)
    satin_pixels = (pred_bgr[:, :, 0] > 200) & (pred_bgr[:, :, 1] < 50) & (pred_bgr[:, :, 2] > 200)

    pred_mask[fill_pixels] = 1
    pred_mask[satin_pixels] = 2

    return pred_mask


def build_boundary_ignore_mask(gt_mask: np.ndarray, tolerance_px: int) -> np.ndarray:
    """Tao vung dem quanh cac duong bien de bo qua nhieu rang cua."""
    if tolerance_px <= 0:
        return np.zeros_like(gt_mask, dtype=bool)

    edges = np.zeros_like(gt_mask, dtype=bool)
    edges |= (gt_mask != np.roll(gt_mask, 1, axis=0))
    edges |= (gt_mask != np.roll(gt_mask, -1, axis=0))
    edges |= (gt_mask != np.roll(gt_mask, 1, axis=1))
    edges |= (gt_mask != np.roll(gt_mask, -1, axis=1))

    edges_u8 = edges.astype(np.uint8)
    kernel = np.ones((tolerance_px * 2 + 1, tolerance_px * 2 + 1), np.uint8)
    dilated = cv2.dilate(edges_u8, kernel, iterations=1)

    return dilated.astype(bool)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def confusion_matrix_3class(pred: np.ndarray, gt: np.ndarray, ignore_mask: np.ndarray = None) -> np.ndarray:
    """Tinh ma tran nham lan, bo qua pixel thuoc ignore_mask."""
    cm = np.zeros((3, 3), dtype=np.int64)
    valid_mask = ~ignore_mask if ignore_mask is not None else np.ones_like(gt, dtype=bool)

    valid_gt = gt[valid_mask]
    valid_pred = pred[valid_mask]

    for gt_c in range(3):
        for pred_c in range(3):
            cm[gt_c, pred_c] = int(np.sum((valid_gt == gt_c) & (valid_pred == pred_c)))
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


def find_matching_image(image_dir: str, base_name: str) -> Optional[str]:
    # Ho tro ca ten goc "14" hoac "14_pred"
    for suffix in ["", "_pred"]:
        for ext in IMAGE_EXTS:
            candidate = os.path.join(image_dir, base_name + suffix + ext)
            if os.path.exists(candidate):
                return candidate
    return None


def main():
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
        preview_path = find_matching_image(PREVIEW_DIR, base)
        if preview_path is None:
            print(f"[bo qua] khong tim thay mask khop voi '{base}' trong {PREVIEW_DIR}")
            n_skipped += 1
            continue

        pred_mask = load_preview_label_mask(preview_path)
        h, w = pred_mask.shape[:2]

        gt_mask = render_gt_label_mask(svg_path, width=w, height=h)
        if gt_mask is None:
            n_skipped += 1
            continue

        ignore_mask = build_boundary_ignore_mask(gt_mask, DEFAULT_BOUNDARY_TOLERANCE_PX)

        cm = confusion_matrix_3class(pred_mask, gt_mask, ignore_mask)
        per_class = metrics_from_confusion(cm)
        mean_iou = np.nanmean([per_class[LABEL_FILL]["iou"], per_class[LABEL_SATIN]["iou"]])
        mean_f1 = np.nanmean([per_class[LABEL_FILL]["f1"], per_class[LABEL_SATIN]["f1"]])
        s_total, s_correct, s_mismatches = shape_level_stats(pred_mask, gt_mask)

        all_cm += cm
        per_image_iou.append(mean_iou)
        per_image_f1.append(mean_f1)
        total_shapes += s_total
        total_correct += s_correct
        n_evaluated += 1

        shape_acc = s_correct / s_total if s_total else float("nan")
        print(f"{base:30s}  meanIoU={mean_iou:.3f}  meanF1={mean_f1:.3f}  "
              f"shape_acc={shape_acc:.3f} ({s_correct}/{s_total})")
        if s_mismatches:
            for area, gt_l, pred_l in sorted(s_mismatches, reverse=True)[:5]:
                print(f"    -> shape sai: area={area:>8d}px  gt={gt_l:6s}  pred={pred_l}")

    print("\n" + "=" * 60)
    print(f"DA DANH GIA: {n_evaluated} anh  (bo qua: {n_skipped})")
    print(f"(Da loai bo nhieu viem voi tolerance = {DEFAULT_BOUNDARY_TOLERANCE_PX}px)")
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


if __name__ == "__main__":
    main()
