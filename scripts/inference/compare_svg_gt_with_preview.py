#!/usr/bin/env python3
"""
So sanh SVG co nhan GT voi anh PREVIEW da chay san (KHONG chay lai classify)
==============================================================================

CHI CAN SUA 2 DUONG DAN BEN DUOI (SVG_DIR va PREVIEW_DIR) roi chay:
    python compare_svg_gt_with_preview.py

Quy uoc ghep cap: file "logo1.svg" trong SVG_DIR se duoc ghep voi file
"logo1.png" (hoac ten trung, duoi bat ky trong IMAGE_EXTS) trong PREVIEW_DIR.

Anh preview phai la anh xuat ra tu save_preview() cua opencv_stitch_classifier.py
(chi co 3 mau: den=nen, vang that=fill, magenta=satin).
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
# SUA 2 DUONG DAN NAY CHO DUNG MAY BAN
# =============================================================================
SVG_DIR = "data/opencv_test/svg"
PREVIEW_DIR = "data/opencv_test/predictions"
# =============================================================================

LABEL_BACKGROUND, LABEL_FILL, LABEL_SATIN = 0, 1, 2
CLASS_NAMES = {LABEL_BACKGROUND: "background", LABEL_FILL: "fill", LABEL_SATIN: "satin"}
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Mau preview theo dung quy uoc cua save_preview() trong opencv_stitch_classifier.py
# (doc bang cv2.imread -> mang BGR)
_PREVIEW_FILL_BGR = (0, 255, 255)     # hien thi la mau VANG
_PREVIEW_SATIN_BGR = (255, 0, 255)    # hien thi la mau MAGENTA


def _tag(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def guess_label(path_elem: ET.Element) -> Optional[str]:
    """Doan nhan GT cua 1 path. SUA HAM NAY neu file SVG cua ban dung quy uoc khac."""
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
        color = "#FF0000" if label == "satin" else "#0000FF"  # danh dau noi bo, khong lien quan mau hien thi
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


def load_preview_label_mask(path: str) -> np.ndarray:
    """Doc anh preview (da chay san tu opencv_stitch_classifier.py) -> label-mask 0/1/2."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    label_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    label_mask[np.all(img == _PREVIEW_FILL_BGR, axis=2)] = LABEL_FILL
    label_mask[np.all(img == _PREVIEW_SATIN_BGR, axis=2)] = LABEL_SATIN
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


def find_matching_image(image_dir: str, base_name: str) -> Optional[str]:
    for ext in IMAGE_EXTS:
        candidate = os.path.join(image_dir, base_name + ext)
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
            print(f"[bo qua] khong tim thay preview khop voi '{base}' trong {PREVIEW_DIR}")
            n_skipped += 1
            continue

        pred_mask = load_preview_label_mask(preview_path)
        h, w = pred_mask.shape[:2]

        gt_mask = render_gt_label_mask(svg_path, width=w, height=h)
        if gt_mask is None:
            n_skipped += 1
            continue

        cm = confusion_matrix_3class(pred_mask, gt_mask)
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