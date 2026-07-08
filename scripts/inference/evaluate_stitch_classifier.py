#!/usr/bin/env python3
"""
Đánh giá IoU / Precision / Recall / F1 cho opencv_stitch_classifier.py
========================================================================

Bài toán là phân loại 3 lớp trên từng pixel: BACKGROUND(0) / FILL(1) / SATIN(2).
Script này so sánh 1 mask DỰ ĐOÁN (từ classify_multicolor_image) với 1 mask
GROUND TRUTH (do người có chuyên môn tô sửa tay) và tính:

  - Confusion matrix 3x3
  - Per-class: IoU, Precision, Recall, F1
  - Mean IoU / Mean F1 (macro, KHÔNG tính background vì background gần như
    luôn khớp 100% -- tính vào sẽ làm số liệu "đẹp giả tạo")
  - Shape-level accuracy: trong số các SHAPE (contour) ở ground truth,
    bao nhiêu % được gán ĐÚNG nhãn -- chỉ số này quan trọng hơn pixel IoU
    với bài toán này, vì 1 chữ "i" nhỏ xíu bị sai nhãn cũng nghiêm trọng
    y hệt 1 mảng nền lớn bị sai, trong khi pixel-IoU sẽ đánh giá thấp tầm
    quan trọng của các shape nhỏ.

CÁCH DÙNG
---------
1) So sánh 1 cặp file:
    python evaluate_stitch_classifier.py --pred preview.png --gt preview_gt.png

2) So sánh cả 1 bộ nhiều logo (thư mục), quy ước tên file:
    preds/    logo1.png  logo2.png ...
    gts/      logo1.png  logo2.png ...
    python evaluate_stitch_classifier.py --pred-dir preds --gt-dir gts

Preview PNG chỉ có 3 màu (đúng theo save_preview() của opencv_stitch_classifier.py):
    nền   -> (0,0,0)       đen
    fill  -> (0,255,255)   cyan  (lưu ý: hiển thị ra sẽ thành VÀNG do BGR/RGB)
    satin -> (255,0,255)   magenta
"""

import argparse
import glob
import os
import sys
from typing import Dict, Tuple

import cv2
import numpy as np

LABEL_BACKGROUND = 0
LABEL_FILL = 1
LABEL_SATIN = 2
CLASS_NAMES = {LABEL_BACKGROUND: "background", LABEL_FILL: "fill", LABEL_SATIN: "satin"}

_COLOR_FILL = (0, 255, 255)
_COLOR_SATIN = (255, 0, 255)


# ---------------------------------------------------------------------------
# Đọc preview PNG (3 màu cố định) -> label mask 0/1/2
# ---------------------------------------------------------------------------
def preview_to_label_mask(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)

    label_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    label_mask[np.all(img == _COLOR_FILL, axis=2)] = LABEL_FILL
    label_mask[np.all(img == _COLOR_SATIN, axis=2)] = LABEL_SATIN
    return label_mask


# ---------------------------------------------------------------------------
# Pixel-level: confusion matrix + IoU/Precision/Recall/F1
# ---------------------------------------------------------------------------
def confusion_matrix_3class(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """cm[i, j] = số pixel có ground-truth=i nhưng bị dự đoán=j."""
    cm = np.zeros((3, 3), dtype=np.int64)
    for gt_c in range(3):
        for pred_c in range(3):
            cm[gt_c, pred_c] = int(np.sum((gt == gt_c) & (pred == pred_c)))
    return cm


def metrics_from_confusion(cm: np.ndarray) -> Dict[int, Dict[str, float]]:
    """Tính IoU/Precision/Recall/F1 cho từng lớp từ confusion matrix."""
    results = {}
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp   # bị dự đoán là c nhưng gt không phải c
        fn = cm[c, :].sum() - tp   # gt là c nhưng không được dự đoán là c

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
              else float("nan"))

        results[c] = {"iou": iou, "precision": precision, "recall": recall, "f1": f1}
    return results


# ---------------------------------------------------------------------------
# Shape-level: mỗi contour trong ground truth được gán đúng/sai
# ---------------------------------------------------------------------------
def shape_level_accuracy(pred: np.ndarray, gt: np.ndarray,
                          min_area_px: int = 4) -> Dict[str, float]:
    """
    Voi moi SHAPE (connected component) trong ground truth -- tach RIENG theo
    tung lop (fill va satin la 2 mask nhi phan doc lap, KHONG gop chung
    thanh 1 khoi foreground, vi lam vay se khien toan bo chu+nen+vien dinh
    thanh 1 khoi duy nhat do background chi nam o ngoai ria logo) -- so nhan
    voi PRED tai dung vi tri do.

    Tra ve accuracy tinh theo SO LUONG SHAPE (khong theo dien tich pixel) --
    1 chu "i" nho sai cung tinh la 1 loi, y het 1 mang nen lon sai.
    """
    total = 0
    correct = 0
    mismatches = []

    for gt_label, other_label in [(LABEL_FILL, LABEL_SATIN), (LABEL_SATIN, LABEL_FILL)]:
        class_mask = (gt == gt_label).astype(np.uint8)
        n_labels, components = cv2.connectedComponents(class_mask, connectivity=8)

        for comp_id in range(1, n_labels):
            comp_mask = components == comp_id
            area = int(comp_mask.sum())
            if area < min_area_px:
                continue

            pred_labels_in_shape = pred[comp_mask]
            n_fill = np.sum(pred_labels_in_shape == LABEL_FILL)
            n_satin = np.sum(pred_labels_in_shape == LABEL_SATIN)
            if n_fill == 0 and n_satin == 0:
                pred_label = LABEL_BACKGROUND
            else:
                pred_label = LABEL_SATIN if n_satin > n_fill else LABEL_FILL

            total += 1
            if pred_label == gt_label:
                correct += 1
            else:
                mismatches.append((area, CLASS_NAMES[gt_label], CLASS_NAMES[pred_label]))

    accuracy = correct / total if total > 0 else float("nan")
    return {"total_shapes": total, "correct_shapes": correct,
            "shape_accuracy": accuracy, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# In báo cáo cho 1 cặp pred/gt
# ---------------------------------------------------------------------------
def evaluate_pair(pred_path: str, gt_path: str, verbose: bool = True) -> Dict:
    pred = preview_to_label_mask(pred_path)
    gt = preview_to_label_mask(gt_path)

    if pred.shape != gt.shape:
        # Resize pred về đúng kích thước gt (đề phòng lệch canvas)
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    cm = confusion_matrix_3class(pred, gt)
    per_class = metrics_from_confusion(cm)
    shape_result = shape_level_accuracy(pred, gt)

    # Mean IoU/F1 CHỈ trên fill + satin (bỏ qua background cho đỡ "ảo")
    mean_iou_fg = np.nanmean([per_class[LABEL_FILL]["iou"], per_class[LABEL_SATIN]["iou"]])
    mean_f1_fg = np.nanmean([per_class[LABEL_FILL]["f1"], per_class[LABEL_SATIN]["f1"]])

    if verbose:
        print(f"\n=== {os.path.basename(pred_path)} ===")
        print("Confusion matrix (hàng=GT, cột=Pred), thứ tự [bg, fill, satin]:")
        print(cm)
        for c in range(3):
            m = per_class[c]
            print(f"  {CLASS_NAMES[c]:10s}: IoU={m['iou']:.3f}  "
                  f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
        print(f"  Mean IoU (fill+satin)={mean_iou_fg:.3f}  Mean F1 (fill+satin)={mean_f1_fg:.3f}")
        print(f"  Shape-level accuracy: {shape_result['correct_shapes']}/"
              f"{shape_result['total_shapes']} = {shape_result['shape_accuracy']:.3f}")
        if shape_result["mismatches"]:
            print("  Shape bị sai (area_px, gt_label, pred_label), lớn->nhỏ:")
            for area, gt_l, pred_l in sorted(shape_result["mismatches"], reverse=True)[:10]:
                print(f"    area={area:>8d}px  gt={gt_l:6s}  pred={pred_l}")

    return {
        "confusion_matrix": cm,
        "per_class": per_class,
        "mean_iou_fg": mean_iou_fg,
        "mean_f1_fg": mean_f1_fg,
        "shape_result": shape_result,
    }


# ---------------------------------------------------------------------------
# Đánh giá cả bộ (nhiều file), gộp theo 2 cách: macro (trung bình mỗi ảnh)
# và micro (gộp toàn bộ pixel/shape của mọi ảnh lại rồi tính 1 lần)
# ---------------------------------------------------------------------------
def evaluate_dataset(pred_dir: str, gt_dir: str) -> None:
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    if not gt_files:
        print(f"Khong tim thay file .png nao trong {gt_dir}")
        return

    all_cm = np.zeros((3, 3), dtype=np.int64)
    per_image_mean_iou = []
    per_image_mean_f1 = []
    total_shapes = 0
    total_correct_shapes = 0

    for gt_path in gt_files:
        name = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, name)
        if not os.path.exists(pred_path):
            print(f"[bo qua] khong co pred tuong ung cho {name}")
            continue

        result = evaluate_pair(pred_path, gt_path, verbose=False)
        all_cm += result["confusion_matrix"]
        per_image_mean_iou.append(result["mean_iou_fg"])
        per_image_mean_f1.append(result["mean_f1_fg"])
        total_shapes += result["shape_result"]["total_shapes"]
        total_correct_shapes += result["shape_result"]["correct_shapes"]

        print(f"{name:30s}  meanIoU={result['mean_iou_fg']:.3f}  "
              f"meanF1={result['mean_f1_fg']:.3f}  "
              f"shape_acc={result['shape_result']['shape_accuracy']:.3f} "
              f"({result['shape_result']['correct_shapes']}/{result['shape_result']['total_shapes']})")

    print("\n" + "=" * 60)
    print(f"TONG SO ANH DANH GIA: {len(per_image_mean_iou)}")

    print("\n--- MACRO (trung binh cong moi anh, moi anh trong so bang nhau) ---")
    print(f"Mean IoU (fill+satin): {np.nanmean(per_image_mean_iou):.3f}")
    print(f"Mean F1  (fill+satin): {np.nanmean(per_image_mean_f1):.3f}")

    print("\n--- MICRO (gop tat ca pixel cua moi anh lai, tinh 1 lan) ---")
    micro_metrics = metrics_from_confusion(all_cm)
    for c in range(3):
        m = micro_metrics[c]
        print(f"  {CLASS_NAMES[c]:10s}: IoU={m['iou']:.3f}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    print("\n--- SHAPE-LEVEL (gop tat ca shape cua moi anh lai) ---")
    print(f"Tong shape: {total_shapes}, dung: {total_correct_shapes}, "
          f"accuracy = {total_correct_shapes/total_shapes:.3f}" if total_shapes else "khong co shape nao")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", help="File preview du doan (1 cap)")
    parser.add_argument("--gt", help="File preview ground truth (1 cap)")
    parser.add_argument("--pred-dir", help="Thu muc chua cac file du doan (ca bo)")
    parser.add_argument("--gt-dir", help="Thu muc chua cac file ground truth (ca bo)")
    args = parser.parse_args()

    if args.pred and args.gt:
        evaluate_pair(args.pred, args.gt, verbose=True)
    elif args.pred_dir and args.gt_dir:
        evaluate_dataset(args.pred_dir, args.gt_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()