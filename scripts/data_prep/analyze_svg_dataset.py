#!/usr/bin/env python3
"""
SVG Dataset Statistics & Consistency Analyzer (V2 - Nâng cao)
Gộp 3 chức năng:
  1. Thống kê tổng quan dataset (fill/satin paths, top complex SVG, ...)
  2. Kiểm tra consistency từng path (missing/invalid label, path_length, ...)
  3. Visualize phân bố + cảnh báo tự động các file outlier
  4. Phân tích sâu (Imbalance, 1-class, Tiny Paths, Text/Shape, Circular, Score, Gợi ý)
"""

import csv
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import math

import pandas as pd

# ==========================
# CONFIG
# ==========================

TRAIN_DIR = "data/logo/train_svg"
VAL_DIR   = "data/logo/val_svg"          # để trống "" nếu chỉ có train

STATS_CSV       = "dataset_statistics.csv"
CONSISTENCY_CSV = "dataset_consistency.csv"
PLOT_PNG        = "dataset_distribution.png"

INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"

SVG_NS = {
    "svg":      "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
}

VALID_LABELS = {"fill", "satin"}

# Ngưỡng phân tích
OUTLIER_STD_THRESHOLD = 1.5
TINY_PATH_THRESHOLD = 30     # Path có length(d) < 30 coi là quá nhỏ

# ==========================
# CORE: phân tích 1 file SVG
# ==========================

def analyze_svg(svg_path: Path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    path_rows = []
    fill = satin = missing = invalid = tiny_paths = 0

    # 1. Phân tích hình học (Check hình tròn)
    has_circle_tag = len(root.findall(".//svg:circle", SVG_NS)) > 0 or \
                     len(root.findall(".//svg:ellipse", SVG_NS)) > 0
    
    is_circular = has_circle_tag
    viewbox = root.get("viewBox")
    if viewbox:
        try:
            _, _, w, h = map(float, viewbox.split())
            if h > 0 and 0.9 <= (w / h) <= 1.1:
                is_circular = True
        except ValueError:
            pass

    # 2. Phân tích từng Path
    for idx, elem in enumerate(root.findall(".//svg:path", SVG_NS)):
        d     = elem.attrib.get("d", "")
        label = elem.attrib.get(INKSCAPE_LABEL, "").strip().lower()
        d_len = len(d)

        if d_len > 0 and d_len < TINY_PATH_THRESHOLD:
            tiny_paths += 1

        is_missing = label == ""
        is_invalid = label != "" and label not in VALID_LABELS

        if is_missing:
            missing += 1
        elif is_invalid:
            invalid += 1
        elif label == "fill":
            fill += 1
        elif label == "satin":
            satin += 1

        path_rows.append({
            "file":          svg_path.name,
            "path_id":       idx,
            "label":         label,
            "path_length":   d_len,
            "missing_label": is_missing,
            "invalid_label": is_invalid,
        })

    total_paths = fill + satin
    ratio = fill / max(total_paths, 1)

    # Heuristic: Mặc định Satin là Text (theo rule project)
    text_paths = satin
    shape_paths = fill

    # 3. Tính Easy Score (100 là dễ nhất)
    # Trừ điểm nếu: nhiều path, nhiều tiny path, quá mất cân bằng
    score = 100
    score -= (total_paths * 0.5)      # Nhiều path -> Rối
    score -= (tiny_paths * 2.0)       # Path nhỏ -> Khó học
    imbalance = abs(ratio - 0.5) * 2  # 0 là cân bằng, 1 là full 1 class
    score -= (imbalance * 15)         # Phạt 15 điểm nếu lệch hoàn toàn
    
    easy_score = max(0, min(100, int(score)))

    file_stats = {
        "svg":          svg_path.name,
        "fill_paths":   fill,
        "satin_paths":  satin,
        "total_paths":  total_paths,
        "tiny_paths":   tiny_paths,
        "missing":      missing,
        "invalid":      invalid,
        "fill_ratio":   ratio,
        "satin_ratio":  1 - ratio,
        "is_circular":  is_circular,
        "text_paths":   text_paths,
        "shape_paths":  shape_paths,
        "has_text":     text_paths > 0,
        "easy_score":   easy_score
    }

    return path_rows, file_stats

# ==========================
# CORE: quét 1 thư mục
# ==========================

def scan_folder(folder: str):
    folder = Path(folder)
    if not folder.exists():
        return [], [], Counter()

    svg_files = sorted(folder.glob("*.svg"))
    print(f"\nScanning {folder}  ({len(svg_files)} files)…")

    all_path_rows  = []
    all_file_stats = []
    summary        = Counter()

    for svg in svg_files:
        try:
            path_rows, file_stats = analyze_svg(svg)
        except Exception as e:
            print(f"  [ERROR] {svg.name}: {e}")
            continue

        all_path_rows.extend(path_rows)
        all_file_stats.append(file_stats)

        for key in ("fill_paths", "satin_paths", "total_paths", "missing", "invalid", "tiny_paths"):
            summary[key] += file_stats[key]
            
        if file_stats["is_circular"]: summary["circular_logos"] += 1
        if file_stats["has_text"]: summary["text_logos"] += 1

    summary["files"] = len(all_file_stats)
    return all_path_rows, all_file_stats, summary

# ==========================
# REPORT helpers (Nâng cao)
# ==========================

def print_deep_analysis(stats_list: list[dict], summary: Counter):
    """In ra 8 mục phân tích chuyên sâu"""
    if not stats_list: return

    almost_satin = []
    almost_fill = []
    balanced = []
    only_satin = []
    only_fill = []
    tiny_offenders = []
    
    for s in stats_list:
        # Class imbalance
        if s["satin_ratio"] > 0.9: almost_satin.append(s)
        elif s["fill_ratio"] > 0.9: almost_fill.append(s)
        elif 0.4 <= s["fill_ratio"] <= 0.6: balanced.append(s)
        
        # 1-class logos
        if s["satin_paths"] > 0 and s["fill_paths"] == 0: only_satin.append(s)
        if s["fill_paths"] > 0 and s["satin_paths"] == 0: only_fill.append(s)
        
        # Tiny paths
        if s["tiny_paths"] > 0: tiny_offenders.append(s)

    tiny_offenders.sort(key=lambda x: x["tiny_paths"], reverse=True)
    df = pd.DataFrame(stats_list)

    print(f"\n==============================================================")
    print(f" DEEP ANALYSIS REPORT")
    print(f"==============================================================")
    
    # 1. Imbalance & 2. One-class
    print(f"\n1. CLASS IMBALANCE (Per SVG)")
    print(f"  SVG gần như toàn SATIN (>90%) : {len(almost_satin):>3} file")
    print(f"  SVG gần như toàn FILL  (>90%) : {len(almost_fill):>3} file")
    print(f"  SVG cân bằng (40-60%)         : {len(balanced):>3} file")
    
    print(f"\n2. SINGLE CLASS LOGOS")
    print(f"  ONLY SATIN LOGOS ({len(only_satin)}): " + ", ".join([s["svg"] for s in only_satin[:5]]) + ("..." if len(only_satin) > 5 else ""))
    print(f"  ONLY FILL LOGOS  ({len(only_fill)}): " + ", ".join([s["svg"] for s in only_fill[:5]]) + ("..." if len(only_fill) > 5 else ""))

    # 3. Tiny Paths
    print(f"\n3. TINY PATHS (len < {TINY_PATH_THRESHOLD})")
    if tiny_offenders:
        for s in tiny_offenders[:5]:
            print(f"  {s['svg']:<10} : {s['tiny_paths']} tiny paths")
    else:
        print("  Tuyệt vời! Không có path nào quá nhỏ.")

    # 4. Text vs Shape & 6. Has Text
    print(f"\n4 & 6. TEXT ANALYSIS (Assumed Satin = Text)")
    print(f"  Logo có chữ    : {summary['text_logos']}")
    print(f"  Logo không chữ : {summary['files'] - summary['text_logos']}")
    print(f"  Tổng Text paths (Satin) : {summary['satin_paths']}")
    print(f"  Tổng Shape paths (Fill) : {summary['fill_paths']}")

    # 5. Circular vs Rectangular
    print(f"\n5. SHAPE DISTRIBUTION")
    print(f"  Circular logos    : {summary['circular_logos']}")
    print(f"  Rectangular/Other : {summary['files'] - summary['circular_logos']}")

    # 7. Easy Score
    print(f"\n7. EASY SCORE (100 = Dễ nhất)")
    df_sorted_score = df.sort_values("easy_score", ascending=False)
    print("  Top 3 DỄ NHẤT:")
    for _, r in df_sorted_score.head(3).iterrows():
        print(f"    - {r['svg']:<10} Score: {r['easy_score']:>3} (Paths: {r['total_paths']}, Tiny: {r['tiny_paths']})")
    print("  Top 3 KHÓ NHẤT:")
    for _, r in df_sorted_score.tail(3).iterrows():
        print(f"    - {r['svg']:<10} Score: {r['easy_score']:>3} (Paths: {r['total_paths']}, Tiny: {r['tiny_paths']})")

    # 8. Suggestions
    print(f"\n==============================================================")
    print(f" DATASET SUGGESTION (Auto-Generated)")
    print(f"==============================================================")
    
    total_paths = summary["fill_paths"] + summary["satin_paths"]
    if total_paths == 0: return
    
    global_satin_ratio = summary["satin_paths"] / total_paths
    global_fill_ratio = summary["fill_paths"] / total_paths
    
    print(f"⚠ Đang phân bổ: Satin = {global_satin_ratio*100:.1f}% | Fill = {global_fill_ratio*100:.1f}%")
    
    print("\n✔ Nên bổ sung thêm:")
    if summary['circular_logos'] < summary['files'] * 0.3:
        target_circle = int((summary['files'] * 0.3) - summary['circular_logos'])
        print(f"    + ~{target_circle} logo tròn (hiện đang thiếu)")
    
    if len(only_fill) < len(only_satin):
        print(f"    + ~{len(only_satin) - len(only_fill)} logo chỉ có Fill (để cân bằng với số lượng Only Satin)")
        
    if len(balanced) < summary['files'] * 0.4:
        print(f"    + ~{int((summary['files'] * 0.4) - len(balanced))} logo cân bằng Fill/Satin (Tránh model bias)")
        
    if (summary['files'] - summary['text_logos']) < summary['files'] * 0.3:
        print(f"    + ~{int((summary['files'] * 0.3) - (summary['files'] - summary['text_logos']))} logo KHÔNG có chữ")

    print("\nKhuyến nghị Action:")
    if global_satin_ratio > 0.6:
        print("    → GIẢM các logo chứa quá nhiều text/satin.")
        print("    → TĂNG các logo dạng flat vector, có mảng Fill lớn.")
    elif global_fill_ratio > 0.6:
        print("    → GIẢM các logo flat vector.")
        print("    → TĂNG các logo có nhiều text/đường viền satin.")
    else:
        print("    → Tỷ lệ class tổng thể đang khá ổn định!")


def print_outlier_warnings(df: pd.DataFrame, source_label: str):
    if df.empty: return
    mean = df["total_paths"].mean()
    std  = df["total_paths"].std()
    if std == 0 or pd.isna(std): return

    low_thresh  = mean - OUTLIER_STD_THRESHOLD * std
    high_thresh = mean + OUTLIER_STD_THRESHOLD * std

    too_few  = df[df["total_paths"] < low_thresh].sort_values("total_paths")
    too_many = df[df["total_paths"] > high_thresh].sort_values("total_paths", ascending=False)

    print(f"\n{'='*62}")
    print(f"  OUTLIER WARNINGS – {source_label}")
    print(f"{'='*62}")

    if not too_few.empty:
        print(f"  ⚠ Quá ÍT path ({len(too_few)} file):")
        for _, row in too_few.head(3).iterrows():
            print(f"     - {row['svg']:<15} total_paths={row['total_paths']}")

    if not too_many.empty:
        print(f"\n  ⚠ Quá NHIỀU path ({len(too_many)} file):")
        for _, row in too_many.head(3).iterrows():
            print(f"     - {row['svg']:<15} total_paths={row['total_paths']}")

# (Giữ nguyên hàm plot_distribution như cũ, không thay đổi)
def plot_distribution(df_train: pd.DataFrame, df_val: pd.DataFrame, output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    has_train = not df_train.empty
    has_val   = not df_val.empty
    if not has_train and not has_val: return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("SVG Dataset – Phân bố & So sánh", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    if has_train: ax.hist(df_train["total_paths"], bins=12, alpha=0.6, label="train")
    if has_val: ax.hist(df_val["total_paths"], bins=12, alpha=0.6, label="val")
    ax.set_title("Phân bố số path")
    ax.legend()

    ax = axes[0, 1]
    box_data, box_labels = [], []
    if has_train: box_data.append(df_train["total_paths"]); box_labels.append("train")
    if has_val: box_data.append(df_val["total_paths"]); box_labels.append("val")
    ax.boxplot(box_data, tick_labels=box_labels)
    ax.set_title("Độ phức tạp")

    ax = axes[1, 0]
    ax.axis('off') # Chừa không gian hoặc vẽ chart khác tùy ý

    ax = axes[1, 1]
    if has_train: ax.scatter(df_train["fill_paths"], df_train["satin_paths"], label="train", alpha=0.7)
    if has_val: ax.scatter(df_val["fill_paths"], df_val["satin_paths"], label="val", alpha=0.7)
    ax.set_title("Fill vs Satin")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

# ==========================
# MAIN
# ==========================

def main():
    train_path_rows, train_file_stats, train_summary = scan_folder(TRAIN_DIR)
    val_path_rows,   val_file_stats,   val_summary   = scan_folder(VAL_DIR) if VAL_DIR else ([], [], Counter())

    all_path_rows  = train_path_rows  + val_path_rows
    all_file_stats = train_file_stats + val_file_stats

    combined_summary = Counter()
    for key in train_summary:
        combined_summary[key] = train_summary[key] + val_summary[key]

    df_stats = pd.DataFrame(all_file_stats).sort_values("total_paths", ascending=False)
    df_stats.to_csv(STATS_CSV, index=False)
    
    if all_path_rows:
        fieldnames = ["file", "path_id", "label", "path_length", "missing_label", "invalid_label"]
        with open(CONSISTENCY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_path_rows)

    # In Report Nâng Cao
    if all_file_stats:
        print_deep_analysis(all_file_stats, combined_summary)

    df_train = pd.DataFrame(train_file_stats) if train_file_stats else pd.DataFrame()
    df_val   = pd.DataFrame(val_file_stats)   if val_file_stats   else pd.DataFrame()

    if not df_train.empty: print_outlier_warnings(df_train, "TRAIN")
    if not df_val.empty: print_outlier_warnings(df_val, "VAL")

    plot_distribution(df_train, df_val, PLOT_PNG)

if __name__ == "__main__":
    main()
