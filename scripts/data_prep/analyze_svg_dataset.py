#!/usr/bin/env python3
"""
SVG Dataset Statistics & Consistency Analyzer
Gộp 3 chức năng:
  1. Thống kê tổng quan dataset (fill/satin paths, top complex SVG, ...)
  2. Kiểm tra consistency từng path (missing/invalid label, path_length, ...)
  3. Visualize phân bố + cảnh báo tự động các file outlier (quá ít/quá nhiều path)
"""

import csv
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

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

# Ngưỡng cảnh báo outlier: số độ lệch chuẩn (std) so với trung bình (mean)
OUTLIER_STD_THRESHOLD = 1.5

# ==========================
# CORE: phân tích 1 file SVG
# ==========================

def analyze_svg(svg_path: Path):
    """
    Trả về:
      path_rows  – list[dict] thông tin từng <path>   (cho consistency CSV)
      file_stats – dict tổng hợp cho file này          (cho statistics CSV + báo cáo)
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    path_rows = []
    fill = satin = missing = invalid = 0

    for idx, elem in enumerate(root.findall(".//svg:path", SVG_NS)):
        d     = elem.attrib.get("d", "")
        label = elem.attrib.get(INKSCAPE_LABEL, "").strip().lower()

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
            "path_length":   len(d),
            "missing_label": is_missing,
            "invalid_label": is_invalid,
        })

    file_stats = {
        "svg":         svg_path.name,
        "fill_paths":  fill,
        "satin_paths": satin,
        "total_paths": fill + satin,
        "missing":     missing,
        "invalid":     invalid,
    }

    return path_rows, file_stats

# ==========================
# CORE: quét 1 thư mục
# ==========================

def scan_folder(folder: str):
    """Quét tất cả *.svg trong folder, trả về rows + summary Counter."""
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

        for key in ("fill_paths", "satin_paths", "total_paths", "missing", "invalid"):
            summary[key] += file_stats[key]

    summary["files"] = len(all_file_stats)
    return all_path_rows, all_file_stats, summary

# ==========================
# REPORT helpers
# ==========================

def print_folder_report(label: str, stats_list: list[dict], summary: Counter):
    total_svg   = summary["files"]
    total_paths = summary["total_paths"]
    total_fill  = summary["fill_paths"]
    total_satin = summary["satin_paths"]

    only_fill = only_satin = both = empty = 0
    for s in stats_list:
        f, sa = s["fill_paths"], s["satin_paths"]
        if f > 0 and sa == 0:   only_fill  += 1
        elif sa > 0 and f == 0: only_satin += 1
        elif f > 0 and sa > 0:  both       += 1
        else:                   empty      += 1

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  REPORT – {label}")
    print(sep)
    print(f"  SVG files            : {total_svg}")
    print(f"  Total paths          : {total_paths}")
    print(f"  Fill paths           : {total_fill}")
    print(f"  Satin paths          : {total_satin}")
    if total_paths > 0:
        print(f"  Fill ratio           : {100*total_fill/total_paths:.2f}%")
        print(f"  Satin ratio          : {100*total_satin/total_paths:.2f}%")
    if total_svg > 0:
        print(f"  Avg paths / SVG      : {total_paths/total_svg:.2f}")
        print(f"  Avg fill  / SVG      : {total_fill/total_svg:.2f}")
        print(f"  Avg satin / SVG      : {total_satin/total_svg:.2f}")
    print(f"  SVG only Fill        : {only_fill}")
    print(f"  SVG only Satin       : {only_satin}")
    print(f"  SVG both classes     : {both}")
    print(f"  SVG empty            : {empty}")
    print(f"  Missing label paths  : {summary['missing']}")
    print(f"  Invalid label paths  : {summary['invalid']}")


def print_top_complex(stats_list: list[dict], n: int = 20):
    df = pd.DataFrame(stats_list).sort_values("total_paths", ascending=False)
    print(f"\n{'='*62}")
    print(f"  TOP {n} MOST COMPLEX SVG")
    print(f"{'='*62}")
    cols = ["svg", "fill_paths", "satin_paths", "total_paths", "missing", "invalid"]
    print(df[cols].head(n).to_string(index=False))

    print(f"\n{'='*62}")
    print("  PATH COUNT DISTRIBUTION (total_paths)")
    print(f"{'='*62}")
    print(df["total_paths"].describe().to_string())


def print_outlier_warnings(df: pd.DataFrame, source_label: str):
    """
    Cảnh báo các file có total_paths quá lệch so với trung bình
    (dùng ngưỡng mean ± OUTLIER_STD_THRESHOLD * std).
    """
    if df.empty:
        return

    mean = df["total_paths"].mean()
    std  = df["total_paths"].std()

    if std == 0 or pd.isna(std):
        return

    low_thresh  = mean - OUTLIER_STD_THRESHOLD * std
    high_thresh = mean + OUTLIER_STD_THRESHOLD * std

    too_few  = df[df["total_paths"] < low_thresh].sort_values("total_paths")
    too_many = df[df["total_paths"] > high_thresh].sort_values("total_paths", ascending=False)

    print(f"\n{'='*62}")
    print(f"  OUTLIER WARNINGS – {source_label}")
    print(f"  (ngưỡng: mean={mean:.1f} ± {OUTLIER_STD_THRESHOLD}×std={std:.1f}  "
          f"→ [{max(low_thresh,0):.1f}, {high_thresh:.1f}])")
    print(f"{'='*62}")

    if too_few.empty and too_many.empty:
        print("  Không có file nào vượt ngưỡng — phân bố tương đối đồng đều.")
        return

    if not too_few.empty:
        print(f"\n  ⚠ Quá ÍT path ({len(too_few)} file):")
        for _, row in too_few.iterrows():
            print(f"     - {row['svg']:<15} total_paths={row['total_paths']}"
                  f"  (fill={row['fill_paths']}, satin={row['satin_paths']})")

    if not too_many.empty:
        print(f"\n  ⚠ Quá NHIỀU path ({len(too_many)} file):")
        for _, row in too_many.iterrows():
            print(f"     - {row['svg']:<15} total_paths={row['total_paths']}"
                  f"  (fill={row['fill_paths']}, satin={row['satin_paths']})")


# ==========================
# VISUALIZE
# ==========================

def plot_distribution(df_train: pd.DataFrame, df_val: pd.DataFrame, output_path: str):
    """
    Vẽ 4 subplot:
      1. Histogram total_paths (train vs val)
      2. Boxplot so sánh train vs val
      3. Bar chart fill vs satin ratio (train vs val)
      4. Scatter fill_paths vs satin_paths (toàn bộ, tô màu theo nguồn)
    Bỏ qua nhẹ nhàng nếu thiếu matplotlib (không crash toàn script).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # không cần display, an toàn khi chạy qua SSH/headless
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[SKIP] Không tìm thấy matplotlib — bỏ qua bước visualize.")
        print("        Cài bằng: pip install matplotlib")
        return

    has_train = not df_train.empty
    has_val   = not df_val.empty

    if not has_train and not has_val:
        print("\n[SKIP] Không có dữ liệu để vẽ.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("SVG Dataset – Phân bố & So sánh Train/Val", fontsize=14, fontweight="bold")

    # --- 1. Histogram total_paths ---
    ax = axes[0, 0]
    bins = 12
    if has_train:
        ax.hist(df_train["total_paths"], bins=bins, alpha=0.6, label="train", color="#4C72B0")
    if has_val:
        ax.hist(df_val["total_paths"], bins=bins, alpha=0.6, label="val", color="#DD8452")
    ax.set_title("Phân bố số path / SVG")
    ax.set_xlabel("total_paths")
    ax.set_ylabel("số lượng SVG")
    ax.legend()

    # --- 2. Boxplot so sánh train vs val ---
    ax = axes[0, 1]
    box_data, box_labels = [], []
    if has_train:
        box_data.append(df_train["total_paths"])
        box_labels.append("train")
    if has_val:
        box_data.append(df_val["total_paths"])
        box_labels.append("val")
    ax.boxplot(box_data, tick_labels=box_labels, showmeans=True)
    ax.set_title("So sánh độ phức tạp Train vs Val")
    ax.set_ylabel("total_paths")

    # --- 3. Bar chart fill vs satin ratio ---
    ax = axes[1, 0]
    groups, fill_vals, satin_vals = [], [], []
    for name, d in (("train", df_train), ("val", df_val)):
        if d.empty:
            continue
        total = d["fill_paths"].sum() + d["satin_paths"].sum()
        if total == 0:
            continue
        groups.append(name)
        fill_vals.append(100 * d["fill_paths"].sum() / total)
        satin_vals.append(100 * d["satin_paths"].sum() / total)

    if groups:
        x = range(len(groups))
        ax.bar(x, fill_vals, label="fill %", color="#55A868")
        ax.bar(x, satin_vals, bottom=fill_vals, label="satin %", color="#C44E52")
        ax.set_xticks(list(x))
        ax.set_xticklabels(groups)
        ax.set_ylim(0, 100)
        ax.set_title("Tỷ lệ Fill vs Satin (%)")
        ax.set_ylabel("%")
        ax.legend()

    # --- 4. Scatter fill_paths vs satin_paths ---
    ax = axes[1, 1]
    if has_train:
        ax.scatter(df_train["fill_paths"], df_train["satin_paths"],
                   label="train", alpha=0.7, color="#4C72B0")
    if has_val:
        ax.scatter(df_val["fill_paths"], df_val["satin_paths"],
                   label="val", alpha=0.7, color="#DD8452")
    ax.set_title("Fill paths vs Satin paths")
    ax.set_xlabel("fill_paths")
    ax.set_ylabel("satin_paths")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n[OK] Biểu đồ phân bố → {output_path}")


# ==========================
# MAIN
# ==========================

def main():
    # --- Quét dữ liệu ---
    train_path_rows, train_file_stats, train_summary = scan_folder(TRAIN_DIR)
    val_path_rows,   val_file_stats,   val_summary   = scan_folder(VAL_DIR) if VAL_DIR else ([], [], Counter())

    all_path_rows  = train_path_rows  + val_path_rows
    all_file_stats = train_file_stats + val_file_stats

    combined_summary = Counter()
    for key in ("files", "fill_paths", "satin_paths", "total_paths", "missing", "invalid"):
        combined_summary[key] = train_summary[key] + val_summary[key]

    # --- Statistics CSV ---
    df_stats = pd.DataFrame(all_file_stats).sort_values("total_paths", ascending=False)
    df_stats.to_csv(STATS_CSV, index=False)
    print(f"\n[OK] Statistics CSV  → {STATS_CSV}")

    # --- Consistency CSV ---
    if all_path_rows:
        fieldnames = ["file", "path_id", "label", "path_length", "missing_label", "invalid_label"]
        with open(CONSISTENCY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_path_rows)
        print(f"[OK] Consistency CSV → {CONSISTENCY_CSV}")

    # --- In báo cáo ---
    if train_file_stats:
        print_folder_report("TRAIN", train_file_stats, train_summary)
    if val_file_stats:
        print_folder_report("VAL",   val_file_stats,   val_summary)
    if train_file_stats and val_file_stats:
        print_folder_report("TOTAL (TRAIN + VAL)", all_file_stats, combined_summary)

    print_top_complex(all_file_stats, n=20)

    # --- Cảnh báo outlier (tách riêng train/val vì 2 phân bố có thể khác nhau) ---
    df_train = pd.DataFrame(train_file_stats) if train_file_stats else pd.DataFrame()
    df_val   = pd.DataFrame(val_file_stats)   if val_file_stats   else pd.DataFrame()

    if not df_train.empty:
        print_outlier_warnings(df_train, "TRAIN")
    if not df_val.empty:
        print_outlier_warnings(df_val, "VAL")

    # --- Visualize ---
    plot_distribution(df_train, df_val, PLOT_PNG)


if __name__ == "__main__":
    main()