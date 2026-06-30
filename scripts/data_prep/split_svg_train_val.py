#!/usr/bin/env python3
"""
Split SVG files from easy/medium/hard folders into train/val sets.

Khác với bản gốc: việc chia train/val không chỉ dựa vào thư mục nguồn
(easy/medium/hard) mà còn STRATIFY theo độ phức tạp thực tế của từng file
(total_paths = số path có label fill/satin). Điều này tránh tình trạng các
file phức tạp nhất (outlier) bị dồn hết vào 1 phía train hoặc val do
random shuffle với mẫu nhỏ.

Cách làm:
  1. Đọc từng SVG, tính total_paths (đếm path có inkscape:label = fill/satin).
  2. Gộp toàn bộ file từ mọi source_dir lại, chia thành N bucket theo
     quantile của total_paths (mặc định N=4, tức theo quartile).
  3. Trong mỗi bucket, shuffle rồi chia theo train_ratio.
  4. Gộp kết quả của tất cả bucket -> đảm bảo train và val có phân bố
     độ phức tạp tương đương nhau.

Nếu không đọc được total_paths của 1 file (lỗi parse, không có path nào),
file đó được gán vào bucket riêng "unknown" và vẫn được chia theo
train_ratio như các bucket khác, không bị loại bỏ.
"""

import os
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
VALID_LABELS = {"fill", "satin"}


# ==========================
# Đếm total_paths của 1 file
# ==========================

def count_total_paths(svg_path: Path) -> int:
    """Trả về số path có label fill/satin trong 1 file SVG. -1 nếu lỗi đọc."""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  [WARN] Không đọc được {svg_path.name}: {e}")
        return -1

    count = 0
    for elem in root.findall(".//svg:path", SVG_NS):
        label = elem.attrib.get(INKSCAPE_LABEL, "").strip().lower()
        if label in VALID_LABELS:
            count += 1
    return count


# ==========================
# Chia bucket theo quantile
# ==========================

def assign_buckets(files_with_counts: List[tuple], n_buckets: int = 4) -> Dict[int, list]:
    """
    files_with_counts: list of (Path, total_paths)
    Trả về dict {bucket_id: [Path, ...]}, file lỗi (total_paths == -1)
    được gom riêng vào bucket_id = -1.
    """
    valid = [(f, c) for f, c in files_with_counts if c >= 0]
    invalid = [f for f, c in files_with_counts if c < 0]

    buckets: Dict[int, list] = {}

    if invalid:
        buckets[-1] = invalid

    if not valid:
        return buckets

    valid_sorted = sorted(valid, key=lambda x: x[1])
    n = len(valid_sorted)

    # Số bucket không nên nhiều hơn số file, tránh bucket trống vô nghĩa
    n_buckets = max(1, min(n_buckets, n))

    for i, (f, c) in enumerate(valid_sorted):
        bucket_id = min(i * n_buckets // n, n_buckets - 1)
        buckets.setdefault(bucket_id, []).append(f)

    return buckets


# ==========================
# Main split logic
# ==========================

def compute_mean_std(files: list, files_with_counts_map: dict):
    counts = [files_with_counts_map[f] for f in files if files_with_counts_map.get(f, -1) >= 0]
    if not counts:
        return 0.0, 0.0, 0
    mean = sum(counts) / len(counts)
    var = sum((c - mean) ** 2 for c in counts) / len(counts)
    return mean, var ** 0.5, len(counts)


def try_split_for_seed(buckets: Dict[int, list], train_ratio: float, seed: int):
    """Chia 1 lần theo seed cho sẵn, trả về (train_files, val_files)."""
    rng = random.Random(seed)
    train_files, val_files = [], []
    for bucket_id in sorted(buckets.keys()):
        bucket_files = buckets[bucket_id][:]
        rng.shuffle(bucket_files)
        split_idx = round(len(bucket_files) * train_ratio)
        train_files.extend(bucket_files[:split_idx])
        val_files.extend(bucket_files[split_idx:])
    return train_files, val_files


def find_best_seed(buckets: Dict[int, list], files_with_counts: list,
                    train_ratio: float, base_seed: int, n_trials: int = 200):
    """
    Thử n_trials seed khác nhau (base_seed, base_seed+1, ...), chọn seed cho
    chênh lệch mean(total_paths) giữa train/val nhỏ nhất. Cần thiết khi N
    nhỏ, vì 1 outlier rơi vào bên nào cũng có thể kéo lệch mean nặng -
    không phải lỗi logic chia, mà là giới hạn cỡ mẫu nhỏ.
    """
    counts_map = dict(files_with_counts)

    best_seed = base_seed
    best_diff = float("inf")
    best_split = None

    for trial in range(n_trials):
        seed = base_seed + trial
        train_files, val_files = try_split_for_seed(buckets, train_ratio, seed)

        train_mean, _, _ = compute_mean_std(train_files, counts_map)
        val_mean, _, _ = compute_mean_std(val_files, counts_map)

        if train_mean == 0:
            continue

        diff = abs(train_mean - val_mean) / train_mean
        if diff < best_diff:
            best_diff = diff
            best_seed = seed
            best_split = (train_files, val_files)

    return best_seed, best_diff, best_split



def split_svg_files(source_dirs: List[str], train_dir: str, val_dir: str,
                     train_ratio: float = 0.7, seed: int = 42, n_buckets: int = 4,
                     auto_seed: bool = False, n_trials: int = 200):
    """Split SVG files from multiple source directories into train/val sets,
    stratified theo độ phức tạp (total_paths) thay vì chỉ theo thư mục nguồn.

    Quan trọng: bucket được tính trên TOÀN BỘ file gộp từ mọi source_dir,
    không tính riêng theo từng thư mục. Nếu mỗi thư mục easy/medium/hard chỉ
    có vài file, chia bucket riêng theo thư mục sẽ tạo ra bucket 1 file, làm
    train_ratio mất ý nghĩa (1 file luôn rơi hết về 1 phía).

    Nếu auto_seed=True: thử n_trials seed (bắt đầu từ `seed`), chọn seed cho
    chênh lệch mean(total_paths) train/val nhỏ nhất. Hữu ích khi N nhỏ và
    có outlier lớn — một seed cố định có thể vô tình dồn outlier về 1 phía.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_files = []  # list of Path, gộp từ mọi source_dir

    for source_dir in source_dirs:
        source_path = os.path.join(PROJECT_ROOT, source_dir)
        if not os.path.exists(source_path):
            print(f"Source directory not found: {source_path}")
            continue

        svg_files = list(Path(source_path).rglob("*.svg"))
        print(f"Found {len(svg_files)} SVG files in {source_dir}")
        all_files.extend(svg_files)

    if not all_files:
        print("\nKhông tìm thấy file SVG nào trong các source_dirs đã cho.")
        return

    # --- Tính total_paths cho TOÀN BỘ file (gộp mọi thư mục) ---
    files_with_counts = [(f, count_total_paths(f)) for f in all_files]

    # --- Chia bucket theo quantile của total_paths trên toàn bộ dataset ---
    buckets = assign_buckets(files_with_counts, n_buckets=n_buckets)

    print(f"\nStratify theo total_paths trên toàn bộ {len(all_files)} file "
          f"({n_buckets} bucket):")
    for bucket_id in sorted(buckets.keys()):
        bucket_files = buckets[bucket_id]
        label = "unknown" if bucket_id == -1 else f"bucket {bucket_id}"
        counts = [c for f, c in files_with_counts if f in bucket_files]
        rng = f"{min(counts)}-{max(counts)}" if counts and bucket_id != -1 else "n/a"
        print(f"  - {label:<10} ({len(bucket_files)} files, total_paths={rng})")

    # --- Chọn seed & thực hiện chia ---
    if auto_seed:
        best_seed, best_diff, best_split = find_best_seed(
            buckets, files_with_counts, train_ratio, base_seed=seed, n_trials=n_trials
        )
        print(f"\n[auto-seed] Đã thử {n_trials} seed, chọn seed={best_seed} "
              f"(chênh lệch mean = {best_diff*100:.1f}%)")
        all_train_files, all_val_files = best_split
    else:
        all_train_files, all_val_files = try_split_for_seed(buckets, train_ratio, seed)
        print(f"\nDùng seed cố định: {seed}")

    print(f"\nTotal across all directories:")
    print(f"Train: {len(all_train_files)} files")
    print(f"Val: {len(all_val_files)} files")

    # --- Copy files ---
    for svg_file in all_train_files:
        shutil.copy2(svg_file, os.path.join(train_dir, svg_file.name))

    for svg_file in all_val_files:
        shutil.copy2(svg_file, os.path.join(val_dir, svg_file.name))

    # --- Kiểm tra nhanh kết quả phân bố (để đối chiếu, không bắt buộc) ---
    print_distribution_check(all_train_files, all_val_files)

    print(f"\nCompleted!")
    print(f"Train files saved to: {train_dir}")
    print(f"Val files saved to: {val_dir}")


def print_distribution_check(train_files: list, val_files: list):
    """In nhanh mean/std của total_paths ở train vs val để xác nhận đã đồng đều hơn."""
    def stats(files):
        counts = [c for f, c in ((f, count_total_paths(f)) for f in files) if c >= 0]
        if not counts:
            return 0, 0, 0
        mean = sum(counts) / len(counts)
        var = sum((c - mean) ** 2 for c in counts) / len(counts)
        return mean, var ** 0.5, len(counts)

    train_mean, train_std, train_n = stats(train_files)
    val_mean, val_std, val_n = stats(val_files)

    print(f"\n{'='*50}")
    print("  KIỂM TRA PHÂN BỐ SAU KHI CHIA")
    print(f"{'='*50}")
    print(f"  Train (n={train_n}): mean total_paths = {train_mean:.1f}, std = {train_std:.1f}")
    print(f"  Val   (n={val_n}): mean total_paths = {val_mean:.1f}, std = {val_std:.1f}")
    if train_mean > 0 and val_mean > 0:
        diff_pct = abs(train_mean - val_mean) / train_mean * 100
        print(f"  Chênh lệch mean: {diff_pct:.1f}%  "
              f"({'OK, khá đồng đều' if diff_pct < 20 else 'CẢNH BÁO: vẫn lệch đáng kể'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split SVG files from easy/medium/hard into train/val sets, "
                     "stratified theo độ phức tạp (total_paths)"
    )
    parser.add_argument("--source-dirs", nargs='+',
                         default=["data/logo/easy", "data/logo/medium", "data/logo/hard"],
                         help="Source directories containing SVG files")
    parser.add_argument("--train-dir", default="data/logo/train_svg",
                         help="Output directory for training set")
    parser.add_argument("--val-dir", default="data/logo/val_svg",
                         help="Output directory for validation set")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                         help="Ratio of training data (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--n-buckets", type=int, default=4,
                         help="Số bucket để stratify theo total_paths (default: 4, "
                              "tức chia theo quartile)")
    parser.add_argument("--auto-seed", action="store_true",
                         help="Tự động thử nhiều seed và chọn seed cho chênh lệch "
                              "mean(total_paths) train/val nhỏ nhất, thay vì dùng "
                              "đúng 1 seed cố định. Khuyến nghị dùng khi dataset nhỏ "
                              "hoặc có outlier lớn.")
    parser.add_argument("--n-trials", type=int, default=200,
                         help="Số seed thử khi dùng --auto-seed (default: 200)")

    args = parser.parse_args()

    train_dir = os.path.join(PROJECT_ROOT, args.train_dir)
    val_dir = os.path.join(PROJECT_ROOT, args.val_dir)

    split_svg_files(args.source_dirs, train_dir, val_dir,
                     args.train_ratio, args.seed, args.n_buckets,
                     auto_seed=args.auto_seed, n_trials=args.n_trials)