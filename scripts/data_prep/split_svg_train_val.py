#!/usr/bin/env python3
"""
Split SVG files from easy/medium/hard folders into train/val sets.

Khác với bản gốc (chỉ stratify theo total_paths):

  [THAY ĐỔI CHÍNH]
  Val set bắt buộc phải có cả fill lẫn satin trong mỗi file. File chỉ có
  fill hoặc chỉ có satin sẽ luôn rơi vào train — vì val chỉ toàn fill/satin
  khiến IoU satin bị lệch nghiêm trọng khi đánh giá.

  [STRATIFY]
  Không dùng total_paths đơn thuần nữa. Với nhóm "both" (có đủ fill+satin),
  bucket được tính theo satin_ratio = satin_paths / total_paths để đảm bảo
  train và val có tỉ lệ satin tương đương nhau.

Cách làm:
  1. Đọc từng SVG, đếm fill_paths và satin_paths riêng.
  2. Phân loại file thành 3 nhóm:
       - "both"   : có đủ fill_paths >= 1 VÀ satin_paths >= 1  → đủ điều kiện vào val
       - "fill_only" : chỉ có fill, không có satin              → chỉ vào train
       - "satin_only": chỉ có satin, không có fill              → chỉ vào train
       - "empty"  : không có path nào (lỗi hoặc file rỗng)     → chỉ vào train
  3. Với nhóm "both": stratify theo satin_ratio bucket → chia train/val theo
     train_ratio. Dùng auto-seed để cân bằng mean satin_ratio giữa 2 tập.
  4. Với mọi nhóm còn lại: toàn bộ vào train.
  5. In thống kê rõ ràng về thành phần val để dễ kiểm tra.
"""

import os
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


# ==========================
# Đếm fill/satin của 1 file
# ==========================

def count_paths(svg_path: Path) -> Tuple[int, int]:
    """
    Trả về (fill_paths, satin_paths) cho 1 file SVG.
    Trả về (-1, -1) nếu lỗi parse.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  [WARN] Không đọc được {svg_path.name}: {e}")
        return -1, -1

    fill_count = 0
    satin_count = 0
    for elem in root.findall(".//svg:path", SVG_NS):
        label = elem.attrib.get(INKSCAPE_LABEL, "").strip().lower()
        if label == "fill":
            fill_count += 1
        elif label == "satin":
            satin_count += 1
    return fill_count, satin_count


def classify_file(fill: int, satin: int) -> str:
    """Phân loại file theo loại path có trong đó."""
    if fill < 0:          return "error"
    if fill > 0 and satin > 0: return "both"
    if fill > 0:          return "fill_only"
    if satin > 0:         return "satin_only"
    return "empty"


# ==========================
# Chia bucket theo satin_ratio
# ==========================

def assign_satin_ratio_buckets(files_meta: List[Tuple[Path, int, int]],
                                n_buckets: int = 4) -> Dict[int, list]:
    """
    files_meta: list of (Path, fill_paths, satin_paths)
    Tính satin_ratio = satin / (fill + satin), chia bucket theo quantile.
    Chỉ nhận file thuộc nhóm "both" (fill >= 1 và satin >= 1).
    """
    valid = [(f, s / (fi + s)) for f, fi, s in files_meta if fi >= 1 and s >= 1]
    if not valid:
        return {}

    valid_sorted = sorted(valid, key=lambda x: x[1])
    n = len(valid_sorted)
    n_buckets = max(1, min(n_buckets, n))

    buckets: Dict[int, list] = {}
    for i, (f, ratio) in enumerate(valid_sorted):
        bucket_id = min(i * n_buckets // n, n_buckets - 1)
        buckets.setdefault(bucket_id, []).append((f, ratio))
    return buckets


# ==========================
# Seed tìm kiếm
# ==========================

def try_split_buckets(buckets: Dict[int, list], train_ratio: float,
                       seed: int) -> Tuple[list, list]:
    """Chia train/val từ buckets với seed cho trước. Trả về (train, val)."""
    rng = random.Random(seed)
    train_files, val_files = [], []
    for bid in sorted(buckets.keys()):
        bucket = [item for item in buckets[bid]]
        rng.shuffle(bucket)
        split_idx = max(1, round(len(bucket) * train_ratio))
        # Đảm bảo val luôn có ít nhất 1 file nếu bucket đủ lớn
        if len(bucket) > 1:
            split_idx = min(split_idx, len(bucket) - 1)
        train_files.extend(f for f, _ in bucket[:split_idx])
        val_files.extend(f for f, _ in bucket[split_idx:])
    return train_files, val_files


def find_best_seed(buckets: Dict[int, list], train_ratio: float,
                    base_seed: int, n_trials: int = 200) -> Tuple[int, float, Tuple[list, list]]:
    """
    Thử n_trials seed, chọn seed cho chênh lệch mean(satin_ratio) nhỏ nhất.
    """
    ratio_map = {f: r for items in buckets.values() for f, r in items}

    best_seed = base_seed
    best_diff = float("inf")
    best_split = None

    for trial in range(n_trials):
        seed = base_seed + trial
        train_files, val_files = try_split_buckets(buckets, train_ratio, seed)

        def mean_ratio(files):
            ratios = [ratio_map[f] for f in files if f in ratio_map]
            return sum(ratios) / len(ratios) if ratios else 0.0

        tr = mean_ratio(train_files)
        vr = mean_ratio(val_files)
        if tr == 0:
            continue
        diff = abs(tr - vr) / tr
        if diff < best_diff:
            best_diff = diff
            best_seed = seed
            best_split = (train_files, val_files)

    return best_seed, best_diff, best_split


# ==========================
# Clear dir
# ==========================

def clear_dir(folder: str):
    folder = Path(folder)
    if not folder.exists():
        return
    removed = sum(1 for f in folder.glob("*.svg") if f.unlink() is None)
    if removed:
        print(f"  Đã xoá {removed} file .svg cũ trong {folder}")


# ==========================
# Check overlap
# ==========================

def check_no_overlap(train_dir: str, val_dir: str):
    train_names = {f.name for f in Path(train_dir).glob("*.svg")}
    val_names   = {f.name for f in Path(val_dir).glob("*.svg")}
    overlap = train_names & val_names
    if overlap:
        print(f"\n  !!! CẢNH BÁO: {len(overlap)} file trùng tên giữa train và val !!!")
        for name in sorted(overlap):
            print(f"     - {name}")
    else:
        print(f"\n  [OK] Không có file trùng giữa train ({len(train_names)}) "
              f"và val ({len(val_names)}) file.")


# ==========================
# Main
# ==========================

def split_svg_files(source_dirs: List[str], train_dir: str, val_dir: str,
                     train_ratio: float = 0.7, seed: int = 42, n_buckets: int = 4,
                     auto_seed: bool = False, n_trials: int = 200,
                     min_satin_for_val: int = 1, min_fill_for_val: int = 1):
    """
    Split SVG files with val-set guarantee: mỗi file val phải có ít nhất
    min_fill_for_val fill paths VÀ min_satin_for_val satin paths.

    Args:
        min_satin_for_val: số satin paths tối thiểu để file được vào val (mặc định 1)
        min_fill_for_val : số fill paths tối thiểu để file được vào val (mặc định 1)
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print("Dọn dẹp thư mục đích trước khi chia lại...")
    clear_dir(train_dir)
    clear_dir(val_dir)

    all_files: List[Path] = []
    for source_dir in source_dirs:
        source_path = os.path.join(PROJECT_ROOT, source_dir)
        if not os.path.exists(source_path):
            print(f"Source directory not found: {source_path}")
            continue
        svg_files = list(Path(source_path).rglob("*.svg"))
        print(f"Found {len(svg_files)} SVG files in {source_dir}")
        all_files.extend(svg_files)

    if not all_files:
        print("\nKhông tìm thấy file SVG nào.")
        return

    # --- Đếm fill/satin cho từng file ---
    print(f"\nĐang đọc metadata ({len(all_files)} files)...")
    files_meta: List[Tuple[Path, int, int]] = []
    for f in all_files:
        fill, satin = count_paths(f)
        files_meta.append((f, fill, satin))

    # --- Phân loại ---
    groups: Dict[str, List[Tuple[Path, int, int]]] = {
        "both": [], "fill_only": [], "satin_only": [], "empty": [], "error": []
    }
    for f, fill, satin in files_meta:
        kind = classify_file(fill, satin)
        # Áp dụng ngưỡng tối thiểu cho "both"
        if kind == "both" and (fill < min_fill_for_val or satin < min_satin_for_val):
            kind = "fill_only" if satin < min_satin_for_val else "satin_only"
        groups[kind].append((f, fill, satin))

    print(f"\n{'='*55}")
    print("  PHÂN LOẠI FILE THEO NỘI DUNG")
    print(f"{'='*55}")
    for kind, items in groups.items():
        if not items:
            continue
        fills  = [fi for _, fi, s in items if fi >= 0]
        satins = [s  for _, fi, s in items if fi >= 0]
        f_rng = f"fill={min(fills)}-{max(fills)}" if fills else ""
        s_rng = f"satin={min(satins)}-{max(satins)}" if satins else ""
        print(f"  {kind:<12}: {len(items):>3} files  {f_rng}  {s_rng}")

    val_eligible = groups["both"]   # chỉ nhóm này được vào val
    train_forced = (groups["fill_only"] + groups["satin_only"]
                    + groups["empty"] + groups["error"])  # luôn vào train

    print(f"\n  → Đủ điều kiện vào val : {len(val_eligible)} files")
    print(f"  → Bắt buộc vào train   : {len(train_forced)} files "
          f"(fill_only + satin_only + empty + error)")

    if not val_eligible:
        print("\nCẢNH BÁO: Không có file nào đủ điều kiện vào val "
              "(cần có cả fill lẫn satin). Toàn bộ vào train.")
        for f, _, _ in all_files:
            shutil.copy2(f, os.path.join(train_dir, f.name))
        return

    # --- Stratify nhóm "both" theo satin_ratio ---
    buckets = assign_satin_ratio_buckets(val_eligible, n_buckets=n_buckets)

    print(f"\n  Stratify {len(val_eligible)} file 'both' theo satin_ratio "
          f"({n_buckets} bucket):")
    for bid in sorted(buckets.keys()):
        items = buckets[bid]
        ratios = [r for _, r in items]
        print(f"    bucket {bid}: {len(items)} files, "
              f"satin_ratio={min(ratios):.2f}-{max(ratios):.2f}")

    # --- Chọn seed & chia ---
    if auto_seed:
        best_seed, best_diff, best_split = find_best_seed(
            buckets, train_ratio, base_seed=seed, n_trials=n_trials
        )
        print(f"\n  [auto-seed] seed={best_seed}, "
              f"chênh lệch mean satin_ratio = {best_diff*100:.1f}%")
        both_train, both_val = best_split
    else:
        both_train, both_val = try_split_buckets(buckets, train_ratio, seed)
        print(f"\n  Dùng seed cố định: {seed}")

    # --- Gộp train ---
    forced_train_paths = [f for f, _, _ in train_forced]
    all_train = both_train + forced_train_paths
    all_val   = both_val

    print(f"\n{'='*55}")
    print("  KẾT QUẢ CHIA")
    print(f"{'='*55}")
    print(f"  Train: {len(all_train)} files  "
          f"(both_train={len(both_train)}, forced={len(forced_train_paths)})")
    print(f"  Val  : {len(all_val)} files  (đều có fill+satin)")

    # --- In thống kê val để xác nhận ---
    val_meta = {f: (fi, s) for f, fi, s in files_meta}
    val_fills  = [val_meta[f][0] for f in all_val if f in val_meta]
    val_satins = [val_meta[f][1] for f in all_val if f in val_meta]
    if val_fills:
        print(f"\n  Val fill_paths  : min={min(val_fills)}, "
              f"max={max(val_fills)}, mean={sum(val_fills)/len(val_fills):.1f}")
        print(f"  Val satin_paths : min={min(val_satins)}, "
              f"max={max(val_satins)}, mean={sum(val_satins)/len(val_satins):.1f}")
        val_ratios = [s/(fi+s) for fi, s in zip(val_fills, val_satins)]
        print(f"  Val satin_ratio : min={min(val_ratios):.2f}, "
              f"max={max(val_ratios):.2f}, mean={sum(val_ratios)/len(val_ratios):.2f}")

    # --- Copy files ---
    for f in all_train:
        shutil.copy2(f, os.path.join(train_dir, f.name))
    for f in all_val:
        shutil.copy2(f, os.path.join(val_dir, f.name))

    check_no_overlap(train_dir, val_dir)

    print(f"\nCompleted!")
    print(f"Train files saved to: {train_dir}")
    print(f"Val files saved to  : {val_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split SVG files stratified theo satin_ratio, "
                    "đảm bảo val chỉ gồm file có cả fill lẫn satin."
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
                         help="Random seed (default: 42)")
    parser.add_argument("--n-buckets", type=int, default=4,
                         help="Số bucket stratify theo satin_ratio (default: 4)")
    parser.add_argument("--auto-seed", action="store_true",
                         help="Tự động chọn seed cân bằng mean satin_ratio train/val")
    parser.add_argument("--n-trials", type=int, default=200,
                         help="Số seed thử khi dùng --auto-seed (default: 200)")
    parser.add_argument("--min-satin-for-val", type=int, default=1,
                         help="Số satin paths tối thiểu để file vào val (default: 1)")
    parser.add_argument("--min-fill-for-val", type=int, default=1,
                         help="Số fill paths tối thiểu để file vào val (default: 1)")

    args = parser.parse_args()

    train_dir = os.path.join(PROJECT_ROOT, args.train_dir)
    val_dir   = os.path.join(PROJECT_ROOT, args.val_dir)

    split_svg_files(
        args.source_dirs, train_dir, val_dir,
        args.train_ratio, args.seed, args.n_buckets,
        auto_seed=args.auto_seed, n_trials=args.n_trials,
        min_satin_for_val=args.min_satin_for_val,
        min_fill_for_val=args.min_fill_for_val,
    )